"""
元学习模块：First-order episodic meta-learning for domain generalization

用于训练域适配模块（DomainFiLM + Reliability/Attention Gate），
使模型在面对新域时具有更强的泛化能力。

核心思想：
- 在源域(CWRU/PU)上进行episodic训练
- Inner loop: 在一个任务(domain)上适配几步
- Outer loop: 用另一个任务评估，更新元参数
- 只训练可适配模块，冻结主干网络

注意（稳定性）：
- 内循环如果直接“就地改 param.data”但不恢复，会导致训练过程参数漂移，极易引发不稳定/坍塌。
- 本实现会在 first_order=True 时也恢复到 initial，再用 (initial - adapted) 作为 meta-gradient 做更新。

基于 First-order MAML / Reptile 的简化实现
"""

import torch
import torch.nn.functional as F


class MetaLearner:
    """
    First-order episodic meta-learning for domain generalization
    """
    def __init__(
        self,
        meta_params,  # 需要元学习的参数（dict: name -> param）
        meta_lr=1e-4,  # 元学习率（outer loop）
        inner_lr=1e-3,  # 内循环学习率
        inner_steps=3,  # 内循环步数
        first_order=True  # 是否使用一阶近似（True更快，内存开销小）
    ):
        """
        Args:
            meta_params: dict of {name: parameter} 需要元学习的参数
            meta_lr: outer loop学习率
            inner_lr: inner loop学习率
            inner_steps: 内循环适配步数
            first_order: 是否使用一阶近似（Reptile风格）
        """
        self.meta_params = meta_params
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        
        # 创建元优化器（用于outer loop更新）
        self.meta_optimizer = torch.optim.Adam(
            meta_params.values(), 
            lr=meta_lr
        )
    
    def clone_params(self):
        """克隆当前参数（用于内循环）"""
        cloned = {}
        for name, param in self.meta_params.items():
            cloned[name] = param.clone().detach().requires_grad_(True)
        return cloned
    
    def compute_task_loss(
        self,
        batch,
        signal_encoder,
        spec_encoder,
        text_encoder,
        fusion_module,
        fusion_decoder,
        device,
        text_prototypes=None,
        use_focal_loss=False,
        focal_gamma=2.0,
        precomputed_class_weights=None,
        label_smoothing=0.0,
        use_moe=False
    ):
        """
        计算单个batch的任务损失
        
        Args:
            batch: 数据batch
            其他：模型组件和配置
            
        Returns:
            loss: 标量损失
        """
        signals = batch['signal'].to(device)
        specs = batch['spec'].to(device)  # 修正：使用'spec'而不是'spectrogram'
        signal_mask = batch['signal_mask'].to(device)
        spec_mask = batch['spec_mask'].to(device)
        labels = batch['label'].to(device)
        texts = batch.get('text', None)
        domains = batch.get('domain', None)
        conditions = batch.get('condition', None)
        
        # 编码domain_ids（用于域残差）
        domain_ids = None
        if domains is not None:
            try:
                import config as cfg
                cwru_gk2_id = int(getattr(cfg, "CWRU_GK2_DOMAIN_ID", 0))
                cwru_gk4_id = int(getattr(cfg, "CWRU_GK4_DOMAIN_ID", 1))
                pu_id = int(getattr(cfg, "PU_DOMAIN_ID", 2))
                hust_id = int(getattr(cfg, "HUST_DOMAIN_ID", 3))
                hit_id = int(getattr(cfg, "HIT_DOMAIN_ID", 4))
            except Exception:
                cwru_gk2_id, cwru_gk4_id, pu_id, hust_id, hit_id = 0, 1, 2, 3, 4
            _encoded_domains = []
            for i, d in enumerate(domains):
                d_up = str(d).upper()
                cond = str(conditions[i]).lower() if (conditions is not None and i < len(conditions)) else ""
                if d_up == "CWRU":
                    if "gk4" in cond or "gk_4" in cond or "hp3" in cond:
                        _encoded_domains.append(cwru_gk4_id)
                    else:
                        _encoded_domains.append(cwru_gk2_id)
                elif d_up == "PU":
                    _encoded_domains.append(pu_id)
                elif d_up == "HUST":
                    _encoded_domains.append(hust_id)
                elif d_up == "HIT":
                    _encoded_domains.append(hit_id)
                else:
                    _encoded_domains.append(cwru_gk2_id)
            domain_ids = torch.tensor(_encoded_domains, dtype=torch.long).to(device)
        
        # 编码text（如果需要）
        z_text = None
        if text_encoder is not None and texts is not None:
            z_text = text_encoder(texts)
        
        # 前向传播 - Signal Encoder
        use_signal_classification_head = getattr(signal_encoder, 'use_classification_head', False)
        if hasattr(signal_encoder, 'use_domain_residual') and signal_encoder.use_domain_residual:
            if use_signal_classification_head:
                signal_feat, signal_logits = signal_encoder(signals, signal_mask, domain_ids=domain_ids, return_logits=True)
            else:
                signal_feat = signal_encoder(signals, signal_mask, domain_ids=domain_ids)
                signal_logits = None
        else:
            if use_signal_classification_head:
                signal_feat, signal_logits = signal_encoder(signals, signal_mask, return_logits=True)
            else:
                signal_feat = signal_encoder(signals, signal_mask)
                signal_logits = None
        
        # 前向传播 - Spec Encoder
        use_spec_classification_head = getattr(spec_encoder, 'use_classification_head', False)
        if use_moe and hasattr(spec_encoder, 'num_bands'):
            # MoE版本
            if use_spec_classification_head:
                # 当 return_moe_stats=False 时，只返回 (z_spec, logits)
                spec_feat, spec_logits = spec_encoder(
                    specs, spec_mask,
                    text_emb=z_text,
                    domain_ids=domain_ids,
                    return_moe_stats=False,  # 元学习不需要统计信息
                    return_logits=True
                )
                moe_stats = None
            else:
                # 当 return_moe_stats=False 时，只返回 z_spec
                spec_feat = spec_encoder(
                    specs, spec_mask,
                    text_emb=z_text,
                    domain_ids=domain_ids,
                    return_moe_stats=False
                )
                spec_logits = None
                moe_stats = None
        else:
            # 普通版本
            if hasattr(spec_encoder, 'use_domain_residual') and spec_encoder.use_domain_residual:
                if use_spec_classification_head:
                    spec_feat, spec_logits = spec_encoder(specs, spec_mask, domain_ids=domain_ids, return_logits=True)
                else:
                    spec_feat = spec_encoder(specs, spec_mask, domain_ids=domain_ids)
                    spec_logits = None
            else:
                if use_spec_classification_head:
                    spec_feat, spec_logits = spec_encoder(specs, spec_mask, return_logits=True)
                else:
                    spec_feat = spec_encoder(specs, spec_mask)
                    spec_logits = None
        
        # Fusion - 根据fusion_module类型使用正确的参数
        if hasattr(fusion_module, 'domain_embedding'):
            # DomainAwareCrossAttentionFusion
            z_fusion = fusion_module(
                signal_feat,
                spec_feat,
                domain_ids=domain_ids,
                text_prototypes=text_prototypes,
                class_ids=None,
                class_probs=None,
                signal_logits=signal_logits,
                spec_logits=spec_logits
            )
        else:
            # CrossGatedFusion或其他类型
            z_fusion = fusion_module(signal_feat, spec_feat)
        
        # Decoder
        # IMPORTANT: keep the same forward path as training/evaluation (domain-aware + prototype head).
        logits = fusion_decoder(z_fusion, domain_ids=domain_ids, text_prototypes=text_prototypes)
        
        # 计算损失
        if use_focal_loss:
            ce_loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=label_smoothing)
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** focal_gamma
            
            if precomputed_class_weights is not None:
                alpha_t = precomputed_class_weights[labels]
                focal_weight = alpha_t * focal_weight
            
            loss = (focal_weight * ce_loss).mean()
        else:
            weight_tensor = precomputed_class_weights if precomputed_class_weights is not None else None
            loss = F.cross_entropy(logits, labels, weight=weight_tensor, label_smoothing=label_smoothing)
        
        return loss
    
    def meta_step(
        self,
        support_batch,  # 任务A的支持集（用于inner loop适配）
        query_batch,    # 任务B的查询集（用于outer loop评估）
        signal_encoder,
        spec_encoder,
        text_encoder,
        fusion_module,
        fusion_decoder,
        device,
        text_prototypes=None,
        **loss_kwargs  # 传递给compute_task_loss的其他参数
    ):
        """
        执行一次meta-learning步骤（episodic training）
        
        Args:
            support_batch: 支持集batch（任务A，用于适配）
            query_batch: 查询集batch（任务B，用于评估）
            其他：模型组件
            
        Returns:
            meta_loss: 元损失（用于监控）
        """
        # 1) Save initial meta-parameter values (for restore + reptile direction)
        initial_data = {name: p.detach().clone() for name, p in self.meta_params.items()}
        
        # 2) Inner loop: adapt on support batch
        for _ in range(self.inner_steps):
            # 计算support loss
            support_loss = self.compute_task_loss(
                support_batch,
                signal_encoder,
                spec_encoder,
                text_encoder,
                fusion_module,
                fusion_decoder,
                device,
                text_prototypes=text_prototypes,
                **loss_kwargs
            )
            
            # 计算梯度
            grads = torch.autograd.grad(
                support_loss,
                self.meta_params.values(),
                create_graph=(not self.first_order),
                allow_unused=True
            )
            
            # Inner loop更新（临时修改参数）
            for (name, param), grad in zip(self.meta_params.items(), grads):
                if grad is not None:
                    param.data = param.data - self.inner_lr * grad

        # Snapshot adapted parameters
        adapted_data = {name: p.detach().clone() for name, p in self.meta_params.items()}
        
        # 3. Outer loop: 在query_batch上评估
        query_loss = self.compute_task_loss(
            query_batch,
            signal_encoder,
            spec_encoder,
            text_encoder,
            fusion_module,
            fusion_decoder,
            device,
            text_prototypes=text_prototypes,
            **loss_kwargs
        )
        
        # 4) Meta update (outer loop)
        if self.first_order:
            # Reptile-style (first-order):
            #   theta <- theta - meta_lr * (theta - theta')
            # We implement this by restoring theta=initial and using grad = (initial - adapted).
            self.meta_optimizer.zero_grad()
            for name, param in self.meta_params.items():
                # Restore initial weights before applying the meta step
                param.data = initial_data[name].data
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.data = (initial_data[name] - adapted_data[name]).to(dtype=param.dtype, device=param.device)
            self.meta_optimizer.step()
        else:
            # MAML风格：保留计算图，反向传播query loss
            self.meta_optimizer.zero_grad()
            query_loss.backward()
            self.meta_optimizer.step()
        
        # 5) Restore (safety): for any path that may have left adapted params in place
        if not self.first_order:
            for name, param in self.meta_params.items():
                param.data = initial_data[name].data
        
        return query_loss.item()


def collect_meta_parameters(signal_encoder, fusion_module):
    """
    收集需要元学习的参数
    
    只训练：
    - signal_encoder.domain_residual (DomainFiLM)
    - fusion_module.reliability_gate (if exists)
    - fusion_module.domain_embedding (可选)
    
    Args:
        signal_encoder: SignalEncoder
        fusion_module: FusionModule
        
    Returns:
        dict of {name: parameter}
    """
    meta_params = {}
    
    # 1. DomainFiLM参数
    if hasattr(signal_encoder, 'domain_residual') and signal_encoder.domain_residual is not None:
        for name, param in signal_encoder.domain_residual.named_parameters():
            meta_params[f'signal_encoder.domain_residual.{name}'] = param
    
    # 2. Reliability Gate参数
    if hasattr(fusion_module, 'reliability_gate') and fusion_module.reliability_gate is not None:
        for name, param in fusion_module.reliability_gate.named_parameters():
            meta_params[f'fusion_module.reliability_gate.{name}'] = param
    
    # 3. Domain embedding（可选）
    if hasattr(fusion_module, 'domain_embedding') and fusion_module.domain_embedding is not None:
        for name, param in fusion_module.domain_embedding.named_parameters():
            meta_params[f'fusion_module.domain_embedding.{name}'] = param
    
    return meta_params


def sample_meta_tasks(train_loader, num_tasks=2, samples_per_task=32):
    """
    从train_loader中采样meta-learning任务
    
    策略：从不同domain采样batch作为不同任务
    
    Args:
        train_loader: 训练DataLoader（包含CWRU+PU混合数据）
        num_tasks: 采样任务数（2表示taskA和taskB）
        samples_per_task: 每个任务的样本数
        
    Returns:
        list of batches（每个batch代表一个任务）
    """
    # 简化实现：从train_loader中随机采样多个batch
    # 更复杂的实现可以根据domain分组采样
    
    tasks = []
    iterator = iter(train_loader)
    
    for _ in range(num_tasks):
        try:
            batch = next(iterator)
            tasks.append(batch)
        except StopIteration:
            # 如果iterator耗尽，重新创建
            iterator = iter(train_loader)
            batch = next(iterator)
            tasks.append(batch)
    
    return tasks
