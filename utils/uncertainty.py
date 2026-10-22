"""
不确定性估计模块
用于计算模态嵌入的不确定性，指导可靠性门控
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyEstimator(nn.Module):
    """
    多种不确定性估计方法的统一接口
    支持：熵、能量、特征方差、MC Dropout
    """
    def __init__(self, method='entropy', num_classes=4, temperature=1.0):
        """
        Args:
            method: 'entropy' | 'energy' | 'variance' | 'mc_dropout'
            num_classes: 类别数（用于熵计算）
            temperature: 温度参数（用于能量计算）
        """
        super().__init__()
        self.method = method
        self.num_classes = num_classes
        self.temperature = temperature
    
    def forward(self, embeddings, logits=None, feature_variance=None):
        """
        计算不确定性分数（越高=越不确定）
        
        Args:
            embeddings: [B, D] 模态嵌入
            logits: [B, C] 可选，用于熵/能量计算
            feature_variance: [B] 可选，深度特征方差
        
        Returns:
            uncertainty: [B] 不确定性分数（归一化到[0,1]）
        """
        if self.method == 'entropy':
            return self._compute_entropy(logits)
        elif self.method == 'energy':
            return self._compute_energy(logits)
        elif self.method == 'variance':
            return self._compute_variance(embeddings, feature_variance)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.method}")
    
    def _compute_entropy(self, logits):
        """
        计算预测熵（Shannon entropy）
        H = -sum(p * log(p))
        """
        if logits is None:
            raise ValueError("logits required for entropy computation")
        
        probs = F.softmax(logits, dim=-1)  # [B, C]
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]
        
        # 归一化到[0,1]: max_entropy = log(num_classes)
        max_entropy = torch.log(torch.tensor(self.num_classes, device=logits.device))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def _compute_energy(self, logits):
        """
        计算能量分数（Energy-based OOD, NeurIPS'20）
        E = -T * log(sum(exp(logits/T)))
        低能量=高置信度，高能量=低置信度
        """
        if logits is None:
            raise ValueError("logits required for energy computation")
        
        # Energy score (越低=越确定)
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)  # [B]
        
        # 归一化到[0,1]并反转（使高能量=高不确定性）
        # 使用sigmoid映射到[0,1]
        normalized_energy = torch.sigmoid(-energy / self.temperature)
        
        return normalized_energy
    
    def _compute_variance(self, embeddings, feature_variance=None):
        """
        计算特征方差作为不确定性
        """
        if feature_variance is not None:
            # 使用预计算的方差
            uncertainty = feature_variance
        else:
            # 计算嵌入维度的方差
            uncertainty = embeddings.var(dim=-1)  # [B]
        
        # 归一化到[0,1]
        uncertainty = torch.sigmoid(uncertainty)
        
        return uncertainty


class ReliabilityGate(nn.Module):
    """
    基于不确定性的可靠性门控
    动态调整模态权重，抑制不可靠模态
    """
    def __init__(self, num_modalities=2, gate_type='soft', temperature=1.0, 
                 learnable_temp=False, init_bias=0.0):
        """
        Args:
            num_modalities: 模态数量
            gate_type: 'soft' | 'hard' | 'learned'
            temperature: 门控温度（控制软硬程度）
            learnable_temp: 是否学习温度参数
            init_bias: 初始偏置（正值=倾向信任，负值=倾向不信任）
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.gate_type = gate_type
        
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # 可学习的偏置（允许模型调整对不确定性的敏感度）
        self.bias = nn.Parameter(torch.full((num_modalities,), init_bias))
        
        # 如果是learned类型，添加小型MLP
        if gate_type == 'learned':
            self.gate_mlp = nn.Sequential(
                nn.Linear(num_modalities, num_modalities * 2),
                nn.ReLU(),
                nn.Linear(num_modalities * 2, num_modalities),
            )
    
    def forward(self, uncertainties):
        """
        根据不确定性计算模态权重
        
        Args:
            uncertainties: [B, M] 各模态的不确定性分数
        
        Returns:
            weights: [B, M] 归一化的模态权重
            gate_info: dict 包含中间信息（用于监控）
            reg_loss: None (普通 ReliabilityGate 不提供正则损失)
        """
        # 不确定性越高，原始分数越低
        reliability_scores = 1.0 - uncertainties + self.bias  # [B, M]
        
        if self.gate_type == 'soft':
            # Softmax门控（温度控制软硬）
            weights = F.softmax(reliability_scores / self.temperature, dim=-1)
        
        elif self.gate_type == 'hard':
            # Gumbel-Softmax（可微的硬门控）
            weights = F.gumbel_softmax(reliability_scores / self.temperature, 
                                       tau=self.temperature, hard=True, dim=-1)
        
        elif self.gate_type == 'learned':
            # 学习的非线性门控
            gate_logits = self.gate_mlp(reliability_scores)
            weights = F.softmax(gate_logits / self.temperature, dim=-1)
        
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")
        
        # 收集监控信息
        gate_info = {
            'uncertainties': uncertainties.detach(),
            'reliability_scores': reliability_scores.detach(),
            'weights': weights.detach(),
            'temperature': self.temperature.item() if isinstance(self.temperature, nn.Parameter) else self.temperature,
            'bias': self.bias.detach(),
        }
        
        return weights, gate_info, None  # reg_loss = None for non-attention gate


class AttentionEnhancedReliabilityGate(nn.Module):
    """
    基于注意力增强的可靠性门控
    结合不确定性分数和模态embedding特征，通过self-attention评估模态质量
    
    关键特性：
    1. 保持原有不确定性门控机制（向后兼容）
    2. 使用self-attention评估模态特征质量
    3. 混合两种机制，初始时完全使用原始gate（不影响现有训练）
    """
    def __init__(
        self, 
        num_modalities=2, 
        embed_dim=512,
        gate_type='soft', 
        temperature=1.0, 
        learnable_temp=False, 
        init_bias=0.0,
        use_attention=True,  # 是否使用注意力机制
        attention_dim=128,   # 注意力特征维度
        attention_heads=2,   # 注意力头数
        attention_dropout=0.1,
        attention_weight_alpha=0.0,  # 注意力权重占比（0=完全使用原始gate，1=完全使用attention）
        # 条件化floor相关参数
        use_conditional_floor=True,  # 是否使用条件化floor（基于domain_embedding）
        domain_embed_dim=64,  # domain embedding维度
        floor_max=0.10,  # floor的最大值（训练初期较保守）
        floor_min=0.0,   # floor的最小值（允许完全抑制坏模态）
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate_type = gate_type
        self.embed_dim = embed_dim
        self.use_attention = use_attention
        self.attention_weight_alpha = attention_weight_alpha  # 可动态调整
        self.use_conditional_floor = use_conditional_floor
        self.floor_max = floor_max
        self.floor_min = floor_min
        
        # 原始gate（保留原有机制）
        if learnable_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        self.bias = nn.Parameter(torch.full((num_modalities,), init_bias))
        
        if gate_type == 'learned':
            self.gate_mlp = nn.Sequential(
                nn.Linear(num_modalities, num_modalities * 2),
                nn.ReLU(),
                nn.Linear(num_modalities * 2, num_modalities),
            )
        
        # ========= 注意力机制（可选） =========
        if use_attention:
            # 特征压缩：将embedding维度压缩到attention_dim
            self.embed_proj = nn.Sequential(
                nn.Linear(embed_dim, attention_dim),
                nn.LayerNorm(attention_dim),
                nn.ReLU(),
                nn.Dropout(attention_dropout)
            )
            
            # Self-Attention：让模态互相评估质量
            self.attention = nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=attention_heads,
                dropout=attention_dropout,
                batch_first=True
            )
            
            # 注意力特征到权重的映射
            self.attention_to_weights = nn.Sequential(
                nn.Linear(attention_dim, attention_dim // 2),
                nn.ReLU(),
                nn.Dropout(attention_dropout),
                nn.Linear(attention_dim // 2, 1)  # 每个模态输出一个权重分数
            )
    
        # ========= 条件化floor网络（可选） =========
        if use_conditional_floor:
            # 输入：domain_emb [B, domain_embed_dim] + uncertainties [B, num_modalities] + (可选) moe_disagreement [B, 1]
            # 输出：每模态floor [B, num_modalities]
            # 最大输入维度：domain_embed_dim + num_modalities + 1（当有moe_disagreement时）
            floor_input_dim_base = domain_embed_dim + num_modalities
            floor_input_dim_max = floor_input_dim_base + 1  # 如果包含moe_disagreement
            # 使用最大维度，兼容有无moe_disagreement的情况
            self.floor_net = nn.Sequential(
                nn.Linear(floor_input_dim_max, (floor_input_dim_max + num_modalities) // 2),
                nn.LayerNorm((floor_input_dim_max + num_modalities) // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear((floor_input_dim_max + num_modalities) // 2, num_modalities),
                nn.Sigmoid()  # 输出[0,1]，然后缩放到[floor_min, floor_max]
            )
            self.floor_input_dim_base = floor_input_dim_base
        else:
            self.floor_net = None
    
    def forward(self, uncertainties, modality_embeddings=None, domain_emb=None, spec_moe_disagreement=None):
        """
        结合不确定性和模态embedding计算权重
        
        Args:
            uncertainties: [B, M] 各模态的不确定性分数
            modality_embeddings: list of [B, D] 可选，各模态的embedding
            domain_emb: [B, domain_embed_dim] 可选，domain embedding（用于条件化floor）
            spec_moe_disagreement: [B] 可选，spec模态的MoE专家歧义度（TODO 3会使用）
            
        Returns:
            weights: [B, M] 归一化的模态权重
            gate_info: dict 包含中间信息（用于监控）
            reg_loss: tensor 可反传的正则损失（用于训练），如果未启用attention则为None
        """
        B, M = uncertainties.shape
        device = uncertainties.device
        
        # 1. 将MoE disagreement融合到spec_uncertainty（如果提供）
        # 方式A：将disagreement作为额外的uncertainty信号
        if spec_moe_disagreement is not None:
            # 归一化disagreement到[0,1]范围（假设disagreement已经在合理范围内，否则需要sigmoid）
            spec_disagreement_norm = torch.clamp(spec_moe_disagreement, 0.0, 1.0)  # [B]
            # 融合：spec_uncertainty = 0.7 * original + 0.3 * disagreement
            # 这样disagreement大时，spec_uncertainty也会增大，从而降低spec权重
            uncertainties_adjusted = uncertainties.clone()
            uncertainties_adjusted[:, 1] = 0.7 * uncertainties[:, 1] + 0.3 * spec_disagreement_norm
            uncertainties = uncertainties_adjusted
        
        # 1. 计算原始不确定性门控权重（保持原有机制）
        reliability_scores = 1.0 - uncertainties + self.bias  # [B, M]
        
        if self.gate_type == 'soft':
            uncertainty_weights = F.softmax(reliability_scores / self.temperature, dim=-1)
        elif self.gate_type == 'hard':
            uncertainty_weights = F.gumbel_softmax(
                reliability_scores / self.temperature, 
                tau=self.temperature, hard=True, dim=-1
            )
        elif self.gate_type == 'learned':
            gate_logits = self.gate_mlp(reliability_scores)
            uncertainty_weights = F.softmax(gate_logits / self.temperature, dim=-1)
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")
        
        # 2. 注意力增强（如果启用且有embedding）
        attention_weights = None
        attention_info = None
        reg_loss = None  # 可反传的正则损失
        
        if self.use_attention and modality_embeddings is not None and self.attention_weight_alpha > 0:
            # 投影到attention维度
            modality_tokens = []
            for emb in modality_embeddings:
                token = self.embed_proj(emb)  # [B, attention_dim]
                modality_tokens.append(token)
            
            # Stack: [B, M, attention_dim]
            modality_tokens = torch.stack(modality_tokens, dim=1)
            
            # Self-Attention：模态互相评估
            attended_tokens, attention_weights_raw = self.attention(
                query=modality_tokens,
                key=modality_tokens,
                value=modality_tokens
            )  # [B, M, attention_dim], [B, M, M]
            
            # 从attended tokens计算权重分数
            attention_scores = []
            for m in range(M):
                score = self.attention_to_weights(attended_tokens[:, m, :])  # [B, 1]
                attention_scores.append(score)
            attention_scores = torch.cat(attention_scores, dim=1)  # [B, M]
            
            # Softmax归一化
            attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)  # [B, M]
            
            # ===== 防坍缩约束：计算KL正则 =====
            # 使用uncertainty_weights作为teacher（detach，避免两者互相追逐）
            # 这能让attention学习到"细粒度修正"，但不会违背基本的可靠性判断
            teacher_weights = uncertainty_weights.detach()  # [B, M]
            
            # KL(att || teacher) = sum_m att[m] * (log(att[m]) - log(teacher[m]))
            # 逐样本计算KL，然后取平均
            eps = 1e-10
            kl_per_sample = torch.sum(
                attention_weights * (torch.log(attention_weights + eps) - torch.log(teacher_weights + eps)),
                dim=-1
            )  # [B]
            attention_kl_reg = kl_per_sample.mean()  # [1] (KL到uncertainty_weights)
            
            # 监控信息：也计算原始的entropy（用于日志）
            attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1).mean()  # [1]
            
            # ===== 关键：保持 reg_loss 可反传 =====
            reg_loss = attention_kl_reg  # 保持梯度，用于训练

            # 监控信息使用 detach 版本（不影响梯度）
            attention_info = {
                'attention_scores': attention_scores.detach(),
                'attention_weights': attention_weights.detach(),
                'attention_matrix': attention_weights_raw.detach(),  # [B, M, M]
                'attention_entropy': attention_entropy.detach(),      # [1] 当前注意力熵
                'attention_kl_reg': attention_kl_reg.detach(),        # [1] KL正则项（用于监控）
                'teacher_weights': teacher_weights.detach()           # [B, M] teacher分布（用于监控）
            }
        
        # 3. 混合权重（如果使用attention）
        if attention_weights is not None:
            alpha = self.attention_weight_alpha
            # 混合：final = (1-alpha) * uncertainty + alpha * attention
            final_weights = (1 - alpha) * uncertainty_weights + alpha * attention_weights
        else:
            # 仅使用不确定性权重（原始行为）
            final_weights = uncertainty_weights
        
        # 4. 防塌缩约束：施加weight floor（条件化或固定）
        if self.use_conditional_floor and self.floor_net is not None:
            # 条件化floor：基于domain_emb、uncertainties和可选 MoE disagreement
            if domain_emb is not None:
                # domain_emb: [B, domain_embed_dim], uncertainties: [B, M]
                floor_input_list = [domain_emb, uncertainties]
                # 如果有MoE disagreement，也加入（作为额外特征）
                if spec_moe_disagreement is not None:
                    # 将disagreement扩展为[B, 1]并加入到输入
                    floor_input_list.append(spec_moe_disagreement.unsqueeze(1))  # [B, 1]
                    floor_input = torch.cat(floor_input_list, dim=-1)  # [B, domain_embed_dim + M + 1]
                else:
                    floor_input = torch.cat(floor_input_list, dim=-1)  # [B, domain_embed_dim + M]
                    # Padding到最大维度（如果网络期望最大维度）
                    if hasattr(self, 'floor_input_dim_base'):
                        current_dim = floor_input.shape[-1]
                        max_dim = self.floor_input_dim_base + 1
                        if current_dim < max_dim:
                            # 添加零填充到最大维度
                            padding = torch.zeros(B, max_dim - current_dim, device=device)
                            floor_input = torch.cat([floor_input, padding], dim=-1)
                floor_raw = self.floor_net(floor_input)  # [B, M], [0,1]
                # 缩放到[floor_min, floor_max]
                weight_floor_per_modality = self.floor_min + (self.floor_max - self.floor_min) * floor_raw  # [B, M]
            else:
                # Fallback到固定floor
                weight_floor_per_modality = torch.full((B, M), self.floor_max, device=device)
        else:
            # 固定floor（向后兼容）
            weight_floor_per_modality = torch.full((B, M), 0.10, device=device)
        
        # 应用每模态的floor
        final_weights = torch.clamp(final_weights, min=weight_floor_per_modality)
        
        # 重新归一化（因为clamp可能破坏归一化）
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 收集监控信息
        gate_info = {
            'uncertainties': uncertainties.detach(),
            'reliability_scores': reliability_scores.detach(),
            'uncertainty_weights': uncertainty_weights.detach(),
            'final_weights': final_weights.detach(),
            'attention_weight_alpha': self.attention_weight_alpha,
            'temperature': self.temperature.item() if isinstance(self.temperature, nn.Parameter) else self.temperature,
            'bias': self.bias.detach(),
        }
        
        if attention_info is not None:
            gate_info.update(attention_info)
        
        return final_weights, gate_info, reg_loss
    
    def set_attention_weight(self, alpha):
        """
        动态设置注意力权重占比（用于渐进式训练）
        
        Args:
            alpha: float [0, 1]，0=完全使用不确定性gate，1=完全使用attention
        """
        self.attention_weight_alpha = max(0.0, min(1.0, alpha))


class ModalityDropout(nn.Module):
    """
    训练期随机丢弃模态，提升鲁棒性
    类似于Dropout，但作用在模态级别
    """
    def __init__(self, num_modalities=2, drop_prob=0.1, drop_mode='random'):
        """
        Args:
            num_modalities: 模态数量
            drop_prob: 丢弃概率
            drop_mode: 'random' | 'adaptive' (基于不确定性自适应丢弃)
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.drop_prob = drop_prob
        self.drop_mode = drop_mode
    
    def forward(self, embeddings_list, uncertainties=None):
        """
        随机丢弃模态
        
        Args:
            embeddings_list: list of [B, ...] 各模态嵌入
            uncertainties: [B, M] 可选，用于自适应丢弃
        
        Returns:
            masked_embeddings: list of [B, ...] 丢弃后的嵌入
            drop_mask: [B, M] 丢弃掩码（1=保留，0=丢弃）
        """
        if not self.training:
            # 评估时不丢弃
            batch_size = embeddings_list[0].size(0)
            drop_mask = torch.ones(batch_size, self.num_modalities, 
                                   device=embeddings_list[0].device)
            return embeddings_list, drop_mask
        
        batch_size = embeddings_list[0].size(0)
        device = embeddings_list[0].device
        
        if self.drop_mode == 'random':
            # 随机丢弃
            keep_prob = float(1.0 - self.drop_prob)
            keep_prob = max(0.0, min(1.0, keep_prob))
            drop_mask = torch.bernoulli(
                torch.full((batch_size, self.num_modalities), keep_prob, device=device)
            )
        
        elif self.drop_mode == 'adaptive':
            # 基于不确定性的自适应丢弃（不确定性高的更容易被丢弃）
            if uncertainties is None:
                raise ValueError("uncertainties required for adaptive dropout")
            
            # 丢弃概率 = base_prob + uncertainty * scale
            adaptive_prob = self.drop_prob + uncertainties * 0.3
            adaptive_prob = torch.nan_to_num(adaptive_prob, nan=0.5, posinf=0.5, neginf=0.0)
            adaptive_prob = torch.clamp(adaptive_prob, 0.0, 0.5)  # 最多丢弃50%
            keep_prob = torch.clamp(1 - adaptive_prob, 0.0, 1.0)
            drop_mask = torch.bernoulli(keep_prob)
        
        else:
            raise ValueError(f"Unknown drop_mode: {self.drop_mode}")
        
        # 确保至少保留一个模态
        all_dropped = (drop_mask.sum(dim=1) == 0)
        if all_dropped.any():
            # 随机恢复一个模态
            for i in torch.where(all_dropped)[0]:
                random_modal = torch.randint(0, self.num_modalities, (1,)).item()
                drop_mask[i, random_modal] = 1.0
        
        # 应用掩码
        masked_embeddings = []
        for m, emb in enumerate(embeddings_list):
            mask = drop_mask[:, m:m+1]  # [B, 1]
            # 扩展mask维度以匹配embedding
            while mask.ndim < emb.ndim:
                mask = mask.unsqueeze(-1)
            masked_embeddings.append(emb * mask)
        
        return masked_embeddings, drop_mask


def compute_ece_loss(logits, labels, n_bins=15):
    """
    计算Expected Calibration Error (ECE)作为校准损失
    
    Args:
        logits: [B, C] 模型输出logits
        labels: [B] 真实标签
        n_bins: 置信度分箱数量
    
    Returns:
        ece: 标量，ECE损失
    """
    probs = F.softmax(logits, dim=-1)  # [B, C]
    confidences, predictions = probs.max(dim=-1)  # [B]
    accuracies = predictions.eq(labels).float()  # [B]
    
    # 分箱
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # 找到在当前bin的样本
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_temperature_scaling_loss(logits, labels, temperature):
    """
    温度缩放校准损失
    
    Args:
        logits: [B, C] 模型输出logits
        labels: [B] 真实标签
        temperature: 温度参数
    
    Returns:
        loss: 标量，NLL损失（用于优化温度）
    """
    scaled_logits = logits / temperature
    loss = F.cross_entropy(scaled_logits, labels)
    return loss

