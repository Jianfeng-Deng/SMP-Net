import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


def stack_prototypes(text_prototypes: Dict[int, torch.Tensor], device):
    """
    将prototype字典转换为矩阵
    
    Args:
        text_prototypes: Dict[int, torch.Tensor] 标签到prototype的映射
        device: 设备
    
    Returns:
        torch.Tensor [C, D] prototype矩阵
    """
    keys = sorted(text_prototypes.keys())
    proto_list = [text_prototypes[k] for k in keys]
    return torch.stack(proto_list, dim=0).to(device)


def compute_uncertainty(embeddings):
    """
    计算embedding的不确定性（使用方差）
    
    Args:
        embeddings: [B, D] 样本embeddings
    
    Returns:
        torch.Tensor [B] 每个样本的不确定性
    """
    # 方法1：使用embedding的L2范数方差（归一化后应该接近1）
    # 如果方差大，说明embedding不稳定
    if embeddings.size(0) == 1:
        return torch.tensor(0.0, device=embeddings.device)
    
    # 计算每个维度上的方差，然后求和
    variance = torch.var(embeddings, dim=0)  # [D]
    uncertainty = variance.sum()  # 标量
    
    # 扩展到batch维度
    return uncertainty.expand(embeddings.size(0))


def clip_update(update, original_param, max_norm=0.01):
    """
    限制参数更新的幅度（implicit differentiation的简化版本）
    
    Args:
        update: 参数更新
        original_param: 原始参数
        max_norm: 最大更新范数
    
    Returns:
        裁剪后的更新
    """
    update_norm = torch.norm(update)
    if update_norm > max_norm:
        update = update * (max_norm / (update_norm + 1e-8))
    return update


class OnlineBayesianFinetuner:
    """
    在线贝叶斯微调器
    在测试时用无标签样本进行少量参数微调
    """
    def __init__(
        self,
        lr=1e-4,
        num_steps=3,
        lambda_uncertainty=0.1,
        max_update_norm=0.01,
        finetune_modules=None
    ):
        """
        Args:
            lr: 微调学习率
            num_steps: 微调步数
            lambda_uncertainty: 不确定性损失的权重
            max_update_norm: 最大更新范数（限制更新幅度）
            finetune_modules: 要微调的模块名称列表（如['fusion', 'decoder']）
        """
        self.lr = lr
        self.num_steps = num_steps
        self.lambda_uncertainty = lambda_uncertainty
        self.max_update_norm = max_update_norm
        self.finetune_modules = finetune_modules or ['fusion', 'decoder']
    
    def finetune(
        self,
        fusion_module: nn.Module,
        fusion_decoder: nn.Module,
        text_prototypes: Dict[int, torch.Tensor],
        z_fusion: torch.Tensor,
        device
    ):
        """
        执行在线微调
        
        Args:
            fusion_module: 融合模块
            fusion_decoder: 解码器
            text_prototypes: Dict[int, torch.Tensor] 文本原型
            z_fusion: [N, D] 测试样本的fusion embeddings（需要梯度）
            device: 设备
        
        Returns:
            tuple: (微调后的fusion_module, fusion_decoder)
        """
        # 保存原始参数
        original_fusion_params = {n: p.clone().detach() for n, p in fusion_module.named_parameters() if p.requires_grad}
        original_decoder_params = {n: p.clone().detach() for n, p in fusion_decoder.named_parameters() if p.requires_grad}
        
        # 创建优化器（只优化fusion和decoder）
        params_to_optimize = [p for p in fusion_module.parameters() if p.requires_grad]
        params_to_optimize += [p for p in fusion_decoder.parameters() if p.requires_grad]
        
        if len(params_to_optimize) == 0:
            print("[Online Finetuning] Warning: No trainable parameters found")
            return fusion_module, fusion_decoder
        
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)
        
        # 将模型设置为训练模式
        fusion_module.train()
        fusion_decoder.train()
        
        # 微调循环
        # 注意：这个方法假设z_fusion已经计算好，但多次backward会导致计算图冲突
        # 建议使用finetune_with_recompute方法，它在每次step中重新计算embeddings
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # 计算损失（使用fusion_decoder以确保梯度传播）
            loss = self.compute_loss(z_fusion, fusion_decoder, text_prototypes, device)
            
            # 反向传播
            # 注意：如果z_fusion是从多个batch concatenate的，第二次backward会失败
            # 这就是为什么我们需要finetune_with_recompute方法
            loss.backward()
            
            # 梯度裁剪（限制更新幅度）
            if len(params_to_optimize) > 0:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, self.max_update_norm * 10)
            
            # 更新参数
            optimizer.step()
            
            # 额外的参数裁剪（确保不会偏离太远）
            with torch.no_grad():
                for name, param in fusion_module.named_parameters():
                    if name in original_fusion_params and param.requires_grad:
                        diff = param - original_fusion_params[name]
                        diff = clip_update(diff, original_fusion_params[name], self.max_update_norm)
                        param.data = original_fusion_params[name] + diff
                
                for name, param in fusion_decoder.named_parameters():
                    if name in original_decoder_params and param.requires_grad:
                        diff = param - original_decoder_params[name]
                        diff = clip_update(diff, original_decoder_params[name], self.max_update_norm)
                        param.data = original_decoder_params[name] + diff
            
            # 关键修复：在每次backward后，detach z_fusion，这样下次step会创建新的计算图
            # 但我们需要重新通过fusion_module计算，这需要原始输入
            # 更好的方案：在AnomalyDetector中，每次step都重新计算embeddings
            # 或者：只进行一次backward，然后使用多个optimizer step（但这不会更新参数）
            
            # 临时方案：使用retain_graph=True，但这会占用内存
            # 或者：修改为只进行一次微调step，但增加学习率
            
            # 最佳方案：修改AnomalyDetector，在每次step中重新计算embeddings
        
        # 恢复eval模式
        fusion_module.eval()
        fusion_decoder.eval()
        
        return fusion_module, fusion_decoder
    
    def finetune_with_recompute(
        self,
        fusion_module: nn.Module,
        fusion_decoder: nn.Module,
        text_prototypes: Dict[int, torch.Tensor],
        finetune_inputs: List[tuple],  # List of (z_sig, z_spec) tuples
        device
    ):
        """
        执行在线微调，在每次step中重新计算embeddings以避免计算图冲突
        
        Args:
            fusion_module: 融合模块
            fusion_decoder: 解码器
            text_prototypes: Dict[int, torch.Tensor] 文本原型
            finetune_inputs: List of (z_sig, z_spec) tuples, encoder outputs (detached)
            device: 设备
        
        Returns:
            tuple: (微调后的fusion_module, fusion_decoder)
        """
        # 保存原始参数
        original_fusion_params = {n: p.clone().detach() for n, p in fusion_module.named_parameters() if p.requires_grad}
        original_decoder_params = {n: p.clone().detach() for n, p in fusion_decoder.named_parameters() if p.requires_grad}
        
        # 创建优化器（只优化fusion和decoder）
        params_to_optimize = [p for p in fusion_module.parameters() if p.requires_grad]
        params_to_optimize += [p for p in fusion_decoder.parameters() if p.requires_grad]
        
        if len(params_to_optimize) == 0:
            print("[Online Finetuning] Warning: No trainable parameters found")
            return fusion_module, fusion_decoder
        
        optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)
        
        # 将模型设置为训练模式
        fusion_module.train()
        fusion_decoder.train()
        
        # 微调循环：每次step重新计算embeddings
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # 重新计算z_fusion（每次都是新的计算图）
            finetune_embeddings = []
            for z_sig, z_spec in finetune_inputs:
                # z_sig and z_spec are detached, so this creates a new graph each time
                z_fusion = fusion_module(z_sig, z_spec)  # [B, D]
                z_fusion = F.normalize(z_fusion, dim=-1)
                finetune_embeddings.append(z_fusion)
            
            # Concatenate embeddings
            z_fusion = torch.cat(finetune_embeddings, dim=0)  # [N, D]
            
            # 计算损失
            loss = self.compute_loss(z_fusion, fusion_decoder, text_prototypes, device)
            
            # 反向传播（每次都是新的计算图，所以不需要retain_graph）
            loss.backward()
            
            # 梯度裁剪（限制更新幅度）
            if len(params_to_optimize) > 0:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, self.max_update_norm * 10)
            
            # 更新参数
            optimizer.step()
            
            # 额外的参数裁剪（确保不会偏离太远）
            with torch.no_grad():
                for name, param in fusion_module.named_parameters():
                    if name in original_fusion_params and param.requires_grad:
                        diff = param - original_fusion_params[name]
                        diff = clip_update(diff, original_fusion_params[name], self.max_update_norm)
                        param.data = original_fusion_params[name] + diff
                
                for name, param in fusion_decoder.named_parameters():
                    if name in original_decoder_params and param.requires_grad:
                        diff = param - original_decoder_params[name]
                        diff = clip_update(diff, original_decoder_params[name], self.max_update_norm)
                        param.data = original_decoder_params[name] + diff
        
        # 恢复eval模式
        fusion_module.eval()
        fusion_decoder.eval()
        
        return fusion_module, fusion_decoder
    
    def compute_loss(
        self,
        z_fusion: torch.Tensor,
        fusion_decoder: nn.Module,
        text_prototypes: Dict[int, torch.Tensor],
        device
    ):
        """
        计算微调损失
        
        Args:
            z_fusion: [N, D] fusion embeddings（需要梯度）
            fusion_decoder: 解码器模块（用于计算logits和不确定性）
            text_prototypes: 文本原型
            device: 设备
        
        Returns:
            torch.Tensor 损失值
        """
        # 确保z_fusion需要梯度
        if not z_fusion.requires_grad:
            z_fusion = z_fusion.detach().requires_grad_(True)
        
        z_fusion = F.normalize(z_fusion, dim=-1)
        
        # 1. 相似度损失：最大化与text prototype的余弦相似度
        proto_mat = stack_prototypes(text_prototypes, device)  # [C, D]
        proto_mat = F.normalize(proto_mat, dim=-1)
        
        # 计算每个样本与所有prototype的相似度
        similarity = z_fusion @ proto_mat.T  # [N, C]
        # 取最大相似度（最匹配的prototype）
        max_similarity = similarity.max(dim=1)[0]  # [N]
        # 损失：最小化负相似度 = 最大化相似度
        loss_similarity = -max_similarity.mean()
        
        # 2. 不确定性损失：使用decoder输出的熵作为不确定性代理
        # 计算logits（这会连接到fusion_decoder的参数）
        logits = fusion_decoder(z_fusion)  # [N, C]
        probs = F.softmax(logits, dim=-1)  # [N, C]
        # 计算熵（熵越高，不确定性越大）
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [N]
        loss_uncertainty = entropy.mean()  # 最小化熵 = 最小化不确定性
        
        # 总损失
        total_loss = loss_similarity + self.lambda_uncertainty * loss_uncertainty
        
        return total_loss


def split_loader(loader, n_samples):
    """
    将DataLoader分割为微调样本和评估样本
    
    Args:
        loader: DataLoader
        n_samples: 用于微调的样本数
    
    Returns:
        tuple: (微调数据列表, 剩余DataLoader)
    """
    finetune_data = []
    eval_data = []
    count = 0
    
    for batch in loader:
        if count < n_samples:
            finetune_data.append(batch)
            count += batch['signal'].size(0)
        else:
            eval_data.append(batch)
    
    return finetune_data, eval_data

