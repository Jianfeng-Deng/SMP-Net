import torch
import torch.nn.functional as F


def moe_load_balancing_loss(expert_loads, num_experts_per_band, num_bands, target_load=0.0):
    """
    Load balancing loss: 鼓励所有专家被均匀使用（原始版本）
    
    Args:
        expert_loads: List of [num_experts_per_band] tensors, 每个频带的专家负载
        num_experts_per_band: 每个频带的专家数量
        num_bands: 频带数量
        target_load: 目标负载（通常为1/num_experts_per_band）
    
    Returns:
        loss: 标量
    """
    if target_load == 0.0:
        target_load = 1.0 / num_experts_per_band
    
    total_loss = 0.0
    for band_loads in expert_loads:
        # band_loads: [num_experts_per_band]
        # 计算与目标负载的方差
        load_variance = torch.var(band_loads)
        # 计算与目标负载的均方误差
        load_mse = torch.mean((band_loads - target_load) ** 2)
        total_loss += load_variance + load_mse
    
    return total_loss / num_bands


def moe_load_balancing_loss_v2(expert_loads, num_experts_per_band, num_bands):
    """
    使用CV^2形式的Load Balancing损失（更有效，类似Switch Transformer）
    CV^2 = (std/mean)^2，鼓励均匀分布
    
    Args:
        expert_loads: List of [num_experts_per_band] tensors, 每个频带的专家负载
        num_experts_per_band: 每个频带的专家数量
        num_bands: 频带数量
    
    Returns:
        loss: 标量
    """
    total_loss = 0.0
    for band_loads in expert_loads:
        # band_loads: [num_experts_per_band]
        mean_load = band_loads.mean()
        std_load = band_loads.std()
        # CV^2 = (std/mean)^2，当分布均匀时接近0
        cv_squared = (std_load / (mean_load + 1e-10)) ** 2
        total_loss += cv_squared
    
    return total_loss / num_bands


def moe_sparse_load_balancing_loss(expert_weights, num_experts_per_band, num_bands, top_k=2):
    """
    稀疏路由专用的Load Balancing损失
    确保所有专家都有机会被选入top-k，防止路由器崩塌
    
    Args:
        expert_weights: List of [B, num_experts_per_band] tensors, 每个频带的专家权重
        num_experts_per_band: 每个频带的专家数量
        num_bands: 频带数量
        top_k: 每次选择的专家数量
    
    Returns:
        loss: 标量
    """
    total_loss = 0.0
    
    for band_weights in expert_weights:
        # band_weights: [B, num_experts_per_band]
        B = band_weights.size(0)
        
        # 计算每个专家被选入top-k的频率
        _, topk_indices = torch.topk(band_weights, k=top_k, dim=-1)  # [B, top_k]
        
        # 统计每个专家被选中的次数
        expert_selection_counts = torch.zeros(num_experts_per_band, device=band_weights.device)
        for expert_id in range(num_experts_per_band):
            # 计算该专家在多少个样本中被选入top-k
            selected = (topk_indices == expert_id).any(dim=-1).float()  # [B]
            expert_selection_counts[expert_id] = selected.sum()
        
        # 归一化为频率
        expert_selection_freq = expert_selection_counts / (B + 1e-10)  # [num_experts_per_band]
        
        # 目标频率：每个专家应该被选中的理想频率
        # 如果有E个专家，每次选top_k个，理想情况下每个专家被选中的概率是 top_k / E
        target_freq = top_k / num_experts_per_band
        
        # 计算频率的方差（鼓励均匀分布）
        freq_variance = torch.var(expert_selection_freq)
        
        # 计算与目标频率的均方误差
        freq_mse = torch.mean((expert_selection_freq - target_freq) ** 2)
        
        # 惩罚从未被选中的专家（额外的正则项）
        min_freq_penalty = F.relu(0.05 - expert_selection_freq).sum()  # 至少5%的样本应选中每个专家
        
        total_loss += freq_variance + freq_mse + 0.5 * min_freq_penalty
    
    return total_loss / num_bands


def moe_entropy_loss(expert_weights, num_bands):
    """
    Entropy loss: 鼓励路由器输出高熵（多样性）
    注意：这里我们希望适度的熵，不要太高也不要太低
    
    Args:
        expert_weights: List of [B, num_experts_per_band] tensors
        num_bands: 频带数量
    
    Returns:
        loss: 标量（负熵，因为我们希望最大化熵）
    """
    total_entropy = 0.0
    total_samples = 0
    
    for band_weights in expert_weights:
        # band_weights: [B, num_experts_per_band]
        # 计算每个样本的熵
        entropy = -torch.sum(band_weights * torch.log(band_weights + 1e-10), dim=-1)  # [B]
        total_entropy += entropy.sum()
        total_samples += entropy.size(0)
    
    avg_entropy = total_entropy / (total_samples + 1e-10)
    
    # 我们希望熵接近log(num_experts)，但不要太高
    # 使用负熵作为损失（鼓励多样性，但通过load balancing约束）
    # 这里我们返回负熵，因为我们要最小化损失
    max_entropy = torch.log(torch.tensor(expert_weights[0].size(-1), dtype=torch.float32, device=expert_weights[0].device))
    # 我们希望熵在合理范围内，惩罚过低或过高的熵
    target_entropy = max_entropy * 0.7  # 目标熵为最大熵的70%
    entropy_loss = torch.abs(avg_entropy - target_entropy)
    
    return entropy_loss


def moe_expert_diversity_loss(expert_outputs, num_bands):
    """
    专家多样性正则：鼓励不同专家产生不同的输出（原始版本）
    
    Args:
        expert_outputs: List of [B, num_experts_per_band, embed_dim] tensors
        num_bands: 频带数量
    
    Returns:
        loss: 标量
    """
    total_diversity_loss = 0.0
    
    for band_outputs in expert_outputs:
        # band_outputs: [B, num_experts_per_band, embed_dim]
        B, num_experts, D = band_outputs.shape
        
        # 归一化专家输出
        expert_norm = F.normalize(band_outputs, dim=-1)  # [B, num_experts, D]
        
        # 计算专家之间的相似度矩阵（对每个样本）
        # expert_norm: [B, num_experts, D]
        # 计算所有专家对之间的余弦相似度
        diversity_loss = 0.0
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # 计算专家i和j之间的相似度
                sim = (expert_norm[:, i, :] * expert_norm[:, j, :]).sum(dim=-1)  # [B]
                # 我们希望相似度低（多样性高），所以惩罚高相似度
                diversity_loss += sim.mean()
        
        # 归一化：num_experts对的数量 = num_experts * (num_experts - 1) / 2
        num_pairs = num_experts * (num_experts - 1) / 2
        total_diversity_loss += diversity_loss / (num_pairs + 1e-10)
    
    return total_diversity_loss / num_bands


def moe_expert_diversity_loss_v2(expert_outputs, num_bands, margin=0.3, debug=False):
    """
    改进的多样性损失：使用margin-based对比学习
    鼓励专家输出之间的余弦相似度低于margin
    
    Args:
        expert_outputs: List of [B, num_experts_per_band, embed_dim] tensors
        num_bands: 频带数量
        margin: 相似度阈值，高于此值的相似度会被惩罚
        debug: 是否输出调试信息
    
    Returns:
        loss: 标量
    """
    total_diversity_loss = 0.0
    all_similarities = []
    
    for band_outputs in expert_outputs:
        # band_outputs: [B, num_experts_per_band, embed_dim]
        B, num_experts, D = band_outputs.shape
        
        # 归一化专家输出
        expert_norm = F.normalize(band_outputs, dim=-1)  # [B, num_experts, D]
        
        # 计算专家之间的相似度矩阵（对每个样本）
        diversity_loss = 0.0
        band_similarities = []
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                # 计算专家i和j之间的相似度
                sim = (expert_norm[:, i, :] * expert_norm[:, j, :]).sum(dim=-1)  # [B]
                band_similarities.append(sim.mean().item())
                all_similarities.append(sim.mean().item())
                # 惩罚高于margin的相似度（使用ReLU确保只惩罚高相似度）
                diversity_loss += F.relu(sim - margin).mean()
        
        # 归一化：num_experts对的数量 = num_experts * (num_experts - 1) / 2
        num_pairs = num_experts * (num_experts - 1) / 2
        total_diversity_loss += diversity_loss / (num_pairs + 1e-10)
        
        # 调试输出
        if debug and len(band_similarities) > 0:
            print(f"[Diversity Debug] Band similarities: min={min(band_similarities):.4f}, "
                  f"max={max(band_similarities):.4f}, mean={sum(band_similarities)/len(band_similarities):.4f}, "
                  f"margin={margin:.4f}")
    
    # 全局调试输出
    if debug and len(all_similarities) > 0:
        print(f"[Diversity Debug] Global similarities: min={min(all_similarities):.4f}, "
              f"max={max(all_similarities):.4f}, mean={sum(all_similarities)/len(all_similarities):.4f}")
    
    return total_diversity_loss / num_bands


def moe_router_regularization_loss(router_weights, num_bands, target_weight=0.0):
    """
    路由器权重正则化：鼓励路由器输出接近均匀分布
    
    Args:
        router_weights: List of [B, num_experts_per_band] tensors，每个频带的路由器权重
        num_bands: 频带数量
        target_weight: 目标权重（通常为1/num_experts_per_band）
    
    Returns:
        loss: 标量
    """
    if len(router_weights) == 0:
        return torch.tensor(0.0, device=router_weights[0].device if len(router_weights) > 0 else 'cpu')
    
    num_experts_per_band = router_weights[0].size(-1)
    if target_weight == 0.0:
        target_weight = 1.0 / num_experts_per_band
    
    total_loss = 0.0
    for band_weights in router_weights:
        # band_weights: [B, num_experts_per_band]
        # 计算与目标权重的均方误差
        mse = torch.mean((band_weights - target_weight) ** 2)
        total_loss += mse
    
    return total_loss / num_bands


def compute_moe_loss(moe_stats, num_experts_per_band, num_bands, 
                     w_load_balance=1.0, w_entropy=0.01, w_diversity=0.1,
                     w_router_reg=0.01, use_v2_losses=True, diversity_margin=0.5,
                     expert_top_k=None, w_sparse_balance=0.5):
    """
    计算完整的MoE损失
    
    Args:
        moe_stats: dict包含'expert_loads', 'expert_weights', 'expert_outputs', 'expert_entropies'
        num_experts_per_band: 每个频带的专家数量
        num_bands: 频带数量
        w_load_balance: load balancing损失权重（默认1.0，增强约束）
        w_entropy: entropy损失权重
        w_diversity: diversity损失权重（默认0.1，增强约束）
        w_router_reg: 路由器正则化权重（默认0.01）
        use_v2_losses: 是否使用改进版本的损失函数
        diversity_margin: diversity损失的margin阈值
        expert_top_k: 稀疏路由的top-k值，如果不为None则使用稀疏路由损失
        w_sparse_balance: 稀疏路由load balancing损失权重
    
    Returns:
        total_loss: 总MoE损失
        loss_dict: 损失分解字典
    """
    expert_loads = moe_stats['expert_loads']
    expert_weights = moe_stats['expert_weights']
    expert_outputs = moe_stats['expert_outputs']
    
    # Load balancing loss（使用改进版本）
    if use_v2_losses:
        loss_load_balance = moe_load_balancing_loss_v2(
            expert_loads, num_experts_per_band, num_bands
        )
    else:
        loss_load_balance = moe_load_balancing_loss(
            expert_loads, num_experts_per_band, num_bands
        )
    
    # 稀疏路由专用的Load Balancing损失
    loss_sparse_balance = torch.tensor(0.0, device=expert_weights[0].device)
    if expert_top_k is not None and expert_top_k < num_experts_per_band:
        loss_sparse_balance = moe_sparse_load_balancing_loss(
            expert_weights, num_experts_per_band, num_bands, top_k=expert_top_k
        )
    
    # Entropy loss
    loss_entropy = moe_entropy_loss(expert_weights, num_bands)
    
    # Diversity loss（使用改进版本）
    if use_v2_losses:
        # 只在第一个epoch或每10个epoch输出调试信息
        debug_diversity = False  # 可以通过参数控制
        loss_diversity = moe_expert_diversity_loss_v2(
            expert_outputs, num_bands, margin=diversity_margin, debug=debug_diversity
        )
    else:
        loss_diversity = moe_expert_diversity_loss(expert_outputs, num_bands)
    
    # 路由器正则化损失
    loss_router_reg = moe_router_regularization_loss(
        expert_weights, num_bands
    )
    
    # 总损失
    total_loss = (
        w_load_balance * loss_load_balance +
        w_sparse_balance * loss_sparse_balance +
        w_entropy * loss_entropy +
        w_diversity * loss_diversity +
        w_router_reg * loss_router_reg
    )
    
    loss_dict = {
        'moe_load_balance': loss_load_balance.item(),
        'moe_sparse_balance': loss_sparse_balance.item() if isinstance(loss_sparse_balance, torch.Tensor) else 0.0,
        'moe_entropy': loss_entropy.item(),
        'moe_diversity': loss_diversity.item(),
        'moe_router_reg': loss_router_reg.item(),
        'moe_total': total_loss.item()
    }
    
    return total_loss, loss_dict


def bayesian_router_kl_loss(kl_losses, w_kl=0.01):
    """
    Compute KL divergence loss for Bayesian routers.
    
    Accumulates KL divergence between posterior and prior for all routers.
    This regularizes the router weights towards the prior distribution.
    
    Args:
        kl_losses: List of KL divergence scalars from each band's router
        w_kl: Weight for KL divergence loss
    
    Returns:
        loss: Weighted KL divergence loss
    """
    if not kl_losses or len(kl_losses) == 0:
        return torch.tensor(0.0)
    
    # Sum KL divergences from all bands
    total_kl = sum(kl_losses)
    
    # Average over bands
    avg_kl = total_kl / len(kl_losses)
    
    return w_kl * avg_kl


def expert_disagreement_regularization(
    disagreements, 
    target_disagreement=0.3, 
    margin=0.1,
    mode='target'
):
    """
    Regularize expert disagreement to encourage moderate diversity.
    
    Too much disagreement -> experts are not learning from each other
    Too little disagreement -> experts are collapsing to same solution
    
    Args:
        disagreements: List of [B] disagreement tensors from each band
        target_disagreement: Target disagreement value
        margin: Margin around target (no penalty if within margin)
        mode: 'target' (penalize deviation from target) or 'minimum' (penalize only below threshold)
    
    Returns:
        loss: Disagreement regularization loss
    """
    if not disagreements or len(disagreements) == 0:
        return torch.tensor(0.0)
    
    # Concatenate all disagreements: [total_samples]
    all_disagreements = torch.cat(disagreements, dim=0)
    
    # Mean disagreement
    mean_disagreement = all_disagreements.mean()
    
    if mode == 'target':
        # Penalize deviation from target (with margin)
        # loss = max(0, |disagreement - target| - margin)
        deviation = torch.abs(mean_disagreement - target_disagreement)
        loss = F.relu(deviation - margin)
    elif mode == 'minimum':
        # Only penalize if disagreement is too low
        # loss = max(0, target - disagreement)
        loss = F.relu(target_disagreement - mean_disagreement)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return loss


def compute_moe_loss_bayesian(
    moe_stats, 
    num_experts_per_band, 
    num_bands,
    # Standard MoE loss weights
    w_load_balance=1.0, 
    w_entropy=0.01, 
    w_diversity=0.1,
    w_router_reg=0.01, 
    use_v2_losses=True, 
    diversity_margin=0.5,
    expert_top_k=None, 
    w_sparse_balance=0.5,
    # Bayesian router weights
    w_kl_div=0.01,
    # Disagreement regularization weights
    w_disagreement_reg=0.01,
    disagreement_target=0.3,
    disagreement_margin=0.1,
    disagreement_mode='target'
):
    """
    Compute complete MoE loss including Bayesian router and disagreement components.
    
    This is an extended version of compute_moe_loss that adds:
    - KL divergence loss for Bayesian routers
    - Expert disagreement regularization
    
    Args:
        moe_stats: dict with MoE statistics
        ... (standard MoE loss args)
        w_kl_div: Weight for Bayesian router KL divergence
        w_disagreement_reg: Weight for disagreement regularization
        disagreement_target: Target disagreement value
        disagreement_margin: Margin around target
        disagreement_mode: 'target' or 'minimum'
    
    Returns:
        total_loss: Total MoE loss
        loss_dict: Loss breakdown dictionary
    """
    # Compute standard MoE losses
    base_loss, loss_dict = compute_moe_loss(
        moe_stats, num_experts_per_band, num_bands,
        w_load_balance=w_load_balance,
        w_entropy=w_entropy,
        w_diversity=w_diversity,
        w_router_reg=w_router_reg,
        use_v2_losses=use_v2_losses,
        diversity_margin=diversity_margin,
        expert_top_k=expert_top_k,
        w_sparse_balance=w_sparse_balance
    )
    
    total_loss = base_loss
    
    # Add Bayesian router KL divergence loss
    if 'kl_losses' in moe_stats and moe_stats['kl_losses']:
        kl_loss = bayesian_router_kl_loss(moe_stats['kl_losses'], w_kl=w_kl_div)
        total_loss = total_loss + kl_loss
        loss_dict['bayesian_kl'] = kl_loss.item()
    else:
        loss_dict['bayesian_kl'] = 0.0
    
    # Add expert disagreement regularization
    if 'expert_disagreements' in moe_stats and moe_stats['expert_disagreements']:
        disagreement_loss = expert_disagreement_regularization(
            moe_stats['expert_disagreements'],
            target_disagreement=disagreement_target,
            margin=disagreement_margin,
            mode=disagreement_mode
        )
        disagreement_loss = w_disagreement_reg * disagreement_loss
        total_loss = total_loss + disagreement_loss
        loss_dict['expert_disagreement_reg'] = disagreement_loss.item()
        
        # Also log mean disagreement for monitoring
        all_disagreements = torch.cat(moe_stats['expert_disagreements'], dim=0)
        loss_dict['mean_expert_disagreement'] = all_disagreements.mean().item()
    else:
        loss_dict['expert_disagreement_reg'] = 0.0
        loss_dict['mean_expert_disagreement'] = 0.0
    
    # Log router uncertainty if available
    if 'router_uncertainties' in moe_stats and moe_stats['router_uncertainties']:
        all_uncertainties = torch.cat(moe_stats['router_uncertainties'], dim=0)
        loss_dict['mean_router_uncertainty'] = all_uncertainties.mean().item()
    else:
        loss_dict['mean_router_uncertainty'] = 0.0
    
    loss_dict['moe_total_bayesian'] = total_loss.item()
    
    return total_loss, loss_dict
