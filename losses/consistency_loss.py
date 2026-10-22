"""
Fusion Embedding Consistency Loss

一致性正则：对于同一个样本的不同增强版本，它们的fusion embedding应该相似。
这有助于提升跨域泛化能力。
"""

import torch
import torch.nn.functional as F


def fusion_consistency_loss(z_fusion_1, z_fusion_2, temperature=1.0):
    """
    计算两个fusion embedding之间的一致性损失
    
    Args:
        z_fusion_1: [B, D] 第一个增强版本的fusion embedding
        z_fusion_2: [B, D] 第二个增强版本的fusion embedding
        temperature: 保留参数以兼容旧调用。当前实现使用有界的(1 - cosine)，建议保持为1.0。
    
    Returns:
        loss: 标量，一致性损失（越小表示越一致）
    """
    # 归一化
    z_fusion_1 = F.normalize(z_fusion_1, dim=-1)
    z_fusion_2 = F.normalize(z_fusion_2, dim=-1)
    
    # 计算余弦相似度（对角线元素，对应同一个样本的两个版本）
    # z_fusion_1: [B, D], z_fusion_2: [B, D]
    # 逐样本计算相似度
    similarities = (z_fusion_1 * z_fusion_2).sum(dim=-1)  # [B]
    
    # 一致性损失（有界、稳定）：1 - cosine，范围[0, 2]
    # 这里不再用temperature放大，避免一致性项主导训练；temperature仅为兼容保留。
    loss = (1.0 - similarities).mean()
    
    return loss


def fusion_consistency_loss_mse(z_fusion_1, z_fusion_2):
    """
    使用MSE的一致性损失（备选方案）
    
    Args:
        z_fusion_1: [B, D] 第一个增强版本的fusion embedding
        z_fusion_2: [B, D] 第二个增强版本的fusion embedding
    
    Returns:
        loss: 标量，MSE损失
    """
    z_fusion_1 = F.normalize(z_fusion_1, dim=-1)
    z_fusion_2 = F.normalize(z_fusion_2, dim=-1)
    
    # MSE损失
    loss = F.mse_loss(z_fusion_1, z_fusion_2)
    
    return loss


def fusion_consistency_loss_contrastive(z_fusion_1, z_fusion_2, temperature=0.07):
    """
    对比学习风格的一致性损失（更严格）
    不仅要求正样本对相似，还要求与其他样本不相似
    
    Args:
        z_fusion_1: [B, D] 第一个增强版本的fusion embedding
        z_fusion_2: [B, D] 第二个增强版本的fusion embedding
        temperature: 温度参数
    
    Returns:
        loss: 标量，对比学习一致性损失
    """
    z_fusion_1 = F.normalize(z_fusion_1, dim=-1)
    z_fusion_2 = F.normalize(z_fusion_2, dim=-1)
    
    # 计算所有样本对之间的相似度矩阵
    similarity_matrix = z_fusion_1 @ z_fusion_2.T / temperature  # [B, B]
    
    # 对角线元素是正样本对（同一个样本的两个版本）
    labels = torch.arange(z_fusion_1.size(0), device=z_fusion_1.device)
    
    # 双向对比损失
    loss_1to2 = F.cross_entropy(similarity_matrix, labels)
    loss_2to1 = F.cross_entropy(similarity_matrix.T, labels)
    
    loss = 0.5 * (loss_1to2 + loss_2to1)
    
    return loss

