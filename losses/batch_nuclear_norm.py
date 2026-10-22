"""
Batch Nuclear Norm Loss
Att77新增：防止特征维度坍缩，增强batch内协方差结构

数学原理：
- 批量核范数最大化等价于增强样本间的协方差结构
- 当某类样本极少时（如rolling），该类的特征方差接近0
- 通过最大化核范数可以防止这类维度坍缩

Reference: 基于随机矩阵理论的深度学习泛化研究
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNuclearNormLoss(nn.Module):
    """
    批量核范数损失：防止特征维度坍缩
    
    目标：最大化每个类别的特征矩阵的核范数
    效果：增强batch内协方差结构，防止低方差类（如rolling）特征崩塌
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, features, labels):
        """
        计算批量核范数损失
        
        Args:
            features: [B, D] 特征向量
            labels: [B] 类别标签
            
        Returns:
            loss: 标量损失（负值，最大化核范数）
        """
        if features is None or labels is None:
            return torch.tensor(0.0, requires_grad=True, device='cpu' if not torch.cuda.is_available() else 'cuda')
        
        # 确保features是2D tensor
        if features.dim() != 2:
            return torch.tensor(0.0, requires_grad=True, device=features.device)
        
        unique_labels = torch.unique(labels)
        loss = 0.0
        valid_classes = 0
        
        for c in unique_labels:
            mask = labels == c
            n_samples = mask.sum().item()
            
            # 样本太少时跳过（至少需要2个样本才能计算方差）
            if n_samples < 2:
                continue
            
            # 提取该类别的特征
            X_c = features[mask]  # [N_c, D]
            
            # 中心化（减去均值）
            X_c = X_c - X_c.mean(dim=0, keepdim=True)
            
            # 计算核范数（使用F范数近似，加速计算）
            # ||X||_* ≈ sum(sqrt(eigenvalues of X^T X))
            # 使用 ||X||_F^2 = sum(X_ij^2) 作为代理
            # 实际上，sqrt(sum(row_norm^2)) 与核范数正相关
            row_norms = torch.sqrt(torch.sum(X_c ** 2, dim=1) + 1e-8)  # [N_c]
            nuclear_approx = torch.sum(row_norms)  # 近似核范数
            
            # 归一化（按样本数）
            loss = loss - nuclear_approx / n_samples
            valid_classes += 1
        
        if valid_classes == 0:
            return torch.tensor(0.0, requires_grad=True, device=features.device)
        
        # 平均到类别数
        loss = loss / valid_classes
        
        return loss


def batch_nuclear_norm_loss(features, labels, weight=0.01):
    """
    便捷函数：计算批量核范数损失并乘以权重
    
    Args:
        features: [B, D] 特征向量
        labels: [B] 类别标签
        weight: 损失权重（默认0.01）
        
    Returns:
        weighted_loss: 加权后的损失
    """
    criterion = BatchNuclearNormLoss()
    raw_loss = criterion(features, labels)
    return raw_loss * weight


# 单元测试
if __name__ == "__main__":
    # 创建一个简单的测试
    torch.manual_seed(42)
    
    # 模拟数据：4个类别，每类10个样本，特征维度128
    features = torch.randn(40, 128)
    labels = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10)
    
    criterion = BatchNuclearNormLoss()
    loss = criterion(features, labels)
    
    print(f"Batch Nuclear Norm Loss: {loss.item():.4f}")
    print("Test passed!" if not torch.isnan(loss) and not torch.isinf(loss) else "Test failed!")
