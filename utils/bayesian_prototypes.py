import torch
import torch.nn.functional as F


class BayesianPrototype:
    """
    贝叶斯原型：管理prototype的均值和协方差
    使用高斯-共轭先验进行贝叶斯更新
    """
    def __init__(self, prior_mean, prior_cov=None, prior_strength=1.0):
        """
        Args:
            prior_mean: [D] text prototype的embedding（先验均值）
            prior_cov: [D, D] 先验协方差矩阵，如果为None则使用对角矩阵
            prior_strength: 先验强度（控制先验的置信度）
        """
        self.mu_prior = prior_mean.clone()
        D = prior_mean.size(0)
        device = prior_mean.device
        
        # 如果没有提供先验协方差，使用对角矩阵（小方差表示高置信度）
        if prior_cov is None:
            self.Sigma_prior = torch.eye(D, device=device) * 0.01 * prior_strength
        else:
            self.Sigma_prior = prior_cov * prior_strength
        
        # 初始化后验为先验
        self.mu_posterior = self.mu_prior.clone()
        self.Sigma_posterior = self.Sigma_prior.clone()
        self.n_updates = 0
    
    def update(self, samples):
        """
        用训练样本更新后验分布（高斯-共轭更新）
        
        Args:
            samples: [N, D] 训练样本的embeddings
        
        Returns:
            self (用于链式调用)
        """
        if samples.size(0) == 0:
            return self
        
        n = samples.size(0)
        D = samples.size(1)
        device = samples.device
        
        # 计算样本统计量
        sample_mean = samples.mean(dim=0)  # [D]
        
        # 计算样本协方差（如果只有一个样本，使用先验协方差）
        if n > 1:
            # 使用无偏估计
            sample_cov = torch.cov(samples.T)  # [D, D]
            # 添加小的正则化项避免数值不稳定
            sample_cov = sample_cov + 1e-6 * torch.eye(D, device=device)
        else:
            sample_cov = self.Sigma_prior
        
        # 贝叶斯更新公式（高斯-共轭）
        # 后验精度 = 先验精度 + 数据精度
        # 后验均值 = (先验精度 * 先验均值 + 数据精度 * 数据均值) / 后验精度
        
        try:
            # 计算精度矩阵（协方差的逆）
            # 添加正则化避免数值不稳定
            reg = 1e-6 * torch.eye(D, device=device)
            Sigma_prior_inv = torch.inverse(self.Sigma_prior + reg)
            Sigma_sample_inv = torch.inverse(sample_cov / n + reg)
            
            # 后验精度
            Sigma_posterior_inv = Sigma_prior_inv + n * Sigma_sample_inv
            
            # 后验协方差
            self.Sigma_posterior = torch.inverse(Sigma_posterior_inv + reg)
            
            # 后验均值
            self.mu_posterior = self.Sigma_posterior @ (
                Sigma_prior_inv @ self.mu_prior + n * Sigma_sample_inv @ sample_mean
            )
            
        except RuntimeError:
            # 如果矩阵求逆失败，使用简化的更新（加权平均）
            alpha = n / (1.0 + n)  # 数据权重
            self.mu_posterior = (1 - alpha) * self.mu_prior + alpha * sample_mean
            # 协方差使用加权平均
            self.Sigma_posterior = (1 - alpha) * self.Sigma_prior + alpha * sample_cov
        
        self.n_updates += n
        return self
    
    def get_mean(self):
        """获取后验均值（归一化）"""
        return F.normalize(self.mu_posterior, dim=0)
    
    def get_covariance(self):
        """获取后验协方差矩阵"""
        return self.Sigma_posterior
    
    def get_uncertainty(self):
        """
        计算不确定性（协方差矩阵的迹，或主对角元素的和）
        """
        return torch.trace(self.Sigma_posterior)
    
    def get_confidence(self):
        """
        计算置信度（不确定性的倒数，归一化到[0,1]）
        """
        uncertainty = self.get_uncertainty()
        # 使用sigmoid将不确定性映射到置信度
        confidence = 1.0 / (1.0 + uncertainty)
        return confidence


def build_bayesian_text_prototypes(text_encoder, label_to_prompts, train_embeddings, device, prior_strength=1.0):
    """
    构建贝叶斯文本原型
    
    Args:
        text_encoder: 文本编码器
        label_to_prompts: Dict[int, List[str]] 标签到提示文本的映射
        train_embeddings: Dict[int, List[torch.Tensor]] 每个类别的训练样本embeddings
        device: 设备
        prior_strength: 先验强度
    
    Returns:
        Dict[int, BayesianPrototype] 贝叶斯原型字典
    """
    text_encoder.eval()
    bayesian_prototypes = {}
    
    with torch.no_grad():
        for label, prompts in label_to_prompts.items():
            # 1. 构建先验（text prototype）
            embs = []
            for p in prompts:
                emb = text_encoder([p]).to(device)
                embs.append(F.normalize(emb.squeeze(0), dim=0))
            
            # 先验均值：text prototype的平均
            prior_mean = torch.stack(embs, dim=0).mean(dim=0)  # [D]
            prior_mean = F.normalize(prior_mean, dim=0)
            
            # 2. 创建贝叶斯原型
            bayesian_proto = BayesianPrototype(
                prior_mean=prior_mean,
                prior_cov=None,  # 使用默认对角协方差
                prior_strength=prior_strength
            )
            
            # 3. 用训练样本更新后验
            if label in train_embeddings and len(train_embeddings[label]) > 0:
                # 将列表转换为tensor（兼容批量收集：如果列表元素已经是批量tensor，直接cat；否则stack）
                emb_list = train_embeddings[label]
                if len(emb_list) == 1 and emb_list[0].dim() == 2:
                    # 如果只有一个元素且是2D tensor（批量），直接使用
                    samples = emb_list[0].to(device)
                else:
                    # 多个元素，需要stack或cat
                    samples = torch.cat(emb_list, dim=0).to(device) if emb_list[0].dim() == 2 else torch.stack(emb_list, dim=0).to(device)
                # 归一化
                samples = F.normalize(samples, dim=-1)
                # 更新后验
                bayesian_proto.update(samples)
            
            bayesian_prototypes[label] = bayesian_proto
    
    return bayesian_prototypes


def extract_posterior_prototypes(bayesian_prototypes):
    """
    从贝叶斯原型中提取后验均值（用于损失计算）
    
    Args:
        bayesian_prototypes: Dict[int, BayesianPrototype]
    
    Returns:
        Dict[int, torch.Tensor] 后验均值字典
    """
    return {
        label: proto.get_mean()
        for label, proto in bayesian_prototypes.items()
    }

