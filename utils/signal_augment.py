"""
Signal Data Augmentation Utilities

为signal模态提供数据增强功能，用于一致性正则训练。
"""

import torch


def add_awgn_torch(signal, snr_db_range=(20.0, 40.0)):
    """
    在PyTorch tensor上添加AWGN（Additive White Gaussian Noise）
    
    Args:
        signal: [B, T] 或 [T] torch.Tensor
        snr_db_range: (min_snr, max_snr) SNR范围（dB）
    
    Returns:
        augmented_signal: 添加噪声后的信号
    """
    device = signal.device
    dtype = signal.dtype
    
    # 如果是2D，保持batch维度
    is_batch = signal.dim() == 2
    if not is_batch:
        signal = signal.unsqueeze(0)
    
    B, T = signal.shape
    
    # 随机SNR
    snr_db = torch.empty(B, device=device).uniform_(snr_db_range[0], snr_db_range[1])
    snr_linear = 10.0 ** (snr_db / 10.0)
    
    # 计算信号功率
    sig_power = signal.pow(2).mean(dim=-1, keepdim=True)  # [B, 1]
    
    # 计算噪声功率
    noise_power = sig_power / (snr_linear.unsqueeze(-1) + 1e-12)  # [B, 1]
    
    # 生成高斯噪声
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    
    # 添加噪声
    augmented = signal + noise
    
    if not is_batch:
        augmented = augmented.squeeze(0)
    
    return augmented.to(dtype)


def time_shift(signal, shift_range=0.1):
    """
    时间轴平移增强
    
    Args:
        signal: [B, T] 或 [T] torch.Tensor
        shift_range: 最大平移比例（相对于信号长度）
    
    Returns:
        shifted_signal: 平移后的信号
    """
    is_batch = signal.dim() == 2
    if not is_batch:
        signal = signal.unsqueeze(0)
    
    B, T = signal.shape
    device = signal.device
    
    # 随机平移量
    max_shift = int(T * shift_range)
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=device)
    
    # 对每个样本应用平移
    shifted = torch.zeros_like(signal)
    for i in range(B):
        shift = shifts[i].item()
        if shift > 0:
            shifted[i, shift:] = signal[i, :-shift]
        elif shift < 0:
            shifted[i, :shift] = signal[i, -shift:]
        else:
            shifted[i] = signal[i]
    
    if not is_batch:
        shifted = shifted.squeeze(0)
    
    return shifted


def amplitude_scale(signal, scale_range=(0.8, 1.2)):
    """
    幅度缩放增强
    
    Args:
        signal: [B, T] 或 [T] torch.Tensor
        scale_range: (min_scale, max_scale) 缩放范围
    
    Returns:
        scaled_signal: 缩放后的信号
    """
    is_batch = signal.dim() == 2
    if not is_batch:
        signal = signal.unsqueeze(0)
    
    B = signal.shape[0]
    device = signal.device
    
    # 随机缩放因子
    scales = torch.empty(B, device=device).uniform_(scale_range[0], scale_range[1])
    scales = scales.view(-1, 1)  # [B, 1]
    
    scaled = signal * scales
    
    if not is_batch:
        scaled = scaled.squeeze(0)
    
    return scaled


def augment_signal(signal, augment_prob=0.5, 
                   noise_prob=0.3, snr_db_range=(20.0, 40.0),
                   shift_prob=0.3, shift_range=0.1,
                   scale_prob=0.3, scale_range=(0.8, 1.2)):
    """
    综合signal增强函数
    
    Args:
        signal: [B, T] torch.Tensor
        augment_prob: 是否应用增强的概率
        noise_prob: 添加噪声的概率（如果应用增强）
        shift_prob: 时间平移的概率（如果应用增强）
        scale_prob: 幅度缩放的概率（如果应用增强）
        snr_db_range: 噪声SNR范围
        shift_range: 平移范围
        scale_range: 缩放范围
    
    Returns:
        augmented_signal: 增强后的信号
    """
    if not torch.rand(1).item() < augment_prob:
        return signal
    
    augmented = signal.clone()
    
    # 添加噪声
    if torch.rand(1).item() < noise_prob:
        augmented = add_awgn_torch(augmented, snr_db_range)
    
    # 时间平移
    if torch.rand(1).item() < shift_prob:
        augmented = time_shift(augmented, shift_range)
    
    # 幅度缩放
    if torch.rand(1).item() < scale_prob:
        augmented = amplitude_scale(augmented, scale_range)
    
    return augmented

