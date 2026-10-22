"""
Instance-Batch Normalization (IBN) and MixStyle modules for domain generalization.

References:
- IBN: "Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net" (ECCV 2018)
- MixStyle: "Domain Generalization with MixStyle" (ICLR 2021)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class IBN(nn.Module):
    """
    Instance-Batch Normalization layer.
    
    Split channels into two parts:
    - First `ibn_ratio * channels` use Instance Normalization (IN)
    - Remaining channels use Batch Normalization (BN)
    
    IN helps preserve instance-specific information (useful for domain-invariant features)
    BN helps with inter-sample statistics (useful for discrimination)
    
    Args:
        num_features: number of channels
        ibn_ratio: ratio of channels using IN (default: 0.5)
    """
    def __init__(self, num_features, ibn_ratio=0.5):
        super().__init__()
        self.num_features = num_features
        split = int(num_features * ibn_ratio)
        self.split = split
        
        # Instance normalization for first `split` channels
        if split > 0:
            self.IN = nn.InstanceNorm2d(split, affine=True)
        else:
            self.IN = None
        
        # Batch normalization for remaining channels
        if split < num_features:
            self.BN = nn.BatchNorm2d(num_features - split)
        else:
            self.BN = None
    
    def forward(self, x):
        """
        x: [B, C, H, W] or [B, C, T] for 1D
        """
        if self.split == 0:
            # All BN
            return self.BN(x)
        elif self.split == self.num_features:
            # All IN
            return self.IN(x)
        else:
            # Split channels
            x_in = x[:, :self.split]
            x_bn = x[:, self.split:]
            
            x_in = self.IN(x_in)
            x_bn = self.BN(x_bn)
            
            return torch.cat([x_in, x_bn], dim=1)


class IBN1d(nn.Module):
    """
    1D Instance-Batch Normalization for temporal/signal data.

    Same principle as IBN (ECCV 2018) but for [B, C, T] tensors:
    - First `ibn_ratio * C` channels → InstanceNorm1d (removes per-sample style)
    - Remaining channels → BatchNorm1d (preserves discriminative statistics)
    """
    def __init__(self, num_features, ibn_ratio=0.5):
        super().__init__()
        self.num_features = num_features
        split = int(num_features * ibn_ratio)
        self.split = split

        if split > 0:
            self.IN = nn.InstanceNorm1d(split, affine=True)
        else:
            self.IN = None

        if split < num_features:
            self.BN = nn.BatchNorm1d(num_features - split)
        else:
            self.BN = None

    def forward(self, x):
        """x: [B, C, T]"""
        if self.split == 0:
            return self.BN(x)
        elif self.split == self.num_features:
            return self.IN(x)
        else:
            x_in = x[:, :self.split]
            x_bn = x[:, self.split:]
            x_in = self.IN(x_in)
            x_bn = self.BN(x_bn)
            return torch.cat([x_in, x_bn], dim=1)


class MixStyle(nn.Module):
    """
    MixStyle for domain generalization.
    
    Randomly mixes instance-level statistics (mean, std) between samples
    to create virtual domains during training. This encourages the model
    to learn domain-invariant features.
    
    Args:
        p: probability of applying MixStyle (default: 0.5)
        alpha: parameter for Beta distribution (default: 0.1)
        eps: small constant for numerical stability
        mix_mode: 'batch' (mix within batch) or 'cross' (mix across batches)
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix_mode='batch'):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.mix_mode = mix_mode
        self._activated = True  # Can be turned off during evaluation
    
    def set_activation_status(self, status):
        """Enable/disable MixStyle."""
        self._activated = status
    
    def forward(self, x):
        """
        x: [B, C, H, W] or [B, C, T]
        """
        if not self.training or not self._activated:
            return x
        
        if random.random() > self.p:
            return x
        
        B = x.size(0)
        if B <= 1:
            # Need at least two samples to mix statistics safely.
            return x
        
        # Compute instance-level statistics
        # For 2D: [B, C, H, W] -> mean/std over (H, W)
        # For 1D: [B, C, T] -> mean/std over (T,)
        if x.dim() == 4:
            # 2D case: [B, C, H, W]
            mu = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
            var = x.var(dim=[2, 3], keepdim=True, unbiased=False)  # [B, C, 1, 1]
        elif x.dim() == 3:
            # 1D case: [B, C, T]
            mu = x.mean(dim=2, keepdim=True)  # [B, C, 1]
            var = x.var(dim=2, keepdim=True, unbiased=False)  # [B, C, 1]
        else:
            return x

        # Guard against degenerate statistics to avoid NaN propagation.
        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        sig = (var + self.eps).sqrt()
        
        # Normalize
        x_normed = (x - mu) / sig
        
        # Generate mixing weights from Beta distribution
        alpha = float(max(self.alpha, 1e-4))
        lmda = torch.distributions.Beta(alpha, alpha).sample((B, 1, 1, 1)).to(x.device)
        if x.dim() == 3:
            lmda = lmda.squeeze(-1)  # [B, 1, 1] for 1D
        lmda = torch.nan_to_num(lmda, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        
        # Random shuffle for mixing
        perm = torch.randperm(B)
        
        # Mix statistics
        mu_mix = lmda * mu + (1 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1 - lmda) * sig[perm]
        
        # Denormalize with mixed statistics
        out = x_normed * sig_mix + mu_mix
        if torch.isnan(out).any() or torch.isinf(out).any():
            return x
        return out


def build_ibn_block(in_channels, out_channels, kernel_size, stride, padding, 
                     ibn_ratio=0.5, use_ibn=True):
    """
    Build a conv block with IBN.
    
    Returns:
        conv: Conv2d layer
        norm: IBN or BatchNorm2d layer
    """
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    if use_ibn and ibn_ratio > 0:
        norm = IBN(out_channels, ibn_ratio=ibn_ratio)
    else:
        norm = nn.BatchNorm2d(out_channels)
    
    return conv, norm

