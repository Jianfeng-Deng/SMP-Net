import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.domain_residual import DomainFiLM
from models.domain_lora import DomainLoRA
from models.ibn_mixstyle import MixStyle, IBN1d
from models.ablation_gates import apply_residual_gate
from utils.ablation_gating import get_gate_bank


class GaussianSmoothing1d(nn.Module):
    """
    [ATT101] Gaussian Smoothing Layer for Anti-Aliasing
    Goal: Widen sparse Dirac impulses so they survive strided convolutions (Stride=2).
    """
    def __init__(self, channels, kernel_size=7, sigma=1.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        # Create 1D Gaussian kernel
        x_cord = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        gaussian_kernel = torch.exp(-0.5 * (x_cord / sigma) ** 2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum() # Normalize
        
        # Reshape to [C, 1, K] for depthwise convolution
        weight = gaussian_kernel.view(1, 1, kernel_size).repeat(channels, 1, 1)
        
        self.register_buffer('weight', weight)
        self.padding = kernel_size // 2
        self.groups = channels

    def forward(self, x):
        # x: [B, C, T]
        return F.conv1d(x, self.weight, padding=self.padding, groups=self.groups)


class SignalEncoder(nn.Module):
    """
    Mask-aware 1D CNN Signal Encoder with Domain Residual
    Input:
        signal: [B, T]
        signal_mask: [B, T]
        domain_ids: [B] (optional, for domain residual)
    Output:
        z_signal: [B, D]
    """

    def __init__(
        self,
        embed_dim=256,
        use_domain_residual=False,
        num_domains=3,
        num_classes=4,
        use_classification_head=False,
        use_domain_lora=False,
        domain_lora_rank=4,
        domain_lora_alpha=None,
        domain_lora_dropout=0.0
    ):
        super().__init__()
        
        self.use_domain_residual = use_domain_residual
        self.use_classification_head = use_classification_head

        # ATT394: IBN-a for SignalEncoder — read config
        try:
            import config as _cfg_ibn
            _use_ibn_signal = bool(getattr(_cfg_ibn, "USE_IBN_SIGNAL", False))
            _ibn_signal_ratio = float(getattr(_cfg_ibn, "IBN_SIGNAL_RATIO", 0.5))
        except Exception:
            _use_ibn_signal = False
            _ibn_signal_ratio = 0.5
        self._use_ibn_signal = _use_ibn_signal

        # [ATT101] Anti-Aliasing Smoothing Layer
        self.smoothing = GaussianSmoothing1d(channels=1, kernel_size=7, sigma=1.0)
        self.log_smoothing_stats = True

        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)

        # ATT394: IBN-a only on shallow layers (bn1, bn2); bn3 always BN
        if _use_ibn_signal:
            self.bn1 = IBN1d(32, ibn_ratio=_ibn_signal_ratio)
            self.bn2 = IBN1d(64, ibn_ratio=_ibn_signal_ratio)
            print(f"[ATT394][SignalEncoder] IBN1d enabled: ratio={_ibn_signal_ratio}, layers=bn1+bn2")
        else:
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # Attempt 66: MixStyle in early 1D layers for domain generalization.
        try:
            import config as cfg
            use_mixstyle_signal = bool(getattr(cfg, "USE_MIXSTYLE_SIGNAL", False))
            mixstyle_p = float(getattr(cfg, "MIXSTYLE_P", 0.5))
            mixstyle_alpha = float(getattr(cfg, "MIXSTYLE_ALPHA", 0.3))
        except Exception:
            use_mixstyle_signal = False
            mixstyle_p = 0.5
            mixstyle_alpha = 0.3
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle_signal else None
        
        # ========= Domain Residual (applied after conv3) =========
        if use_domain_residual:
            self.domain_residual = DomainFiLM(
                num_domains=num_domains,
                num_features=128,  # After conv3
                init_scale=0.0  # Start from identity
            )
        else:
            self.domain_residual = None

        self.fc = nn.Linear(128, embed_dim)
        self.use_domain_lora = use_domain_lora
        if use_domain_lora:
            self.domain_lora_fc = DomainLoRA(
                in_features=128,
                out_features=embed_dim,
                rank=domain_lora_rank,
                num_domains=num_domains,
                alpha=domain_lora_alpha,
                dropout=domain_lora_dropout
            )
        
        # ========= Lightweight Classification Head (for uncertainty estimation) =========
        if use_classification_head:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim // 2, num_classes)
            )
        else:
            self.classifier = None

    def forward(self, signal, signal_mask, domain_ids=None, return_tokens=False, return_logits=False):
        """
        signal: [B, T]
        signal_mask: [B, T]
        domain_ids: [B] (optional, for domain residual)
        return_tokens: bool, 如果True，返回token序列 [B, T', D]；否则返回pooled embedding [B, D]
        return_logits: bool, 如果True且use_classification_head=True，同时返回分类logits [B, num_classes]
        """

        # [B, 1, T]
        _gate_bank = get_gate_bank(self)
        x = signal.unsqueeze(1)

        # CNN backbone
        x = F.relu(self.bn1(self.conv1(x)))
        if self.mixstyle is not None:
            _x_ms = self.mixstyle(x)
            x = apply_residual_gate(_gate_bank, "domain_cond_mixstyle_signal", _x_ms, x)
        x = F.relu(self.bn2(self.conv2(x)))
        if self.mixstyle is not None:
            _x_ms = self.mixstyle(x)
            x = apply_residual_gate(_gate_bank, "domain_cond_mixstyle_signal", _x_ms, x)
        x = F.relu(self.bn3(self.conv3(x)))   # [B, C, T']

        # ===== Domain Residual (applied after conv3, before pooling) =====
        if self.use_domain_residual and domain_ids is not None:
            _x_dr = self.domain_residual(x, domain_ids)
            x = apply_residual_gate(_gate_bank, "domain_cond_domain_film_signal", _x_dr, x)

        # ---- mask downsampling ----
        # 原始 mask: [B, T]
        mask = signal_mask.unsqueeze(1)       # [B, 1, T]

        # [ATT99] Sparse Signal Preservation: Switch Mask Downsampling to MaxPool
        # Rationale: Impulse at t=1 might be the ONLY valid feature. AvgPool would dilute it. MaxPool keeps it.
        # We use ceil_mode=True to simpler match the extensive strided convs.
        mask = F.max_pool1d(
            F.max_pool1d(
                F.max_pool1d(mask, kernel_size=2, stride=2, ceil_mode=True),
                kernel_size=2, stride=2, ceil_mode=True
            ),
            kernel_size=2, stride=2, ceil_mode=True
        )  # [B, 1, T']

        # ---- mask-aware global average pooling ----
        # 优化：减少squeeze/unsqueeze操作
        mask_1d = mask.squeeze(1)             # [B, T']
        x = x * mask_1d.unsqueeze(1)          # [B, C, T'] * [B, 1, T'] -> [B, C, T']

        if return_tokens:
            # 返回token序列：对每个token位置投影
            # x: [B, C, T'] -> [B, T', C] -> [B, T', D]
            x_tokens = x.permute(0, 2, 1)  # [B, T', C]
            z_tokens = self.fc(x_tokens)   # [B, T', D]
            return z_tokens, mask_1d  # 同时返回mask用于后续处理
        
        # [ATT100] Global SoftPool (Approximate MaxPool)
        # Rationale: MaxPool is sensitive to noise outliers (the "Noise Trap").
        # AvgPool dilutes sparse signals (the "Sparsity Paradox").
        # SoftPool (Log-Sum-Exp) with beta=10 provides a smooth maximum.
        # Formula: y = sum(exp(beta * x) * x) / sum(exp(beta * x))
        beta = 10.0
        x_scaled = x * beta
        
        # Apply mask to attention weights: illegal positions get -inf
        if mask_1d is not None:
            mask_expanded = mask_1d.unsqueeze(1).expand_as(x) # [B, C, T']
            x_scaled = x_scaled.masked_fill(mask_expanded < 0.5, -1e9)
        
        soft_max_weights = F.softmax(x_scaled, dim=-1) # [B, C, T']
        pooled = (x * soft_max_weights).sum(dim=-1) # [B, C]

        # Projection
        z = self.fc(pooled)                     # [B, D]
        if self.use_domain_lora:
            lora_delta = self.domain_lora_fc(pooled, domain_ids)
            z = apply_residual_gate(_gate_bank, "domain_cond_lora_signal", z + lora_delta, z)

        # Classification head (for uncertainty estimation)
        if return_logits and self.use_classification_head and self.classifier is not None:
            logits = self.classifier(z)  # [B, num_classes]
            return z, logits
        elif return_logits and (not self.use_classification_head or self.classifier is None):
            # If classification head not enabled, return None for logits
            return z, None
        else:
            return z
    
    def get_domain_residual_info(self):
        """Get domain residual gate value for monitoring."""
        if self.use_domain_residual and self.domain_residual is not None:
            return {
                'gate_value': self.domain_residual.get_gate_value(),
                'type': 'FiLM'
            }
        return None
