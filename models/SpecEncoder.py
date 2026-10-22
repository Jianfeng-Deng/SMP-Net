import torch.nn as nn
import torch.nn.functional as F

from models.domain_residual import DomainFiLM


class SpecEncoder(nn.Module):
    """
    Mask-aware Multi-Channel Spectrogram Encoder with Domain Residual
    Input:
        spec: [B, C, F, T]
        spec_mask: [B, T]
        domain_ids: [B] (optional, for domain residual)
    Output:
        z_spec: [B, D]
    """

    def __init__(self, in_channels, embed_dim=256, use_domain_residual=False, num_domains=3):
        super().__init__()
        
        self.use_domain_residual = use_domain_residual

        # ========= CNN backbone =========
        self.conv1 = nn.Conv2d(
            in_channels, 32,
            kernel_size=(3, 7),
            stride=(1, 2),
            padding=(1, 3)
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=(3, 5),
            stride=(1, 2),
            padding=(1, 2)
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            64, 128,
            kernel_size=(3, 3),
            stride=(1, 2),
            padding=(1, 1)
        )
        self.bn3 = nn.BatchNorm2d(128)
        
        # ========= Domain Residual (applied after conv3) =========
        if use_domain_residual:
            self.domain_residual = DomainFiLM(
                num_domains=num_domains,
                num_features=128,  # After conv3
                init_scale=0.0  # Start from identity
            )
        else:
            self.domain_residual = None

        self.proj = nn.Linear(128, embed_dim)

    def forward(self, spec, spec_mask, domain_ids=None, return_tokens=False, stat_vec=None):
        """
        spec: [B, C, F, T]
        spec_mask: [B, T]
        domain_ids: [B] (optional, for domain residual)
        return_tokens: bool, 如果True，返回token序列 [B, T', D]；否则返回pooled embedding [B, D]
        """

        # ===== CNN =====
        x = F.relu(self.bn1(self.conv1(spec)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))    # [B, 128, F', T']
        
        # ===== Domain Residual (applied after conv3, before pooling) =====
        if self.use_domain_residual and domain_ids is not None:
            x = self.domain_residual(x, domain_ids)  # [B, 128, F', T']

        # ===== frequency pooling =====
        x = x.mean(dim=2)                      # [B, 128, T']

        # ===== mask downsampling (structure-aligned) =====
        mask = spec_mask.unsqueeze(1)          # [B, 1, T]

        for _ in range(3):  # 对应 3 次 stride=(1,2)
            mask = F.avg_pool1d(
                mask,
                kernel_size=2,
                stride=2,
                ceil_mode=True
            )

        mask = mask.squeeze(1)                 # [B, T']

        # ===== mask-aware temporal pooling =====
        x = x * mask.unsqueeze(1)  # [B, 128, T']
        
        if return_tokens:
            # 返回token序列：对每个token位置投影
            # x: [B, 128, T'] -> [B, T', 128] -> [B, T', D]
            x_tokens = x.permute(0, 2, 1)  # [B, T', 128]
            z_tokens = self.proj(x_tokens)  # [B, T', D]
            return z_tokens, mask  # 同时返回mask用于后续处理
        
        sum_x = x.sum(dim=-1)                  # [B, 128]
        denom = mask.sum(dim=-1).clamp(min=1e-6)
        pooled = sum_x / denom.unsqueeze(-1)

        # ===== projection =====
        z = self.proj(pooled)                  # [B, D]

        return z
    
    def get_domain_residual_info(self):
        """Get domain residual gate value for monitoring."""
        if self.use_domain_residual and self.domain_residual is not None:
            return {
                'gate_value': self.domain_residual.get_gate_value(),
                'type': 'FiLM'
            }
        return None