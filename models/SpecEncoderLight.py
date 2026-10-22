import math
import os
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.domain_residual import DomainFiLM
from models.ibn_mixstyle import IBN, MixStyle
from models.domain_discriminator import DomainDiscriminator
from models.grl import grad_reverse
from models.ablation_gates import apply_residual_gate
from utils.ablation_gating import get_gate_bank

logger = logging.getLogger(__name__)


class StatFiLMGenerator(nn.Module):
    """
    ATT377-StatFiLM: generates channel-wise affine params (gamma, beta) and a
    residual gate scalar from an N-dim statistical feature vector (N=11 after ATT379b-S2).
    Identity-initialized: gamma=1, beta=0, gate=0 at init → no-op start.
    """
    def __init__(self, n_stat_features: int, n_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.film_net = nn.Sequential(
            nn.Linear(n_stat_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_channels * 2),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(n_stat_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # ATT378b: small random init so gamma/beta start non-zero and can receive gradient signal.
        # Previous zeros_ made film_net a permanent identity (gamma=1, beta=0, grad=0).
        nn.init.normal_(self.film_net[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.film_net[-1].bias)
        nn.init.normal_(self.gate_net[-1].weight, mean=0.0, std=0.02)
        nn.init.constant_(self.gate_net[-1].bias, 0.0)

    def forward(self, stat_vec):
        """
        stat_vec: [B, N_feat]
        Returns: gamma [B, C, 1, 1], beta [B, C, 1, 1], gate [B, 1, 1, 1]
        """
        film_out = self.film_net(stat_vec)
        gamma_raw, beta = film_out.chunk(2, dim=-1)
        gamma = 1.0 + gamma_raw
        gate = torch.sigmoid(self.gate_net(stat_vec))
        return (
            gamma.unsqueeze(-1).unsqueeze(-1),
            beta.unsqueeze(-1).unsqueeze(-1),
            gate.unsqueeze(-1).unsqueeze(-1),
        )


class FsConditioningFiLM(nn.Module):
    """
    ATT423-P2: Sampling-rate conditioned FiLM (Feature-wise Linear Modulation).

    Generates channel-wise affine transforms (gamma, beta) from a normalized
    sampling rate scalar. This allows the SpecEncoder to adapt its feature
    extraction based on the physical frequency range of the input domain.

    Identity-initialized: gamma_raw=0 → effective gamma=1, beta=0 → no-op at start.
    The encoder gradually learns domain-adaptive modulations during training.

    Reference: Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"
    (AAAI 2018) — general-purpose conditional feature modulation via learned affine transforms.
    """

    def __init__(self, n_channels: int, hidden_dim: int = 32, ref_fs: float = 64000.0):
        super().__init__()
        self.ref_fs = max(1.0, ref_fs)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, n_channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, fs_values: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] feature map
        fs_values: [B] tensor of sampling rate values (Hz)
        """
        fs_norm = (fs_values.float() / self.ref_fs).unsqueeze(-1)  # [B, 1]
        out = self.net(fs_norm)  # [B, C*2]
        gamma_raw, beta = out.chunk(2, dim=-1)
        gamma = 1.0 + gamma_raw
        return gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)


class FreqLayerNorm(nn.Module):
    """
    Frequency-axis Instance Normalization for STFT spectrograms.

    Normalizes along the frequency dimension (dim=2) per (sample, channel, timestep),
    removing dataset-specific absolute frequency amplitude biases while preserving
    relative spectral patterns (fault harmonics, BPFO peaks).

    Theoretical basis:
      - Kim et al. "Relaxed Instance Frequency-wise Normalization (RFN)", Interspeech 2022
        (DCASE 2021 Task1A champion, +18.2pp on unseen recording devices)
      - In audio/vibration spectrograms, cross-domain discrepancy is encoded primarily
        along the frequency axis (not channel statistics), making F-axis normalization
        more effective than BN, GN, or IN for cross-dataset transfer.

    For bearing fault diagnosis:
      - BPFO shifts in absolute bin position across datasets (different fs, bearing geometry)
      - FreqLayerNorm converts absolute spectral amplitudes → relative spectral patterns
      - Batch-independent: no global statistics, stable across CWRU (small batch)
        and PU (large, high-fs batch) training phases.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T]
        mean = x.mean(dim=2, keepdim=True)                           # [B, C, 1, T]
        std = x.std(dim=2, keepdim=True, unbiased=False)             # [B, C, 1, T]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FreqAttnPool(nn.Module):
    """
    Learnable frequency-axis attention pooling with scaled-score stabilization.

    ATT164: removes ATT163's failed ReZero residual (α went negative → anti-attention),
    replacing it with standard attention score scaling of 1/sqrt(F'), analogous to
    Vaswani et al. (2017) Scaled Dot-Product Attention.

    Mathematical motivation:
      - At init, gate weights w=0 → scores=0 → softmax=uniform → output=mean_pooled.
      - After E1, Adam updates gate to w~O(1e-2).  Raw scores ~ O(1e-2×F_val).
        Without scaling: softmax receives logits ~ O(1e-2), distribution still near-uniform.
      - BUT Adam's v_t (second moment) starts from 0; effective LR = η/(sqrt(v_t)+ε) ≈ η/ε
        is ~1000× larger in E1 than steady-state.  Gate weight updates in E1 alone can reach
        O(0.1-0.3), causing scores to become O(0.1-0.3), softmax peaking sharply in E2.
      - 1/sqrt(F') scaling: if F'=64, divides scores by 8. This does NOT prevent the gate
        from learning, but it multiplies the "effective temperature" of softmax by sqrt(F'),
        keeping the distribution closer to uniform for the first several epochs while the
        downstream representation layers adapt. By E5-E8 (when Adam v_t stabilises),
        the gate learns meaningful frequency attention at a bounded distribution shift rate.
      - No learnable parameters added. No risk of negative-value feedback loops.
      - Ref: Vaswani et al. "Attention Is All You Need" (NeurIPS 2017) — score scaling
             prevents softmax saturation and stabilises gradient magnitudes;
             Kim & Misu "Frequency-Aware Attention" (Interspeech 2022) — frequency-selective
             pooling in audio spectrograms;
             Frequency Channel-Attention ViT (Expert Sys. Appl. 2025).

    Why ATT163 (ReZero) failed:
      - ReZero assumes the gated module has initial expressive power (e.g., Transformer block).
        FreqAttnPool.gate starts from random/zero init → attn_pooled ≈ mean_pooled at start,
        but gradient of α: ∂L/∂α = ∂L/∂output × (attn_pooled - mean_pooled) > 0 when
        attn_pooled is slightly worse than mean_pooled (as expected for an untrained gate).
        This pushes α negative, creating an anti-attention loop. PU α stabilised at −0.037.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        # Depthwise 1×1 conv — per-channel frequency scoring, no cross-channel mixing.
        # groups=num_channels keeps parameters minimal (num_channels scalars + biases).
        self.gate = nn.Conv2d(
            num_channels, num_channels,
            kernel_size=1, groups=num_channels, bias=True,
        )
        # Zero-init gate: softmax(0…0) = uniform → output = mean_pooled at start.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F', T']
        F_prime = x.shape[2]
        scores = self.gate(x)                                      # [B, C, F', T']
        scores = scores / math.sqrt(F_prime)                       # scaled: prevents early softmax saturation
        attn = torch.softmax(scores, dim=2)                        # normalize over F' → [B, C, F', T']
        return (x * attn).sum(dim=2)                               # [B, C, T']


class GroupedFreqPool(nn.Module):
    """
    Two-group frequency pooling: low-frequency bands vs high-frequency bands.

    Designed for SWT>=10 where low-frequency bands carry most fault-discriminative cues.
    We keep a tiny learnable gate (2 logits) to adapt low/high contribution per task.
    """

    def __init__(self, low_bands: int = 5, high_bands: int = 6):
        super().__init__()
        self.low_bands = max(1, int(low_bands))
        self.high_bands = max(1, int(high_bands))
        # Initialize with low-frequency prior (0.7 / 0.3 after softmax-like normalization).
        self.group_logits = nn.Parameter(torch.tensor([0.7, 0.3], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F', T']
        f_dim = int(x.shape[2])
        low_take = min(self.low_bands, f_dim)
        high_take = min(self.high_bands, f_dim)

        # By construction in SWT stack, lower frequencies are closer to the tail bins.
        low_feat = x[:, :, f_dim - low_take :, :].mean(dim=2)   # [B, C, T']
        high_feat = x[:, :, :high_take, :].mean(dim=2)          # [B, C, T']

        w = torch.softmax(self.group_logits, dim=0)             # [2]
        return w[0] * low_feat + w[1] * high_feat


class SEBlock(nn.Module):
    """Channel recalibration block for spectral feature emphasis."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


class DepthwiseSeparableConvBlock(nn.Module):
    """Lightweight Conv block for spectrogram feature extraction."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride_t=1,
        dropout=0.0,
        use_ibn=False,
        ibn_ratio=0.5,
        use_se=False,
        se_ratio=4,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=(1, stride_t),
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = IBN(out_channels, ibn_ratio=ibn_ratio) if use_ibn else nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=se_ratio) if use_se else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.se(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class SpecEncoderLight(nn.Module):
    """
    Lightweight Spec encoder.

    Interface is aligned with existing training/evaluation calls:
    - forward(spec, spec_mask, domain_ids=None, return_tokens=False, return_logits=False, ...)
    """

    def __init__(
        self,
        in_channels,
        embed_dim=256,
        use_domain_residual=False,
        num_domains=3,
        base_channels=32,
        dropout=0.1,
        num_classes=4,
        use_classification_head=False,
    ):
        super().__init__()
        self.use_domain_residual = use_domain_residual
        self.use_classification_head = use_classification_head
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        try:
            import config as cfg

            self.use_ibn_spec = bool(getattr(cfg, "USE_IBN_SPEC", True))
            self.ibn_spec_ratio = float(getattr(cfg, "IBN_SPEC_RATIO", 0.5))
            self.use_ibn_spec_stem_only = bool(getattr(cfg, "USE_IBN_SPEC_STEM_ONLY", True))
            self.use_gn_spec_stem = bool(getattr(cfg, "USE_GN_SPEC_STEM", False))
            self.use_freqln_spec_stem = bool(getattr(cfg, "USE_FREQLN_SPEC_STEM", False))
            self.use_freq_attn_pool = bool(getattr(cfg, "USE_FREQ_ATTN_POOL", False))
            self.use_grouped_freq_pool = bool(getattr(cfg, "USE_GROUPED_FREQ_POOL", False))
            self.grouped_freq_pool_low_bands = int(getattr(cfg, "GROUPED_FREQ_POOL_LOW_BANDS", 5))
            self.grouped_freq_pool_high_bands = int(getattr(cfg, "GROUPED_FREQ_POOL_HIGH_BANDS", 6))
            self.use_mixstyle_spec = bool(getattr(cfg, "USE_MIXSTYLE_SPEC", False))
            self.mixstyle_p = float(getattr(cfg, "MIXSTYLE_P", 0.5))
            self.mixstyle_alpha = float(getattr(cfg, "MIXSTYLE_ALPHA", 0.3))
            self.domain_cond_strength = float(getattr(cfg, "DOMAIN_COND_STRENGTH", 1.0))
            self.domain_cond_strength = max(0.0, min(1.0, self.domain_cond_strength))
            self.use_multi_scale_fusion = bool(getattr(cfg, "USE_SPEC_MULTI_SCALE_FUSION", False))
            self.use_temporal_attn_pool = bool(getattr(cfg, "USE_TEMPORAL_ATTN_POOL", False))
            # ATT173: Domain-Invariant Disentanglement
            self.use_spec_disentangle = bool(getattr(cfg, "USE_SPEC_DISENTANGLE", False))
            self.disentangle_inv_dim = int(getattr(cfg, "SPEC_DISENTANGLE_INV_DIM", embed_dim))
            self.disentangle_spe_dim = int(getattr(cfg, "SPEC_DISENTANGLE_SPE_DIM", 64))
            # ATT174: 输入级 SpecAugment（频率掩码 + 时间掩码）
            self.use_spec_augment = bool(getattr(cfg, "USE_SPEC_AUGMENT", False))
            self.spec_augment_freq_mask_f = float(getattr(cfg, "SPEC_AUGMENT_FREQ_MASK_F", 0.10))
            self.spec_augment_time_mask_t = float(getattr(cfg, "SPEC_AUGMENT_TIME_MASK_T", 0.10))
            self.spec_augment_num_freq_masks = int(getattr(cfg, "SPEC_AUGMENT_NUM_FREQ_MASKS", 1))
            self.spec_augment_num_time_masks = int(getattr(cfg, "SPEC_AUGMENT_NUM_TIME_MASKS", 1))
            self.spec_augment_mask_prob = float(getattr(cfg, "SPEC_AUGMENT_MASK_PROB", 0.5))
            self.use_detached_spec_align_proj = bool(getattr(cfg, "USE_DETACHED_SPEC_ALIGN_PROJ", False))
            self.train_log_specenc_norm_stats = bool(getattr(cfg, "TRAIN_LOG_SPECENC_NORM_STATS", True))
            self.train_log_specenc_norm_step_stride = int(getattr(cfg, "TRAIN_LOG_SPECENC_NORM_STEP_STRIDE", 100))
            # Core refactor: reduce domain-dependent absolute amplitude shortcuts before head.
            self.use_spec_instance_norm_before_head = bool(
                getattr(cfg, "SPEC_INSTANCE_NORM_BEFORE_HEAD", True)
            )
            # Domain-agnostic spectral pattern feature: adjacent SWT-band energy ratios.
            self.use_freq_ratio_features = bool(getattr(cfg, "SPEC_USE_FREQ_RATIO_FEATURES", True))
            self.spec_freq_ratio_eps = float(getattr(cfg, "SPEC_FREQ_RATIO_EPS", 1e-6))
            self.spec_freq_ratio_blend = float(getattr(cfg, "SPEC_FREQ_RATIO_BLEND", 0.35))
            # Physical prior: compensate sampling-rate induced amplitude bias.
            self.use_spec_fs_aware_norm = bool(getattr(cfg, "SPEC_FS_AWARE_NORM_ENABLE", True))
            self.spec_fs_aware_ref_fs = float(getattr(cfg, "SPEC_FS_AWARE_REF_FS", 25600.0))
            self.spec_fs_aware_power = float(getattr(cfg, "SPEC_FS_AWARE_NORM_POWER", 0.5))
            self.log_spec_fs_aware_norm = bool(getattr(cfg, "SPEC_FS_AWARE_LOG", True))
            self.domain_fs_map = {
                int(getattr(cfg, "CWRU_GK2_DOMAIN_ID", 0)): float(getattr(cfg, "SAMPLING_RATES", {}).get("CWRU", 12000)),
                int(getattr(cfg, "CWRU_GK4_DOMAIN_ID", 1)): float(getattr(cfg, "SAMPLING_RATES", {}).get("CWRU", 12000)),
                int(getattr(cfg, "PU_DOMAIN_ID", 2)): float(getattr(cfg, "SAMPLING_RATES", {}).get("PU", 64000)),
                int(getattr(cfg, "HUST_DOMAIN_ID", 3)): float(getattr(cfg, "SAMPLING_RATES", {}).get("HUST", 25600)),
                int(getattr(cfg, "HIT_DOMAIN_ID", 4)): float(getattr(cfg, "SAMPLING_RATES", {}).get("HIT", 25000)),
            }
            # ATT423-P2: Frequency-Conditioned FiLM
            self.use_fs_cond_film = bool(getattr(cfg, "SPEC_FS_COND_FILM_ENABLE", False))
            self._fs_cond_film_hidden = int(getattr(cfg, "SPEC_FS_COND_FILM_HIDDEN_DIM", 32))
            self._fs_cond_film_ref_fs = float(getattr(cfg, "SPEC_FS_COND_FILM_REF_FS", 64000.0))
            # ATT465: Domain-Conditioned Normalization + Multi-Level Alignment
            self._domain_bn_enable = bool(getattr(cfg, "DOMAIN_BN_ENABLE", False))
            self._mlsa_enable = bool(getattr(cfg, "MLSA_ENABLE", False))
        except Exception:
            self.use_ibn_spec = True
            self.ibn_spec_ratio = 0.5
            self.use_ibn_spec_stem_only = True
            self.use_gn_spec_stem = False
            self.use_freqln_spec_stem = False
            self.use_freq_attn_pool = False
            self.use_grouped_freq_pool = False
            self.grouped_freq_pool_low_bands = 5
            self.grouped_freq_pool_high_bands = 6
            self.use_mixstyle_spec = False
            self.mixstyle_p = 0.5
            self.mixstyle_alpha = 0.3
            self.domain_cond_strength = 0.5
            self.use_multi_scale_fusion = False
            self.use_temporal_attn_pool = False
            self.use_spec_disentangle = False
            self.disentangle_inv_dim = embed_dim
            self.disentangle_spe_dim = 64
            self.use_spec_augment = False
            self.spec_augment_freq_mask_f = 0.10
            self.spec_augment_time_mask_t = 0.10
            self.spec_augment_num_freq_masks = 1
            self.spec_augment_num_time_masks = 1
            self.spec_augment_mask_prob = 0.5
            self.use_detached_spec_align_proj = False
            self.train_log_specenc_norm_stats = True
            self.train_log_specenc_norm_step_stride = 100
            # ATT377: defaults must match config.py (all disabled after revert)
            self.use_spec_instance_norm_before_head = False
            self.use_freq_ratio_features = False
            self.spec_freq_ratio_eps = 1e-6
            self.spec_freq_ratio_blend = 0.35
            self.use_spec_fs_aware_norm = False
            self.spec_fs_aware_ref_fs = 25600.0
            self.spec_fs_aware_power = 0.5
            self.log_spec_fs_aware_norm = False
            self.domain_fs_map = {0: 12000.0, 1: 12000.0, 2: 64000.0, 3: 25600.0, 4: 25000.0}
            self.use_fs_cond_film = False
            self._fs_cond_film_hidden = 32
            self._fs_cond_film_ref_fs = 64000.0
            self._domain_bn_enable = False
            self._mlsa_enable = False
        self.train_log_specenc_norm_step_stride = max(1, int(self.train_log_specenc_norm_step_stride))
        self._specenc_forward_step = 0

        # ATT377-StatFiLM: numerical stat vector conditioning
        try:
            import config as _cfg_sf
            self.use_stat_film = bool(getattr(_cfg_sf, "USE_STAT_FILM", False))
            _sf_n_feat = int(getattr(_cfg_sf, "STAT_FILM_N_FEATURES", 8))
            _sf_hidden = int(getattr(_cfg_sf, "STAT_FILM_HIDDEN_DIM", 64))
        except Exception:
            self.use_stat_film = False
            _sf_n_feat = 8
            _sf_hidden = 64
        c2 = base_channels * 2
        if self.use_stat_film:
            self.stat_film = StatFiLMGenerator(
                n_stat_features=_sf_n_feat,
                n_channels=c2,
                hidden_dim=_sf_hidden,
            )
        else:
            self.stat_film = None
        self._stat_film_gate_ema = 0.0
        self._stat_film_log_count = 0

        # ATT423-P2: Frequency-Conditioned FiLM (applied after stem and block2)
        if self.use_fs_cond_film:
            self.fs_film_stem = FsConditioningFiLM(
                c1, hidden_dim=self._fs_cond_film_hidden, ref_fs=self._fs_cond_film_ref_fs,
            )
            self.fs_film_block2 = FsConditioningFiLM(
                c2, hidden_dim=self._fs_cond_film_hidden, ref_fs=self._fs_cond_film_ref_fs,
            )
        else:
            self.fs_film_stem = None
            self.fs_film_block2 = None
        self._fs_film_log_count = 0

        def _build_stem_norm(channels: int) -> nn.Module:
            if self._domain_bn_enable:
                from models.domain_bn import DomainBN
                _dbn = DomainBN(channels, num_domains=num_domains)
                print(f"[ATT465-DomainBN] stem norm = DomainBN(C={channels}, D={num_domains})")
                return _dbn
            if self.use_ibn_spec:
                return IBN(channels, ibn_ratio=self.ibn_spec_ratio)
            if self.use_freqln_spec_stem:
                return FreqLayerNorm(channels)
            if self.use_gn_spec_stem:
                num_groups = min(8, channels)
                while channels % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
            return nn.BatchNorm2d(channels)

        _stem_norm = _build_stem_norm(c1)
        _stem_norm_name = type(_stem_norm).__name__
        self._stem_conv = nn.Conv2d(
            in_channels,
            c1,
            kernel_size=(3, 7),
            stride=(1, 2),
            padding=(1, 3),
            bias=False,
        )
        self._stem_norm = _stem_norm
        self._stem_act = nn.SiLU(inplace=True)
        self.stem = nn.Sequential(self._stem_conv, self._stem_norm, self._stem_act)
        self._stem_uses_domain_bn = self._domain_bn_enable
        print(f"[ATT394][SpecEncoderLight] stem_norm={_stem_norm_name}, "
              f"ibn_spec={int(self.use_ibn_spec)}, ratio={self.ibn_spec_ratio}, "
              f"stem_only={int(self.use_ibn_spec_stem_only)}, "
              f"fs_cond_film={int(self.use_fs_cond_film)}")
        use_ibn_blocks = self.use_ibn_spec and (not self.use_ibn_spec_stem_only)
        self.block2 = DepthwiseSeparableConvBlock(
            c1,
            c2,
            stride_t=2,
            dropout=dropout,
            use_ibn=use_ibn_blocks,
            ibn_ratio=self.ibn_spec_ratio,
            use_se=False,
        )
        self.block3 = DepthwiseSeparableConvBlock(
            c2,
            c3,
            stride_t=2,
            dropout=dropout,
            use_ibn=use_ibn_blocks,
            ibn_ratio=self.ibn_spec_ratio,
            use_se=False,
        )
        # ATT149 P1: 对齐 Signal 分支做法，在浅层引入 MixStyle（训练时启用、推理时自动关闭）。
        self.mixstyle = (
            MixStyle(p=self.mixstyle_p, alpha=self.mixstyle_alpha)
            if self.use_mixstyle_spec
            else None
        )

        if use_domain_residual:
            self.domain_residual = DomainFiLM(
                num_domains=num_domains,
                num_features=c3,
                init_scale=0.0,
            )
        else:
            self.domain_residual = None

        # ATT162 P1: FreqAttnPool — 可学习频率注意力池化，替换 mean(dim=2)。
        # 零初始化保证启动行为等价于 mean pooling，逐步学习 BPFO/BPFI/BSF 相关频带权重。
        self.freq_attn_pool = FreqAttnPool(c3) if self.use_freq_attn_pool else None
        self.grouped_freq_pool = (
            GroupedFreqPool(
                low_bands=self.grouped_freq_pool_low_bands,
                high_bands=self.grouped_freq_pool_high_bands,
            )
            if self.use_grouped_freq_pool
            else None
        )
        self.ms_lateral_stem = None
        self.ms_lateral_block2 = None
        self.ms_scale_logits = None
        self.temporal_attn_pool = None

        # ATT138 P0: 回退到 ATT134 稳定结构（mean 频率池化 + 128→256 信息瓶颈）
        # 说明：ATT136/137 的残差门控改变了训练轨迹并造成跨工况泛化退化，先恢复已验证基线。
        self.proj = nn.Linear(c3, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # ATT465-MLSA: auxiliary mid-level projection for multi-level spec-text alignment
        if self._mlsa_enable:
            self._mlsa_pool = nn.AdaptiveAvgPool2d(1)
            self._mlsa_proj = nn.Sequential(
                nn.Linear(c2, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            print(f"[ATT465-MLSA] mid-level alignment enabled: c2={c2} -> embed_dim={embed_dim}")
        else:
            self._mlsa_pool = None
            self._mlsa_proj = None

        self.align_proj = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            if self.use_detached_spec_align_proj
            else None
        )
        if self.align_proj is not None:
            # Identity-biased init: keep early geometry close to z_spec,
            # then let align_proj learn semantic correction progressively.
            try:
                _lin = self.align_proj[0]
                if isinstance(_lin, nn.Linear) and _lin.in_features == _lin.out_features:
                    nn.init.eye_(_lin.weight)
                    if _lin.bias is not None:
                        nn.init.zeros_(_lin.bias)
            except Exception:
                pass
        # ATT239(S1): use linear projection for SupCon to reduce gradient attenuation
        # caused by the previous MLP bottleneck.
        self.supcon_proj = nn.Linear(embed_dim, embed_dim)

        # ATT173: 域不变特征解耦双投影头
        # proj_inv 替代 proj 作为主投影（输出 z_spec 给所有下游路径），接口与原 proj 完全兼容；
        # proj_spe 为辅助投影（域特异特征，仅用于训练损失，推理时丢弃）；
        # dc_spe/dc_inv 为两个独立的小型域分类器（训练时分别用于 z_spe 域识别和 z_inv 域对抗）。
        if self.use_spec_disentangle:
            num_domains_dc = 5  # CWRU_gk2=0, CWRU_gk4=1, PU=2, HUST=3, HIT=4
            self.proj_inv = nn.Linear(c3, self.disentangle_inv_dim)
            self.norm_inv = nn.LayerNorm(self.disentangle_inv_dim)
            self.proj_spe = nn.Linear(c3, self.disentangle_spe_dim)
            self.dc_spe = DomainDiscriminator(
                in_dim=self.disentangle_spe_dim,
                num_domains=num_domains_dc,
                hidden_dim=128,
                dropout=0.1,
            )
            self.dc_inv = DomainDiscriminator(
                in_dim=self.disentangle_inv_dim,
                num_domains=num_domains_dc,
                hidden_dim=128,
                dropout=0.1,
            )
            # Runtime buffers for training loop to retrieve (set in forward, no grad)
            self._last_z_inv_raw = None
            self._last_z_spe_raw = None
        else:
            self.proj_inv = None
            self.norm_inv = None
            self.proj_spe = None
            self.dc_spe = None
            self.dc_inv = None
            self._last_z_inv_raw = None
            self._last_z_spe_raw = None
        self._last_z_spec_pre_norm = None

        # ATT429-P2: Hierarchical diagnosis head config
        try:
            import config as _cfg_h
            self._use_hierarchical_spec_head = bool(int(os.environ.get(
                "USE_HIERARCHICAL_SPEC_HEAD",
                str(int(getattr(_cfg_h, "USE_HIERARCHICAL_SPEC_HEAD", True)))
            )))
            self._hierarchical_nf_temperature = float(getattr(_cfg_h, "HIERARCHICAL_NF_TEMPERATURE", 0.07))
            self._hierarchical_fault_temperature = float(getattr(_cfg_h, "HIERARCHICAL_FAULT_TEMPERATURE", 0.07))
            self._hierarchical_fault_classes = int(getattr(_cfg_h, "HIERARCHICAL_FAULT_CLASSES", 3))
        except Exception:
            self._use_hierarchical_spec_head = True
            self._hierarchical_nf_temperature = 0.07
            self._hierarchical_fault_temperature = 0.07
            self._hierarchical_fault_classes = 3

        # ATT431: Dual disentangle removed (falsified in ATT430: F_DD)
        self._use_dual_disentangle = False
        self._last_z_spe_dual = None

        # ATT446: WST bypass dimension — widens classifier input when WST features are injected
        self._wst_bypass_dim = int(getattr(cfg, "WST_BYPASS_DIM", 0)) if bool(getattr(cfg, "WST_BYPASS_INJECT", False)) else 0
        # ATT449-S3: Envelope bypass dimension + BPFI/BPFO ratio scalar
        self._env_bypass_dim = int(getattr(cfg, "ENV_BYPASS_DIM", 0)) if bool(getattr(cfg, "ENV_BYPASS_INJECT", False)) else 0
        self._env_ratio_dim = 1 if self._env_bypass_dim > 0 else 0
        # ATT464b: SCD bypass dimension
        self._scd_bypass_dim = int(getattr(cfg, "SCD_BYPASS_DIM", 0)) if bool(getattr(cfg, "SCD_BYPASS_INJECT", False)) else 0

        # ATT450-S1: Learnable scale parameters for bypass feature normalization.
        # z_spec_pre_norm has L2 norm ~22; bypass features are 4-50x smaller per-dim.
        # L2-normalize each bypass branch, then scale to match z_spec magnitude.
        _spec_norm_init = float(embed_dim) ** 0.5
        if self._wst_bypass_dim > 0:
            self._wst_scale = nn.Parameter(torch.tensor(_spec_norm_init))
        else:
            self._wst_scale = None
        if self._env_bypass_dim > 0:
            self._env_scale = nn.Parameter(torch.tensor(_spec_norm_init))
        else:
            self._env_scale = None
        if self._env_ratio_dim > 0:
            self._ratio_scale = nn.Parameter(torch.tensor(5.0))
        else:
            self._ratio_scale = None
        if self._scd_bypass_dim > 0:
            self._scd_scale = nn.Parameter(torch.tensor(_spec_norm_init))
        else:
            self._scd_scale = None
        self._bypass_scale_log_count = 0

        _spec_head_mode = str(getattr(cfg, "SPEC_HEAD_MODE", "mlp")).strip().lower()
        self._spec_head_mode = _spec_head_mode

        if use_classification_head and _spec_head_mode == "proto":
            self._spec_proto_temperature = float(getattr(cfg, "SPEC_PROTO_TEMPERATURE", 0.1))
            self._spec_proto_margin = float(getattr(cfg, "SPEC_PROTO_MARGIN", 0.0))
            self.spec_prototypes = nn.Parameter(
                self._init_simplex_etf(num_classes, embed_dim)
            )
            self.classifier = None
            self.nf_head = None
            self.fault_head = None
            self._nf_temperature = None
            self._fault_temperature = None
            print(f"[ATT453-SpecHead] Proto cosine classifier, C={num_classes}, D={embed_dim}, "
                  f"T={self._spec_proto_temperature:.2f}, margin={self._spec_proto_margin:.2f}, "
                  f"init=simplex_etf(inter_cos={-1.0/(num_classes-1):.3f})")
        elif use_classification_head:
            _cls_in = embed_dim + self._wst_bypass_dim + self._env_bypass_dim + self._env_ratio_dim + self._scd_bypass_dim
            self.classifier = nn.Sequential(
                nn.Linear(_cls_in, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes),
            )
            self.spec_prototypes = None
            _parts = [f"spec={embed_dim}"]
            if self._wst_bypass_dim > 0:
                _parts.append(f"wst={self._wst_bypass_dim}(scale_init={_spec_norm_init:.1f})")
            if self._env_bypass_dim > 0:
                _parts.append(f"env={self._env_bypass_dim}(scale_init={_spec_norm_init:.1f})")
            if self._env_ratio_dim > 0:
                _parts.append(f"ratio={self._env_ratio_dim}(scale_init=5.0)")
            if self._scd_bypass_dim > 0:
                _parts.append(f"scd={self._scd_bypass_dim}(scale_init={_spec_norm_init:.1f})")
            print(f"[ATT450-SpecHead] MLP classifier, C={num_classes}, D_in={_cls_in} ({'+'.join(_parts)})")

            # ATT429-P2: Hierarchical NF/Fault heads (kept as MLP to match S0)
            if self._use_hierarchical_spec_head:
                self.nf_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim // 2, 2),
                )
                self._nf_temperature = None

                n_fault = self._hierarchical_fault_classes
                self.fault_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim // 2, n_fault),
                )
                self._fault_temperature = None
                print(f"[ATT430-S0] Hierarchical heads: nf_head(2, MLP), fault_head({n_fault}, MLP)")
            else:
                self.nf_head = None
                self.fault_head = None
                self._nf_temperature = None
                self._fault_temperature = None
        else:
            self.classifier = None
            self.spec_prototypes = None
            self._spec_head_temperature = None
            self.nf_head = None
            self.fault_head = None
            self._nf_temperature = None
            self._fault_temperature = None

    @property
    def has_spec_head(self) -> bool:
        """ATT453: unified check for SpecHead availability (MLP or Proto mode)."""
        return (self.classifier is not None) or (self.spec_prototypes is not None)

    @staticmethod
    def _init_simplex_etf(n_classes: int, embed_dim: int) -> torch.Tensor:
        """Initialize prototypes as simplex ETF vertices (maximum equiangular separation).
        For C classes: inter_cos = -1/(C-1). E.g. C=4 → inter_cos = -0.333.
        """
        M = torch.randn(embed_dim, n_classes)
        Q, _ = torch.linalg.qr(M)
        proto = Q[:, :n_classes].T
        proto = proto - proto.mean(dim=0, keepdim=True)
        proto = F.normalize(proto, dim=-1)
        _cos = proto @ proto.T
        _inter = (_cos.sum() - _cos.trace()) / (n_classes * (n_classes - 1))
        print(f"[ATT453-ETF] Init {n_classes} prototypes in {embed_dim}D, inter_cos={_inter:.4f}")
        return proto

    def _domain_ids_to_fs_tensor(self, domain_ids: torch.Tensor, device) -> torch.Tensor:
        """Convert domain_id tensor [B] to sampling rate tensor [B] using domain_fs_map."""
        did = domain_ids.detach().view(-1).long().cpu().tolist()
        fs_list = [float(self.domain_fs_map.get(int(d), 25600.0)) for d in did]
        return torch.tensor(fs_list, device=device, dtype=torch.float32)

    def _downsample_mask(self, spec_mask):
        mask = spec_mask.unsqueeze(1)  # [B, 1, T]
        for _ in range(3):
            mask = F.avg_pool1d(mask, kernel_size=2, stride=2, ceil_mode=True)
        return mask.squeeze(1)  # [B, T']

    def _apply_fs_aware_norm(self, spec: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        if (not self.use_spec_fs_aware_norm) or (not torch.is_tensor(domain_ids)) or spec.dim() != 4:
            return spec
        if int(spec.size(0)) != int(domain_ids.numel()):
            return spec
        ref_fs = float(max(1.0, self.spec_fs_aware_ref_fs))
        pwr = float(self.spec_fs_aware_power)
        did = domain_ids.detach().view(-1).long().cpu().tolist()
        scales = []
        for _d in did:
            _fs = float(self.domain_fs_map.get(int(_d), ref_fs))
            _scale = (ref_fs / max(1.0, _fs)) ** pwr
            scales.append(_scale)
        scale_t = torch.tensor(scales, device=spec.device, dtype=spec.dtype).view(-1, 1, 1, 1)
        return spec * scale_t

    def _compute_freq_ratio_summary(self, x_4d: torch.Tensor) -> torch.Tensor:
        if (not self.use_freq_ratio_features) or (not torch.is_tensor(x_4d)) or x_4d.dim() != 4:
            return None
        if int(x_4d.size(2)) <= 1:
            return None
        eps = float(max(1e-8, self.spec_freq_ratio_eps))
        # [B, C, F] energy profile on frequency axis.
        freq_energy = x_4d.abs().mean(dim=-1)
        num = freq_energy[:, :, 1:]
        den = freq_energy[:, :, :-1].clamp_min(eps)
        ratio = (num / den).clamp_min(eps)
        # Log-ratio is more stable across domains than raw amplitudes.
        ratio_log = torch.log(ratio)
        # [B, C], summarize adjacent-band trends.
        ratio_summary = ratio_log.mean(dim=-1)
        return ratio_summary

    def forward(
        self,
        spec,
        spec_mask,
        domain_ids=None,
        return_tokens=False,
        return_logits=False,
        text_emb=None,
        return_moe_stats=False,
        stat_vec=None,
        wst_aux=None,
        env_aux=None,
        ratio_aux=None,
        scd_aux=None,
    ):
        del text_emb  # keep forward signature compatibility

        if self.use_spec_fs_aware_norm and torch.is_tensor(domain_ids) and spec.dim() == 4:
            spec = self._apply_fs_aware_norm(spec, domain_ids)

        # ATT174: 输入级 SpecAugment（仅 training 模式生效，eval 时恒等，零分布偏移）
        # spec: [B, C, F, T] — per-sample 逐样本独立随机掩码（不同于 batch 级 MixStyle）
        if self.training and self.use_spec_augment and spec.dim() == 4:
            spec = spec.clone()  # 避免 in-place 修改原始 tensor
            B, _C, F_dim, T_dim = spec.shape
            for b in range(B):
                # 频率掩码：随机遮盖连续 freq bins，模拟频率偏移不变性
                for _ in range(self.spec_augment_num_freq_masks):
                    if random.random() < self.spec_augment_mask_prob:
                        f_max = max(1, int(self.spec_augment_freq_mask_f * F_dim))
                        f = random.randint(1, f_max)
                        f0 = random.randint(0, max(0, F_dim - f))
                        spec[b, :, f0:f0 + f, :] = 0.0
                # 时间掩码：随机遮盖连续 time steps，提升时间鲁棒性
                for _ in range(self.spec_augment_num_time_masks):
                    if random.random() < self.spec_augment_mask_prob:
                        t_max = max(1, int(self.spec_augment_time_mask_t * T_dim))
                        t = random.randint(1, t_max)
                        t0 = random.randint(0, max(0, T_dim - t))
                        spec[b, :, :, t0:t0 + t] = 0.0

        # ATT465: DomainBN needs domain_ids passed explicitly; skip nn.Sequential path
        if self._stem_uses_domain_bn and torch.is_tensor(domain_ids):
            x_stem = self._stem_conv(spec)
            x_stem = self._stem_norm(x_stem, domain_ids)
            x_stem = self._stem_act(x_stem)
        else:
            x_stem = self.stem(spec)
        _gate_bank = get_gate_bank(self)
        # ATT423-P2: FiLM conditioning after stem
        if self.fs_film_stem is not None and torch.is_tensor(domain_ids):
            _fs_vals = self._domain_ids_to_fs_tensor(domain_ids, spec.device)
            _x_filmed = self.fs_film_stem(x_stem, _fs_vals)
            _s = float(getattr(self, "domain_cond_strength", 1.0))
            _x_cond = x_stem + _s * (_x_filmed - x_stem)
            x_stem = apply_residual_gate(_gate_bank, "domain_cond_fs_film", _x_cond, x_stem)
        if self.mixstyle is not None:
            _x_ms = self.mixstyle(x_stem)
            x_stem = apply_residual_gate(_gate_bank, "domain_cond_mixstyle_spec", _x_ms, x_stem)
        x_block2 = self.block2(x_stem)
        # ATT423-P2: FiLM conditioning after block2
        if self.fs_film_block2 is not None and torch.is_tensor(domain_ids):
            _fs_vals = self._domain_ids_to_fs_tensor(domain_ids, spec.device)
            _x_filmed = self.fs_film_block2(x_block2, _fs_vals)
            _s = float(getattr(self, "domain_cond_strength", 1.0))
            _x_cond = x_block2 + _s * (_x_filmed - x_block2)
            x_block2 = apply_residual_gate(_gate_bank, "domain_cond_fs_film", _x_cond, x_block2)
            if self.training and self._fs_film_log_count < 3:
                self._fs_film_log_count += 1
                _g_stem = self.fs_film_stem.net[-1].weight.detach()
                _g_b2 = self.fs_film_block2.net[-1].weight.detach()
                logger.info(
                    f"[ATT423-FsFilm] step={self._fs_film_log_count}, "
                    f"stem_w_norm={float(_g_stem.norm()):.6f}, "
                    f"block2_w_norm={float(_g_b2.norm()):.6f}, "
                    f"fs_values={_fs_vals[:4].tolist()}"
                )
        if self.mixstyle is not None:
            _x_ms = self.mixstyle(x_block2)
            x_block2 = apply_residual_gate(_gate_bank, "domain_cond_mixstyle_spec", _x_ms, x_block2)

        # ATT377-StatFiLM: apply FiLM conditioning + residual gate after block2
        if self.use_stat_film and self.stat_film is not None and stat_vec is not None:
            if torch.is_tensor(stat_vec) and stat_vec.dim() == 2 and stat_vec.size(0) == x_block2.size(0):
                _gamma, _beta, _gate = self.stat_film(stat_vec)
                _s = float(getattr(self, "domain_cond_strength", 1.0))
                _gate = _gate * _s
                x_block2_raw = x_block2
                x_block2_film = _gamma * x_block2 + _beta
                _x_stat = _gate * x_block2_film + (1.0 - _gate) * x_block2_raw
                x_block2 = apply_residual_gate(_gate_bank, "domain_cond_stat_film", _x_stat, x_block2_raw)
                # EMA tracking for diagnostics
                self._stat_film_gate_ema = 0.95 * self._stat_film_gate_ema + 0.05 * float(_gate.detach().mean().cpu().item())
                self._stat_film_log_count += 1
                if self.training and self._stat_film_log_count <= 3:
                    print(
                        f"[StatFiLM] step={self._stat_film_log_count}, "
                        f"gate_mean={float(_gate.detach().mean().cpu().item()):.6f}, "
                        f"gamma_std={float(_gamma.detach().std().cpu().item()):.6f}, "
                        f"beta_std={float(_beta.detach().std().cpu().item()):.6f}, "
                        f"stat_vec_mean={float(stat_vec.detach().mean().cpu().item()):.4f}, "
                        f"stat_vec_std={float(stat_vec.detach().std().cpu().item()):.4f}"
                    )

        # ATT465-MLSA: compute mid-level representation from block2 output
        _z_mid = None
        if self._mlsa_pool is not None and self._mlsa_proj is not None:
            _mid_pooled = self._mlsa_pool(x_block2).squeeze(-1).squeeze(-1)  # [B, c2]
            _z_mid = self._mlsa_proj(_mid_pooled)  # [B, embed_dim]

        x_block3 = self.block3(x_block2)  # [B, C, F', T']
        x = x_block3

        if self.use_domain_residual and domain_ids is not None:
            _x_dr = self.domain_residual(x, domain_ids)
            _s = float(getattr(self, "domain_cond_strength", 1.0))
            _x_cond = x + _s * (_x_dr - x)
            x = apply_residual_gate(_gate_bank, "domain_cond_domain_film_spec", _x_cond, x)

        if self.use_spec_instance_norm_before_head and x.dim() == 4:
            # Per-sample/channel instance norm on [F', T'] reduces absolute-domain amplitude bias.
            x = F.instance_norm(x)

        ratio_summary = self._compute_freq_ratio_summary(x)

        # P3: 优先频率分组池化，其次频率注意力池化，否则回退到 mean 频率池化。
        if self.grouped_freq_pool is not None:
            x = self.grouped_freq_pool(x)  # [B, C, T']  ← low/high grouped pooling
        elif self.freq_attn_pool is not None:
            # ATT162 P1: 频率注意力池化
            x = self.freq_attn_pool(x)  # [B, C, T']  ← learned freq-weighted pooling
        else:
            x = x.mean(dim=2)           # [B, C, T']  ← uniform mean pooling (ATT138 baseline)

        mask = self._downsample_mask(spec_mask)
        x = x * mask.unsqueeze(1)

        if return_tokens:
            x_tokens = x.permute(0, 2, 1).contiguous()  # [B, T', C]
            if self.use_spec_disentangle and self.proj_inv is not None:
                z_tokens = self.norm_inv(self.proj_inv(x_tokens))  # [B, T', D_inv]
            else:
                z_tokens = self.norm(self.proj(x_tokens))          # [B, T', D]
            return z_tokens, mask

        sum_x = x.sum(dim=-1)  # [B, C]
        denom = mask.sum(dim=-1).clamp(min=1e-6).unsqueeze(-1)
        pooled = sum_x / denom  # [B, C]
        if ratio_summary is not None and ratio_summary.shape == pooled.shape:
            _blend = float(max(0.0, min(1.0, self.spec_freq_ratio_blend)))
            pooled = (1.0 - _blend) * pooled + _blend * ratio_summary

        if self.use_spec_disentangle and self.proj_inv is not None:
            # ATT173: 双投影头路径
            # z_inv_raw [B, D_inv]: 域不变特征（主路径，与原 proj 接口兼容）
            # z_spe_raw [B, D_spe]: 域特异特征（仅辅助训练，存入 _last_* 供 training loop 读取）
            z_inv_raw = self.proj_inv(pooled)          # [B, D_inv]
            z_spe_raw = self.proj_spe(pooled)          # [B, D_spe]
            # 存储未归一化的原始特征（保留梯度，training loop 计算 L_orth / L_domain）
            self._last_z_inv_raw = z_inv_raw
            self._last_z_spe_raw = z_spe_raw
            z_spec_pre_norm = self.norm_inv(z_inv_raw)
            self._last_z_spec_pre_norm = z_spec_pre_norm
            z_spec = F.normalize(z_spec_pre_norm, dim=-1)   # [B, D_inv]
        else:
            self._last_z_inv_raw = None
            self._last_z_spe_raw = None
            z_spec_pre_norm = self.norm(self.proj(pooled))
            self._last_z_spec_pre_norm = z_spec_pre_norm
            z_spec = F.normalize(z_spec_pre_norm, dim=-1)
            self._last_z_spe_dual = None
        try:
            z_spec._pre_l2_feature = z_spec_pre_norm
        except Exception:
            pass
        if self.training and self.train_log_specenc_norm_stats:
            self._specenc_forward_step += 1
            if self._specenc_forward_step == 1:
                print(
                    f"[ATT377][SpecEnc] instance_norm={int(self.use_spec_instance_norm_before_head)}, "
                    f"freq_ratio={int(self.use_freq_ratio_features)}, "
                    f"fs_aware={int(self.use_spec_fs_aware_norm)}"
                )
            if self._specenc_forward_step == 1 or (self._specenc_forward_step % self.train_log_specenc_norm_step_stride == 0):
                try:
                    _pre = z_spec_pre_norm.detach().float()
                    _post = z_spec.detach().float()
                    _pre_std = float(_pre.std(unbiased=False).cpu().item())
                    _post_std = float(_post.std(unbiased=False).cpu().item())
                    _pre_norm = float(_pre.norm(dim=-1).mean().cpu().item())
                    _post_norm = float(_post.norm(dim=-1).mean().cpu().item())
                    _msg = (
                        f"[SpecEnc] step={int(self._specenc_forward_step)} "
                        f"pre_norm_std={_pre_std:.6f}, post_norm_std={_post_std:.6f}, "
                        f"pre_norm_mean={_pre_norm:.6f}, post_norm_mean={_post_norm:.6f}"
                    )
                    if ratio_summary is not None:
                        _ratio_mean = float(ratio_summary.detach().float().mean().cpu().item())
                        _ratio_std = float(ratio_summary.detach().float().std(unbiased=False).cpu().item())
                        _msg += f", ratio_mean={_ratio_mean:.6f}, ratio_std={_ratio_std:.6f}"
                    if self.use_spec_fs_aware_norm and torch.is_tensor(domain_ids) and self.log_spec_fs_aware_norm:
                        _did = domain_ids.detach().view(-1).long().cpu().tolist()
                        _sc = []
                        _ref = float(max(1.0, self.spec_fs_aware_ref_fs))
                        _pwr = float(self.spec_fs_aware_power)
                        for _d in _did:
                            _fs = float(self.domain_fs_map.get(int(_d), _ref))
                            _sc.append((_ref / max(1.0, _fs)) ** _pwr)
                        if len(_sc) > 0:
                            _msg += (
                                f", fs_scale_mean={float(np.mean(_sc)):.6f}, "
                                f"fs_scale_min={float(np.min(_sc)):.6f}, fs_scale_max={float(np.max(_sc)):.6f}"
                            )
                    logger.info(_msg)
                    print(_msg)
                except Exception:
                    pass

        logits = None
        if return_logits and self.use_classification_head:
            if self.spec_prototypes is not None:
                logits = self._compute_proto_logits(z_spec_pre_norm)
            elif self.classifier is not None:
                _cls_input = self._build_cls_input(z_spec_pre_norm, wst_aux, env_aux, ratio_aux, scd_aux)
                logits = self.classifier(_cls_input)  # [B, C]

        # ATT429-P2: Hierarchical head logits (MLP, direct on z_spec_pre_norm)
        nf_logits = None
        fault_logits = None
        if self.nf_head is not None:
            nf_logits = self.nf_head(z_spec_pre_norm)  # [B, 2]
        if self.fault_head is not None:
            fault_logits = self.fault_head(z_spec_pre_norm)  # [B, n_fault]

        # Store hierarchical logits for training loop access
        self._last_nf_logits = nf_logits
        self._last_fault_logits = fault_logits
        # ATT465-MLSA: store mid-level representation for alignment loss
        self._last_z_mid = _z_mid

        # Defensive compatibility when accidentally called with MoE-style flag.
        if return_moe_stats:
            moe_stats = {}
            if logits is not None:
                return z_spec, moe_stats, logits
            return z_spec, moe_stats

        if logits is not None:
            return z_spec, logits
        return z_spec

    def compute_spec_head_logits(self, z_spec, wst_aux=None, env_aux=None, ratio_aux=None, scd_aux=None):
        """
        Compute SpecHead logits. Supports MLP (legacy) and Proto (ATT453) modes.
        """
        z_pre = getattr(z_spec, '_pre_l2_feature', z_spec)

        if self.spec_prototypes is not None:
            return self._compute_proto_logits(z_pre)

        if self.classifier is None:
            return None
        _cls_input = self._build_cls_input(z_pre, wst_aux, env_aux, ratio_aux, scd_aux)
        return self.classifier(_cls_input)

    def _compute_proto_logits(self, z_pre):
        """ATT453: Cosine similarity against learnable ETF prototypes."""
        z_normed = F.normalize(z_pre, dim=-1)
        proto_normed = F.normalize(self.spec_prototypes, dim=-1)
        cos_sim = z_normed @ proto_normed.T
        logits = cos_sim / max(self._spec_proto_temperature, 1e-6)
        return logits

    def _build_cls_input(self, z_pre, wst_aux=None, env_aux=None, ratio_aux=None, scd_aux=None):
        """Build the concatenated input for the SpecHead classifier.

        ATT450-S1: L2-normalize vector bypass branches (wst, env, scd), then multiply
        by learnable scale parameters.  This resolves the 4-50x magnitude gap
        that caused bypass features to be drowned out by z_spec_pre_norm.
        """
        parts = [z_pre]
        B = z_pre.size(0)
        dev = z_pre.device
        dt = z_pre.dtype
        _gate_bank = get_gate_bank(self)
        if self._wst_bypass_dim > 0:
            if wst_aux is not None:
                _wst = F.normalize(wst_aux, dim=-1) * self._wst_scale
                _wst = apply_residual_gate(
                    _gate_bank, "physics_wst", _wst, torch.zeros_like(_wst),
                )
                parts.append(_wst)
            else:
                parts.append(torch.zeros(B, self._wst_bypass_dim, device=dev, dtype=dt))
        if self._env_bypass_dim > 0:
            if env_aux is not None:
                _env = F.normalize(env_aux, dim=-1) * self._env_scale
                _env = apply_residual_gate(
                    _gate_bank, "physics_env", _env, torch.zeros_like(_env),
                )
                parts.append(_env)
            else:
                parts.append(torch.zeros(B, self._env_bypass_dim, device=dev, dtype=dt))
        if self._env_ratio_dim > 0:
            if ratio_aux is not None:
                _ratio = ratio_aux * self._ratio_scale
                _ratio = apply_residual_gate(
                    _gate_bank, "physics_env", _ratio, torch.zeros_like(_ratio),
                )
                parts.append(_ratio)
            else:
                parts.append(torch.zeros(B, self._env_ratio_dim, device=dev, dtype=dt))
        if self._scd_bypass_dim > 0:
            if scd_aux is not None:
                _scd = F.normalize(scd_aux, dim=-1) * self._scd_scale
                parts.append(_scd)
            else:
                parts.append(torch.zeros(B, self._scd_bypass_dim, device=dev, dtype=dt))

        if self.training and self._bypass_scale_log_count < 5:
            self._bypass_scale_log_count += 1
            _msg_parts = [f"[ATT450-ScaleNorm] step={self._bypass_scale_log_count}"]
            _msg_parts.append(f"z_pre_norm={float(z_pre.detach().norm(dim=-1).mean()):.3f}")
            if self._wst_scale is not None:
                _msg_parts.append(f"wst_scale={float(self._wst_scale.detach()):.3f}")
                if wst_aux is not None:
                    _raw_norm = float(wst_aux.detach().norm(dim=-1).mean())
                    _scaled_norm = float((F.normalize(wst_aux.detach(), dim=-1) * self._wst_scale.detach()).norm(dim=-1).mean())
                    _msg_parts.append(f"wst_raw_norm={_raw_norm:.3f}")
                    _msg_parts.append(f"wst_scaled_norm={_scaled_norm:.3f}")
            if self._env_scale is not None:
                _msg_parts.append(f"env_scale={float(self._env_scale.detach()):.3f}")
                if env_aux is not None:
                    _raw_norm = float(env_aux.detach().norm(dim=-1).mean())
                    _scaled_norm = float((F.normalize(env_aux.detach(), dim=-1) * self._env_scale.detach()).norm(dim=-1).mean())
                    _msg_parts.append(f"env_raw_norm={_raw_norm:.3f}")
                    _msg_parts.append(f"env_scaled_norm={_scaled_norm:.3f}")
            if self._ratio_scale is not None:
                _msg_parts.append(f"ratio_scale={float(self._ratio_scale.detach()):.3f}")
            if self._scd_scale is not None:
                _msg_parts.append(f"scd_scale={float(self._scd_scale.detach()):.3f}")
                if scd_aux is not None:
                    _raw_norm = float(scd_aux.detach().norm(dim=-1).mean())
                    _scaled_norm = float((F.normalize(scd_aux.detach(), dim=-1) * self._scd_scale.detach()).norm(dim=-1).mean())
                    _msg_parts.append(f"scd_raw_norm={_raw_norm:.3f}")
                    _msg_parts.append(f"scd_scaled_norm={_scaled_norm:.3f}")
            print(", ".join(_msg_parts))

        return torch.cat(parts, dim=-1)

    def compute_hierarchical_logits(self, z_spec):
        """
        ATT439: Hierarchical SpecHead prediction.
        Returns 4-class logits synthesized from nf_head and fault_head.
        P(Normal) = P(NF=0); P(IR/OR/Ball) = P(NF=1) * P(fault_class|fault)
        """
        if self.nf_head is None or self.fault_head is None:
            return None
        z_pre = getattr(z_spec, '_pre_l2_feature', z_spec)
        nf_logits = self.nf_head(z_pre)
        fault_logits = self.fault_head(z_pre)
        nf_probs = torch.softmax(nf_logits, dim=-1)
        fault_probs = torch.softmax(fault_logits, dim=-1)
        p_normal = nf_probs[:, 0:1]
        p_fault = nf_probs[:, 1:2]
        n_fault_cls = fault_probs.size(-1)
        p_subtypes = p_fault * fault_probs
        hier_probs = torch.cat([p_normal, p_subtypes], dim=-1)
        hier_logits = torch.log(hier_probs.clamp(min=1e-8))
        return hier_logits

    def reset_bn_stats_for_domain(self, dataloader, device, max_batches=64):
        """
        ATT429-P1: Full BN statistics reset for target domain evaluation.
        ATT430-S0 FIX: handle batch["domain"] being a list of strings (not tensor).
        """
        self.train()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.reset_running_stats()
                m.momentum = None

        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                if n_batches >= max_batches:
                    break
                spec = batch["spec"].to(device)
                spec_mask = batch["spec_mask"].to(device)
                domain_ids = None
                if "domain_id" in batch and torch.is_tensor(batch["domain_id"]):
                    domain_ids = batch["domain_id"].to(device)
                self(spec, spec_mask, domain_ids=domain_ids)
                n_batches += 1

        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = 0.1
        print(f"[ATT430-P1] BN stats reset: {n_batches} batches accumulated")

    def get_hierarchical_prediction(self, z_spec_pre_norm, num_classes=4):
        """
        ATT430-S0: Hierarchical inference with MLP heads.
        NF decision first, then fault subtype. Returns composed logits.
        """
        if self.nf_head is None or self.fault_head is None:
            return None

        nf_logits = self.nf_head(z_spec_pre_norm)  # [B, 2]
        nf_probs = F.softmax(nf_logits, dim=-1)    # [B, 2] (Normal, Fault)

        fault_logits = self.fault_head(z_spec_pre_norm)  # [B, n_fault]
        fault_probs = F.softmax(fault_logits, dim=-1)    # [B, n_fault]

        p_normal = nf_probs[:, 0:1]
        p_fault = nf_probs[:, 1:2]
        composed_probs = torch.cat([p_normal, p_fault * fault_probs], dim=1)

        if composed_probs.size(1) > num_classes:
            composed_probs = composed_probs[:, :num_classes]
        elif composed_probs.size(1) < num_classes:
            pad = torch.zeros(composed_probs.size(0), num_classes - composed_probs.size(1),
                              device=composed_probs.device, dtype=composed_probs.dtype)
            composed_probs = torch.cat([composed_probs, pad], dim=1)

        composed_logits = torch.log(composed_probs.clamp(min=1e-8))
        return composed_logits

    def compute_disentangle_losses(self, domain_ids):
        """ATT431: stub — dual disentangle removed (falsified F_DD)."""
        return {}

    def compute_variance_loss(self, target_std=1.0):
        """ATT431: stub — spec variance removed (falsified F_SV)."""
        return None

    def project_for_proto(self, z_spec: torch.Tensor, detach_input: bool = False) -> torch.Tensor:
        """Project z_spec into proto-aligned space with optional detach."""
        if z_spec is None:
            return z_spec
        x = z_spec.detach() if detach_input else z_spec
        if self.align_proj is None:
            return F.normalize(x, dim=-1)
        y = self.align_proj(x)
        if y.shape == x.shape:
            y = y + x
        return F.normalize(y, dim=-1)

    def project_for_supcon(self, z_spec: torch.Tensor) -> torch.Tensor:
        """
        ATT240(S1): remove SupCon projection path and use direct normalized z_spec.
        Keep dedicated projection parameters defined for checkpoint compatibility,
        but bypass them to avoid additional gradient attenuation.
        """
        if z_spec is None:
            return z_spec
        return F.normalize(z_spec, dim=-1)
