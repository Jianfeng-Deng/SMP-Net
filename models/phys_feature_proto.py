"""
ATT432-δ: Physical Feature Prototype Module with Class-Conditional Projection.

Replaces text-based prototype semantics with direct numerical diagnostic features:
  - Preserves discriminative power lost by text encoding ("RMS=0.35" vs "RMS=0.42")
  - Class-conditional projection heads learn fault-type-specific feature importance
  - EMA centroids serve as zero-shot-transferable prototypes

Architecture:
  Input: x_phys [B, F] raw diagnostic features (22-dim)
  -> BatchNorm1d (domain-robust normalization)
  -> Shared Encoder (F -> hidden -> D)
  -> Per-class Projection Head (D -> D, selected by label during training)
  -> L2-normalize -> z_phys [B, D]

  Class centroids are maintained via EMA for prototype-based inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


PHYS_FEATURE_KEYS = [
    "rms", "std", "min", "max", "median", "peak_to_peak", "crest_factor",
    "kurtosis", "skewness", "dominant_freq", "spectral_entropy",
    "spectral_centroid", "spectral_bandwidth", "wavelet_entropy",
    "band_low_ratio", "band_mid_ratio", "band_high_ratio",
    "env_kurtosis", "env_dominant_freq", "env_sideband_ratio",
    "env_harmonic_strength", "env_modulation_index",
]
PHYS_FEATURE_DIM = len(PHYS_FEATURE_KEYS)


def features_dict_to_tensor(features_dict: dict) -> torch.Tensor:
    """Convert extracted features dict to a [F] float tensor."""
    return torch.tensor(
        [float(features_dict.get(k, 0.0)) for k in PHYS_FEATURE_KEYS],
        dtype=torch.float32,
    )


class PhysFeatureProtoModule(nn.Module):

    def __init__(
        self,
        feat_dim: int = PHYS_FEATURE_DIM,
        embed_dim: int = 512,
        num_classes: int = 4,
        hidden_dim: int = 128,
        ema_momentum: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.feat_norm = nn.BatchNorm1d(feat_dim, affine=True)

        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for _ in range(num_classes)
        ])

        self.register_buffer("class_centroids", torch.zeros(num_classes, embed_dim))
        self.register_buffer("class_counts", torch.zeros(num_classes))
        self._ema_momentum = ema_momentum

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_phys: torch.Tensor, class_ids: torch.Tensor = None):
        """
        Args:
            x_phys: [B, F] raw physical features
            class_ids: [B] int class labels (required during training)
        Returns:
            z_phys: [B, D] L2-normalized physical feature embeddings
        """
        z_shared = self.encoder(self.feat_norm(x_phys))

        if class_ids is not None and self.training:
            z_out = torch.zeros_like(z_shared)
            for c in range(self.num_classes):
                mask = class_ids == c
                if mask.any():
                    z_out[mask] = self.class_heads[c](z_shared[mask])
            z_out = F.normalize(z_out, dim=-1)
            self._update_centroids(z_out, class_ids)
            return z_out

        z_all = torch.stack(
            [F.normalize(head(z_shared), dim=-1) for head in self.class_heads],
            dim=1,
        )  # [B, C, D]
        centroids = F.normalize(self.class_centroids, dim=-1)
        sims = torch.einsum("bcd,cd->bc", z_all, centroids)
        best = sims.argmax(dim=-1)
        return z_all[torch.arange(z_all.size(0), device=z_all.device), best]

    @torch.no_grad()
    def _update_centroids(self, z_out: torch.Tensor, class_ids: torch.Tensor):
        mom = self._ema_momentum
        for c in range(self.num_classes):
            mask = class_ids == c
            if not mask.any():
                continue
            batch_mean = F.normalize(z_out[mask].mean(dim=0), dim=-1)
            if self.class_counts[c] == 0:
                self.class_centroids[c] = batch_mean
            else:
                self.class_centroids[c] = F.normalize(
                    (1 - mom) * self.class_centroids[c] + mom * batch_mean,
                    dim=-1,
                )
            self.class_counts[c] += mask.sum()

    def get_prototypes_dict(self) -> dict:
        centroids = F.normalize(self.class_centroids, dim=-1)
        return {i: centroids[i] for i in range(self.num_classes)}

    def get_prototypes_tensor(self) -> torch.Tensor:
        return F.normalize(self.class_centroids, dim=-1)

    @torch.no_grad()
    def centroid_cosine_matrix(self) -> torch.Tensor:
        c = F.normalize(self.class_centroids, dim=-1)
        return c @ c.t()
