"""
ATT392-S1: Learnable Prototypes with Orthogonality + Anchor Constraints.

Replaces fixed text/ETF prototypes with gradient-trainable parameters.
Initialized from text-encoder embeddings, then optimized with two regularizers:
  1. separation_loss: pushes all pairwise cos(p_i, p_j) below a configurable ceiling
  2. anchor_loss:     keeps each prototype near its text-semantic origin

References:
  - Neural Collapse (Papyan et al. 2020): optimal classifiers converge to simplex ETF
  - Proxy-NCA (Movshovitz-Attias et al. 2017): learnable proxies for metric learning
  - ProtoNet (Snell et al. 2017): prototype-based classification
  - CoOp (Zhou et al. 2022): context optimization for vision-language models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class LearnablePrototypes(nn.Module):
    def __init__(self, text_proto_init: torch.Tensor, min_cos: float = -0.3,
                 init_from_etf: bool = False):
        """
        Args:
            text_proto_init: [C, D] L2-normalized text embeddings (used as anchor even when ETF init)
            min_cos: target ceiling for pairwise cosine similarity (separation_loss
                     penalizes cos > min_cos). For 4 classes, simplex ETF = -1/3 ≈ -0.333;
                     -0.3 provides a slightly relaxed target.
            init_from_etf: if True, initialize protos from simplex ETF instead of text embeddings.
                           text_proto_init is still stored as anchor for anchor_loss.
        """
        super().__init__()
        C, D = text_proto_init.shape
        if init_from_etf:
            etf_vectors = self.generate_etf(C, D, device=text_proto_init.device)
            self.protos = nn.Parameter(etf_vectors)
        else:
            self.protos = nn.Parameter(text_proto_init.clone())
        self.max_source_residual_norm = float(os.environ.get(
            "NEW_IDEA_PROTO_MAX_SOURCE_RESIDUAL_NORM", "0.0"
        ))
        if self.max_source_residual_norm < 0.0:
            raise ValueError("prototype source residual norm must be non-negative")
        self.use_self_attention = bool(int(os.environ.get("NEW_IDEA_PROTO_SELF_ATTN", "0")))
        if self.use_self_attention:
            self.prototype_attention = nn.MultiheadAttention(
                D, num_heads=4, batch_first=True
            )
            self.prototype_attention_scale = nn.Parameter(torch.zeros(()))
        else:
            self.prototype_attention = None
            self.register_parameter("prototype_attention_scale", None)
        self.register_buffer("text_anchor", text_proto_init.clone())
        self.register_buffer("source_anchor", torch.zeros_like(text_proto_init))
        self.register_buffer("source_anchor_active", torch.tensor(False))
        self.register_buffer("relation_target", torch.empty(0))
        self.min_cos = min_cos
        self._n_classes = C
        self._dim = D

    @staticmethod
    def generate_etf(num_classes: int, embed_dim: int,
                     device: torch.device = None) -> torch.Tensor:
        """Generate simplex ETF vectors [C, D] with all pairwise cos = -1/(C-1).

        Construction:
          1. Simplex ETF in C-space: M = sqrt(C/(C-1)) * (I_C - 1/C * 11^T)
          2. Random orthonormal basis Q [D, C] via QR decomposition
          3. Project: result = M @ Q^T -> [C, D], preserving inner products
        """
        C, D = num_classes, embed_dim
        M = torch.eye(C) - (1.0 / C) * torch.ones(C, C)
        M *= (C / (C - 1.0)) ** 0.5

        rand_mat = torch.randn(D, C)
        Q, _ = torch.linalg.qr(rand_mat)  # [D, C] orthonormal

        result = M @ Q.t()  # [C, D]
        result = F.normalize(result, dim=-1)
        if device is not None:
            result = result.to(device)
        return result

    def forward(self) -> torch.Tensor:
        """Returns L2-normalized prototypes [C, D]."""
        p = F.normalize(self.protos, dim=-1)
        if self.prototype_attention is not None:
            attended, _ = self.prototype_attention(
                p.unsqueeze(0), p.unsqueeze(0), p.unsqueeze(0), need_weights=False
            )
            p = F.normalize(p + torch.tanh(self.prototype_attention_scale) * attended.squeeze(0), dim=-1)
        if self.max_source_residual_norm > 0.0 and bool(self.source_anchor_active):
            anchor = F.normalize(self.source_anchor, dim=-1)
            delta = p - anchor
            # The residual length is strictly below the configured bound,
            # including after the optional prototype self-attention.
            delta = delta / (1.0 + delta.norm(dim=-1, keepdim=True))
            p = F.normalize(anchor + self.max_source_residual_norm * delta, dim=-1)
        return p

    @torch.no_grad()
    def set_from_signal_centroids(self, centroids: torch.Tensor, semantic_blend: float = 0.0) -> None:
        """Update from source centroids with an optional MAP-style semantic prior blend."""
        if centroids.shape != self.protos.shape:
            raise ValueError("signal centroids must match prototype shape")
        c = F.normalize(centroids.to(self.protos), dim=-1)
        b = float(max(0.0, min(1.0, semantic_blend)))
        if b > 0.0:
            prior = self.forward().detach()
            c = F.normalize((1.0 - b) * c + b * prior, dim=-1)
        preserve = bool(int(os.environ.get("NEW_IDEA_PRESERVE_PROTOS_ON_CENTROID_UPDATE", "0")))
        if not preserve:
            self.protos.copy_(c)
        else:
            print("[VGSP] source anchor updated; learned prototype parameters preserved")
        self.source_anchor.copy_(c)
        self.source_anchor_active.fill_(True)

    def set_relation_target(self, target: torch.Tensor) -> None:
        """Set a fixed semantic relation matrix without target-domain data."""
        if target.shape != (self._n_classes, self._n_classes):
            raise ValueError("relation target has an invalid shape")
        self.relation_target = target.detach().to(self.protos)

    def semantic_relation_loss(self) -> torch.Tensor:
        """Match learned prototype geometry to the frozen language relation."""
        if self.relation_target.numel() == 0:
            return self.protos.sum() * 0.0
        p = self.forward()
        pred = p @ p.t()
        mask = ~torch.eye(self._n_classes, dtype=torch.bool, device=p.device)
        return F.smooth_l1_loss(pred[mask], self.relation_target[mask])

    def separation_loss(self) -> torch.Tensor:
        """Hinge-style loss pushing all off-diagonal cos below min_cos."""
        p = self.forward()  # [C, D]
        cos = p @ p.t()     # [C, C]
        mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
        excess = F.relu(cos[mask] - self.min_cos)
        return excess.mean()

    def anchor_loss(self) -> torch.Tensor:
        """Cosine distance to original text-semantic directions."""
        p = self.forward()
        a = F.normalize(self.text_anchor, dim=-1)
        return (1.0 - (p * a).sum(dim=-1)).mean()

    @torch.no_grad()
    def proto_cosine_matrix(self) -> torch.Tensor:
        """Returns [C, C] cosine similarity matrix for diagnostics."""
        p = self.forward()
        return p @ p.t()

    @torch.no_grad()
    def anchor_cosine(self) -> torch.Tensor:
        """Returns [C] per-class cosine similarity to text anchor."""
        p = self.forward()
        a = F.normalize(self.text_anchor, dim=-1)
        return (p * a).sum(dim=-1)

    @torch.no_grad()
    def drift_from_anchor(self) -> float:
        """Mean angular drift from text anchor (in cosine distance)."""
        return float((1.0 - self.anchor_cosine()).mean().item())

    def as_dict(self) -> dict:
        """Returns {class_idx: tensor[D]} for compatibility with existing proto dict format."""
        p = self.forward().detach()
        return {i: p[i] for i in range(self._n_classes)}
