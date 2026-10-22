"""
ATT391-S1: Proto Space Adapter — lightweight linear mapping from ETF-aligned
feature space to text prototype space.

Training:  z_fusion aligned to ETF prototypes via PACL (cos(i,j)=-1/(C-1)).
Adapter:   learns to rotate ETF-space features into text-prototype space.
Eval:      adapter(z_fusion) cosine-matched with text prototypes for zero-shot.

Identity-initialized so the adapter starts as a no-op and gradually learns
the rotation needed for semantic transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoSpaceAdapter(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z), dim=-1)

    @torch.no_grad()
    def weight_deviation_from_identity(self) -> float:
        """Frobenius distance from identity — monitors how much the adapter has learned."""
        eye = torch.eye(self.proj.weight.size(0), device=self.proj.weight.device)
        return float((self.proj.weight - eye).norm().item())
