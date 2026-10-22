import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    """
    Simple MLP discriminator for DANN.
    Input: feature embedding [B, D]
    Output: domain logits [B, num_domains]
    """
    def __init__(self, in_dim: int, num_domains: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        h = int(hidden_dim)
        d = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), h),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(h, int(num_domains)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

