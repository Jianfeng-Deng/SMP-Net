"""
ATT449-S3: TinyEnvEncoder — lightweight encoder for envelope spectrum bypass injection.

Processes envelope STFT features [B, 1, H, W] through a small ConvNet,
producing a compact z_env vector that is injected into SpecHead only,
bypassing the main SpecEncoder→Fusion path.
"""

import torch
import torch.nn as nn


class TinyEnvEncoder(nn.Module):

    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] envelope STFT features
        Returns:
            z_env: [B, out_dim]
        """
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
