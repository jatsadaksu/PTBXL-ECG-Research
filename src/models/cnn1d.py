"""G2-locked lightweight 1D CNN for five-label PTB-XL classification.

Input: (batch, 12, 1000); output: five *raw logits* (no sigmoid).
"""

from __future__ import annotations

import torch
from torch import nn


class LightweightCNN1D(nn.Module):
    """Three Conv-BN-ReLU-MaxPool stages followed by global average pooling."""

    def __init__(self, in_channels: int = 12, num_classes: int = 5) -> None:
        super().__init__()
        if in_channels != 12 or num_classes != 5:
            raise ValueError("G2 lock requires exactly 12 input leads and 5 output labels")
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (12, 1000):
            raise ValueError(f"Expected input shape (batch, 12, 1000), got {tuple(x.shape)}")
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


# Convenience alias; both names refer to the same locked architecture.
CNN1D = LightweightCNN1D
