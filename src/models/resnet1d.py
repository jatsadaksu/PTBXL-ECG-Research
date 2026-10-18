"""G2-locked three-stage residual 1D CNN for five-label PTB-XL classification.

Input: (batch, 12, 1000); output: five *raw logits* (no sigmoid).
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock1D(nn.Module):
    """Two k=3 convolutions with an optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet1D(nn.Module):
    """Stem + 2 basic blocks per stage (32, 64, 128 channels)."""

    def __init__(self, in_channels: int = 12, num_classes: int = 5) -> None:
        super().__init__()
        if in_channels != 12 or num_classes != 5:
            raise ValueError("G2 lock requires exactly 12 input leads and 5 output labels")
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=1, padding=3, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2),
        )
        self.stage1 = nn.Sequential(BasicBlock1D(32, 32), BasicBlock1D(32, 32))
        self.stage2 = nn.Sequential(BasicBlock1D(32, 64, stride=2), BasicBlock1D(64, 64))
        self.stage3 = nn.Sequential(BasicBlock1D(64, 128, stride=2), BasicBlock1D(128, 128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(128, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (12, 1000):
            raise ValueError(f"Expected input shape (batch, 12, 1000), got {tuple(x.shape)}")
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(self.dropout(self.pool(x).flatten(1)))
