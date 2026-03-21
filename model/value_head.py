"""Value head: predicts position evaluation as scalar in [-1, 1]."""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Elo-independent scalar evaluation head.

    Architecture: Conv 1x1 → 32, BN, ReLU → FC 256 → FC 1 → tanh
    Output range: [-1, 1] where -1 = losing, 0 = equal, 1 = winning
    (from the perspective of the side to move)
    """

    def __init__(self, in_channels: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(32 * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute position value.

        Args:
            features: (batch, 128, 8, 8) from backbone

        Returns:
            (batch, 1) value in [-1, 1]
        """
        out = self.relu(self.bn(self.conv(features)))
        out = out.reshape(out.size(0), -1)  # (batch, 32*8*8)
        out = self.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out
