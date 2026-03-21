"""ResNet backbone for chess position encoding."""

import torch
import torch.nn as nn

from .film import FiLMLayer


class ResidualBlock(nn.Module):
    """Single residual block with FiLM conditioning.

    Conv 3x3 → BN → ReLU → Conv 3x3 → BN → FiLM → Skip + ReLU
    """

    def __init__(self, num_channels: int = 128, embed_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.film = FiLMLayer(embed_dim, num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, elo_embed: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.film(out, elo_embed)
        out = self.relu(out + residual)
        return out


class ResNetBackbone(nn.Module):
    """ResNet tower: initial conv + N residual blocks.

    Input: (batch, 18, 8, 8) - encoded board
    Output: (batch, 128, 8, 8) - feature maps
    """

    def __init__(self, in_channels: int = 18, num_channels: int = 128,
                 num_blocks: int = 10, embed_dim: int = 32):
        super().__init__()
        # Initial convolution to project input planes to feature space
        self.initial_conv = nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList([
            ResidualBlock(num_channels, embed_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor, elo_embed: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.blocks:
            out = block(out, elo_embed)
        return out
