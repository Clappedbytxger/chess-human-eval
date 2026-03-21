"""Complete chess neural network: ResNet + FiLM + Policy/Value heads."""

import torch
import torch.nn as nn

from .film import EloEmbedding
from .resnet import ResNetBackbone
from .policy_head import PolicyHead
from .value_head import ValueHead


class ChessNet(nn.Module):
    """Elo-conditioned chess move prediction network.

    Takes a board position (18x8x8) and Elo rating as input.
    Outputs:
        - Policy: probability distribution over legal moves (4672)
        - Value: position evaluation scalar [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 18,
        num_channels: int = 128,
        num_blocks: int = 10,
        embed_dim: int = 32,
        num_elo_brackets: int = 20,
        elo_min: int = 800,
        elo_max: int = 2800,
        policy_planes: int = 73,
    ):
        super().__init__()
        self.elo_embedding = EloEmbedding(
            num_brackets=num_elo_brackets,
            embed_dim=embed_dim,
            elo_min=elo_min,
            elo_max=elo_max,
        )
        self.backbone = ResNetBackbone(
            in_channels=in_channels,
            num_channels=num_channels,
            num_blocks=num_blocks,
            embed_dim=embed_dim,
        )
        self.policy_head = PolicyHead(
            in_channels=num_channels,
            policy_planes=policy_planes,
        )
        self.value_head = ValueHead(
            in_channels=num_channels,
        )

    def forward(
        self,
        board: torch.Tensor,
        elo: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            board: (batch, 18, 8, 8) encoded board position
            elo: (batch,) Elo ratings
            legal_mask: (batch, 4672) optional binary mask for legal moves

        Returns:
            policy: (batch, 4672) log-probabilities over moves
            value: (batch, 1) position evaluation
        """
        elo_embed = self.elo_embedding(elo)  # (batch, embed_dim)
        features = self.backbone(board, elo_embed)  # (batch, 128, 8, 8)

        policy = self.policy_head(features, legal_mask)
        value = self.value_head(features)

        return policy, value

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
