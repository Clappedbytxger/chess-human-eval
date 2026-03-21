"""FiLM (Feature-wise Linear Modulation) for Elo conditioning."""

import torch
import torch.nn as nn


class EloEmbedding(nn.Module):
    """Embeds Elo ratings into a continuous vector space.

    Elo is discretized into brackets (800-899, 900-999, ..., 2700-2799).
    Each bracket gets a learned embedding. For Elo values between brackets,
    we interpolate between the two nearest embeddings.
    """

    def __init__(self, num_brackets: int = 20, embed_dim: int = 32,
                 elo_min: int = 800, elo_max: int = 2800):
        super().__init__()
        self.num_brackets = num_brackets
        self.embed_dim = embed_dim
        self.elo_min = elo_min
        self.elo_max = elo_max
        self.bracket_size = (elo_max - elo_min) / num_brackets

        self.embeddings = nn.Embedding(num_brackets, embed_dim)

    def forward(self, elo: torch.Tensor) -> torch.Tensor:
        """Convert Elo ratings to embedding vectors with interpolation.

        Args:
            elo: Tensor of shape (batch,) with Elo ratings (800-2800)

        Returns:
            Tensor of shape (batch, embed_dim)
        """
        # Clamp and normalize to bracket indices
        elo_clamped = elo.float().clamp(self.elo_min, self.elo_max - 1)
        bracket_float = (elo_clamped - self.elo_min) / self.bracket_size

        # Get lower and upper bracket indices
        lower_idx = bracket_float.long().clamp(0, self.num_brackets - 1)
        upper_idx = (lower_idx + 1).clamp(0, self.num_brackets - 1)

        # Interpolation weight
        alpha = (bracket_float - lower_idx.float()).unsqueeze(-1)

        lower_embed = self.embeddings(lower_idx)
        upper_embed = self.embeddings(upper_idx)

        return lower_embed * (1 - alpha) + upper_embed * alpha


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Applies learned, Elo-dependent scale (gamma) and shift (beta)
    to feature maps: output = gamma * features + beta
    """

    def __init__(self, embed_dim: int = 32, num_channels: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels * 2),  # gamma + beta
        )

    def forward(self, features: torch.Tensor, elo_embed: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            features: (batch, channels, h, w) - feature maps from conv
            elo_embed: (batch, embed_dim) - Elo embedding vector

        Returns:
            Modulated features with same shape as input
        """
        params = self.mlp(elo_embed)  # (batch, channels*2)
        gamma, beta = params.chunk(2, dim=-1)  # each (batch, channels)

        # Reshape for broadcasting with spatial dims
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (batch, channels, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * features + beta
