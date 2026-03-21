"""Loss functions for combined policy + value training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """Combined policy (cross-entropy) + value (MSE) loss.

    total_loss = policy_loss + value_weight * value_loss
    """

    def __init__(self, value_weight: float = 0.5):
        super().__init__()
        self.value_weight = value_weight

    def forward(
        self,
        policy_logprobs: torch.Tensor,
        policy_target: torch.Tensor,
        value_pred: torch.Tensor,
        value_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Args:
            policy_logprobs: (batch, 4672) log-probabilities from policy head
            policy_target: (batch,) target move indices
            value_pred: (batch, 1) predicted values
            value_target: (batch, 1) target values (tanh-scaled centipawns)

        Returns:
            (total_loss, policy_loss, value_loss)
        """
        # Policy: NLL loss (input is already log-softmax)
        policy_loss = F.nll_loss(policy_logprobs, policy_target)

        # Value: MSE loss
        value_loss = F.mse_loss(value_pred, value_target)

        total = policy_loss + self.value_weight * value_loss

        return total, policy_loss, value_loss


def centipawn_to_value(cp: float | torch.Tensor) -> float | torch.Tensor:
    """Convert centipawn evaluation to [-1, 1] value target.

    Uses tanh(cp / 1000) which maps:
        0 cp → 0.0
        +100 cp → ~0.1
        +300 cp → ~0.29
        +1000 cp → ~0.76
        +3000 cp → ~0.995
    """
    if isinstance(cp, torch.Tensor):
        return torch.tanh(cp / 1000.0)
    return float(np.tanh(cp / 1000.0))
