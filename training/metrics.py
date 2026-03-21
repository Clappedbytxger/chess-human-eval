"""Training and evaluation metrics."""

import torch
import numpy as np
from collections import defaultdict


class MetricsTracker:
    """Track training metrics with running averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, metrics: dict[str, float], n: int = 1):
        """Add a batch of metrics."""
        for key, value in metrics.items():
            self._sums[key] += value * n
            self._counts[key] += n

    def average(self) -> dict[str, float]:
        """Get running averages."""
        return {
            key: self._sums[key] / self._counts[key]
            for key in self._sums
        }


def compute_accuracy(
    policy_logprobs: torch.Tensor,
    targets: torch.Tensor,
    top_k: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute top-k accuracy for policy predictions.

    Args:
        policy_logprobs: (batch, 4672) log-probabilities
        targets: (batch,) target move indices
        top_k: Tuple of k values to compute

    Returns:
        Dict with "top_1_acc", "top_5_acc", etc.
    """
    results = {}
    batch_size = targets.size(0)

    for k in top_k:
        _, top_indices = policy_logprobs.topk(k, dim=-1)
        correct = top_indices.eq(targets.unsqueeze(-1)).any(dim=-1)
        results[f"top_{k}_acc"] = correct.float().sum().item() / batch_size

    return results


def compute_per_elo_accuracy(
    policy_logprobs: torch.Tensor,
    targets: torch.Tensor,
    elos: torch.Tensor,
    bracket_size: int = 100,
    elo_min: int = 800,
    elo_max: int = 2800,
) -> dict[str, float]:
    """Compute top-1 accuracy per Elo bracket.

    Returns:
        Dict with "acc_800", "acc_900", etc.
    """
    results = {}
    _, top1 = policy_logprobs.topk(1, dim=-1)
    correct = top1.squeeze(-1).eq(targets)

    for lo in range(elo_min, elo_max, bracket_size):
        hi = lo + bracket_size
        mask = (elos >= lo) & (elos < hi)
        if mask.sum() > 0:
            results[f"acc_{lo}"] = correct[mask].float().mean().item()

    return results
