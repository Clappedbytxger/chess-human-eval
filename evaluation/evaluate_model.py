"""Evaluate trained model on test set."""

from pathlib import Path

import torch
from tqdm import tqdm

from model.chess_net import ChessNet
from data.dataset import create_dataloader
from training.metrics import MetricsTracker, compute_accuracy, compute_per_elo_accuracy
from training.checkpoint import CheckpointManager


def evaluate(
    checkpoint_path: Path,
    test_chunks_dir: Path,
    batch_size: int = 1024,
    device: str = "auto",
) -> dict:
    """Run full evaluation on test set.

    Args:
        checkpoint_path: Path to model checkpoint
        test_chunks_dir: Directory with test HDF5 chunks
        batch_size: Evaluation batch size
        device: Torch device

    Returns:
        Dict with overall and per-Elo metrics
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = ChessNet().to(device)
    CheckpointManager.load(checkpoint_path, model, device=device)
    model.eval()

    # Data
    test_loader = create_dataloader(
        test_chunks_dir,
        batch_size=batch_size,
        shuffle=False,
    )

    # Evaluate
    metrics = MetricsTracker()
    all_policy_logprobs = []
    all_targets = []
    all_elos = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            board = batch["board"].to(device)
            policy_target = batch["policy_target"].to(device)
            legal_mask = batch["legal_mask"].to(device)
            elo = batch["elo"].to(device)

            policy_logprobs, _ = model(board, elo, legal_mask)

            acc = compute_accuracy(policy_logprobs, policy_target)
            metrics.update(acc, n=board.size(0))

            all_policy_logprobs.append(policy_logprobs.cpu())
            all_targets.append(policy_target.cpu())
            all_elos.append(elo.cpu())

    # Overall metrics
    results = metrics.average()

    # Per-Elo metrics
    all_logprobs = torch.cat(all_policy_logprobs)
    all_tgts = torch.cat(all_targets)
    all_e = torch.cat(all_elos)
    results["per_elo"] = compute_per_elo_accuracy(all_logprobs, all_tgts, all_e)

    print("\n=== Evaluation Results ===")
    print(f"Top-1 Accuracy: {results['top_1_acc']:.4f}")
    print(f"Top-5 Accuracy: {results['top_5_acc']:.4f}")
    print("\nPer-Elo Accuracy:")
    for key, val in sorted(results["per_elo"].items()):
        print(f"  {key}: {val:.4f}")

    return results
