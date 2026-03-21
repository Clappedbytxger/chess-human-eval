"""Visualization utilities for training curves and evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    log_data: dict[str, list[float]],
    output_path: Path | None = None,
):
    """Plot training loss and accuracy curves.

    Args:
        log_data: Dict with keys like "total_loss", "top_1_acc", etc.
                  Each value is a list of measurements.
        output_path: Save plot to file if provided.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    if "total_loss" in log_data:
        axes[0].plot(log_data["total_loss"], label="Total", alpha=0.7)
    if "policy_loss" in log_data:
        axes[0].plot(log_data["policy_loss"], label="Policy", alpha=0.7)
    if "value_loss" in log_data:
        axes[0].plot(log_data["value_loss"], label="Value", alpha=0.7)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Accuracy curves
    if "top_1_acc" in log_data:
        axes[1].plot(log_data["top_1_acc"], label="Top-1")
    if "top_5_acc" in log_data:
        axes[1].plot(log_data["top_5_acc"], label="Top-5")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    # Learning rate
    if "lr" in log_data:
        axes[2].plot(log_data["lr"])
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Step")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def plot_per_elo_accuracy(
    per_elo: dict[str, float],
    output_path: Path | None = None,
):
    """Plot accuracy vs Elo bracket.

    Args:
        per_elo: Dict from compute_per_elo_accuracy (e.g. {"acc_800": 0.35, ...})
        output_path: Save plot to file if provided.
    """
    elos = sorted([int(k.split("_")[1]) for k in per_elo.keys()])
    accs = [per_elo[f"acc_{e}"] for e in elos]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(elos)), accs, tick_label=[str(e) for e in elos])
    plt.xlabel("Elo Bracket")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Move Prediction Accuracy by Elo")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
