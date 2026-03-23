"""Main training loop."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model.chess_net import ChessNet
from data.dataset import create_dataloader
from .config import TrainConfig
from .losses import CombinedLoss
from .metrics import MetricsTracker, compute_accuracy
from .checkpoint import CheckpointManager


def train(config: TrainConfig | None = None, resume_from: Path | None = None):
    """Run the full training loop.

    Args:
        config: Training configuration. Uses defaults if None.
        resume_from: Path to checkpoint to resume from.
    """
    config = config or TrainConfig()
    device = config.resolve_device()
    print(f"Training on: {device}")

    # Model
    model = ChessNet(
        num_channels=config.num_channels,
        num_blocks=config.num_blocks,
        embed_dim=config.embed_dim,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer + Scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    total_steps = _estimate_total_steps(config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    ) if config.use_cosine_annealing else None

    # Loss
    criterion = CombinedLoss(value_weight=config.value_loss_weight)

    # Checkpoint manager
    ckpt_manager = CheckpointManager(
        config.checkpoint_dir,
        save_interval_min=config.checkpoint_interval_min,
    )

    # Resume if specified
    start_epoch = 0
    global_step = 0
    if resume_from:
        state = ckpt_manager.load(
            resume_from, model, optimizer, scheduler, device
        )
        start_epoch = state["epoch"]
        global_step = state["step"]

    # Data
    train_loader = create_dataloader(
        config.chunks_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Tensorboard
    writer = None
    if config.use_tensorboard:
        writer = SummaryWriter(config.tensorboard_dir)

    # Training loop
    metrics = MetricsTracker()

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_start = time.time()

        for batch in train_loader:
            board = batch["board"].to(device)
            policy_target = batch["policy_target"].to(device)
            elo = batch["elo"].to(device)

            # Forward (no legal mask during training — CE loss handles it)
            policy_logprobs, value_pred = model(board, elo)

            # Value target: placeholder zeros (real targets come from Stockfish evals)
            # TODO: Add actual value targets from Lichess eval dataset
            value_target = torch.zeros_like(value_pred)

            # Loss
            total_loss, policy_loss, value_loss = criterion(
                policy_logprobs, policy_target, value_pred, value_target
            )

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()

            # Metrics
            global_step += 1
            acc = compute_accuracy(policy_logprobs, policy_target)
            metrics.update({
                "total_loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                **acc,
            })

            # Log
            if global_step % config.log_interval == 0:
                avg = metrics.average()
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[Epoch {epoch+1}/{config.num_epochs}] "
                    f"Step {global_step} | "
                    f"Loss: {avg['total_loss']:.4f} "
                    f"(P: {avg['policy_loss']:.4f}, V: {avg['value_loss']:.4f}) | "
                    f"Top-1: {avg['top_1_acc']:.3f} Top-5: {avg['top_5_acc']:.3f} | "
                    f"LR: {lr:.6f}"
                )

                if writer:
                    for key, val in avg.items():
                        writer.add_scalar(f"train/{key}", val, global_step)
                    writer.add_scalar("train/lr", lr, global_step)

                metrics.reset()

            # Checkpoint
            ckpt_manager.save_if_due(
                model, optimizer, scheduler, epoch, global_step, total_loss.item()
            )

        # End of epoch
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {elapsed/60:.1f} minutes")

        ckpt_manager.save(
            model, optimizer, scheduler, epoch + 1, global_step,
            total_loss.item(), filename=f"epoch_{epoch+1}.pt"
        )

    # Final save
    ckpt_manager.save(
        model, optimizer, scheduler, config.num_epochs, global_step,
        total_loss.item(), filename="final_model.pt"
    )

    if writer:
        writer.close()

    print("Training complete!")


def _estimate_total_steps(config: TrainConfig) -> int:
    """Rough estimate of total training steps for scheduler."""
    # Count samples from chunk files
    try:
        import h5py
        total_samples = 0
        for chunk in Path(config.chunks_dir).glob("chunk_*.h5"):
            with h5py.File(chunk, "r") as f:
                total_samples += len(f["boards"])
        steps_per_epoch = total_samples // config.batch_size
        return steps_per_epoch * config.num_epochs
    except Exception:
        return 100_000  # fallback estimate


if __name__ == "__main__":
    train()
