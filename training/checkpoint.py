"""Model checkpointing: save and load training state."""

from pathlib import Path
import time

import torch


class CheckpointManager:
    """Manages saving/loading of training checkpoints."""

    def __init__(self, checkpoint_dir: Path, save_interval_min: int = 30):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval_sec = save_interval_min * 60
        self._last_save_time = time.time()
        self.best_loss = float("inf")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object | None,
        epoch: int,
        step: int,
        loss: float,
        filename: str | None = None,
    ) -> Path:
        """Save a complete training checkpoint.

        Args:
            model: The model
            optimizer: The optimizer
            scheduler: LR scheduler (optional)
            epoch: Current epoch
            step: Global step count
            loss: Current loss value
            filename: Custom filename (default: checkpoint_{step}.pt)

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_{step:08d}.pt"

        path = self.checkpoint_dir / filename
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "loss": loss,
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, path)
        print(f"Checkpoint saved: {path}")
        self._last_save_time = time.time()
        return path

    def save_if_due(self, model, optimizer, scheduler, epoch, step, loss) -> Path | None:
        """Save checkpoint if enough time has elapsed since last save."""
        if time.time() - self._last_save_time >= self.save_interval_sec:
            return self.save(model, optimizer, scheduler, epoch, step, loss)
        return None

    def save_best(self, model, optimizer, scheduler, epoch, step, loss) -> Path | None:
        """Save if this is the best loss so far."""
        if loss < self.best_loss:
            self.best_loss = loss
            return self.save(
                model, optimizer, scheduler, epoch, step, loss,
                filename="best_model.pt"
            )
        return None

    @staticmethod
    def load(
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: object | None = None,
        device: str = "cpu",
    ) -> dict:
        """Load a checkpoint and restore model/optimizer/scheduler state.

        Args:
            path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optional optimizer to restore
            scheduler: Optional scheduler to restore
            device: Device to load tensors onto

        Returns:
            Dict with epoch, step, loss from checkpoint
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from {path} (step {checkpoint['step']})")

        return {
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"],
            "loss": checkpoint["loss"],
        }
