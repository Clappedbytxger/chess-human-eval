"""Training hyperparameter configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Data
    chunks_dir: Path = Path("data/chunks")
    val_chunks_dir: Path | None = None

    # Model
    num_channels: int = 128
    num_blocks: int = 10
    embed_dim: int = 32

    # Training
    batch_size: int = 1024
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    value_loss_weight: float = 0.5

    # LR Schedule
    warmup_steps: int = 1000
    use_cosine_annealing: bool = True

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_interval_min: int = 30  # save every N minutes
    save_best: bool = True

    # Logging
    log_interval: int = 100  # log every N steps
    eval_interval: int = 1000  # evaluate every N steps
    use_tensorboard: bool = True
    tensorboard_dir: Path = Path("runs")

    # Hardware
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 0  # 0 for Colab
    pin_memory: bool = False

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
