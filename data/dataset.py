"""PyTorch Dataset for lazy-loading HDF5 training chunks."""

from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ChessDataset(Dataset):
    """Lazy-loading dataset from HDF5 chunks.

    Loads one chunk at a time into memory and serves samples from it.
    Chunk-level shuffling ensures variety without loading everything into RAM.
    """

    def __init__(self, chunks_dir: Path, shuffle_chunks: bool = True):
        """Initialize dataset.

        Args:
            chunks_dir: Directory containing chunk_*.h5 files
            shuffle_chunks: Whether to randomize chunk order
        """
        self.chunk_paths = sorted(Path(chunks_dir).glob("chunk_*.h5"))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunks found in {chunks_dir}")

        if shuffle_chunks:
            random.shuffle(self.chunk_paths)

        # Build index: (chunk_idx, sample_idx_within_chunk)
        self._index = []
        self._chunk_sizes = []
        for chunk_idx, path in enumerate(self.chunk_paths):
            with h5py.File(path, "r") as f:
                size = len(f["boards"])
                self._chunk_sizes.append(size)
                self._index.extend(
                    (chunk_idx, i) for i in range(size)
                )

        # Cache for current chunk
        self._cached_chunk_idx = -1
        self._cached_data = None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        chunk_idx, sample_idx = self._index[idx]

        # Load chunk if not cached
        if chunk_idx != self._cached_chunk_idx:
            self._load_chunk(chunk_idx)

        return {
            "board": torch.from_numpy(self._cached_data["boards"][sample_idx]),
            "policy_target": torch.tensor(
                self._cached_data["policies"][sample_idx], dtype=torch.long
            ),
            "legal_mask": torch.from_numpy(
                self._cached_data["legal_masks"][sample_idx]
            ),
            "elo": torch.tensor(
                self._cached_data["elos"][sample_idx], dtype=torch.float32
            ),
        }

    def _load_chunk(self, chunk_idx: int):
        """Load a chunk into memory cache."""
        path = self.chunk_paths[chunk_idx]
        with h5py.File(path, "r") as f:
            self._cached_data = {
                "boards": f["boards"][:],
                "policies": f["policies"][:],
                "legal_masks": f["legal_masks"][:],
                "elos": f["elos"][:],
            }
        self._cached_chunk_idx = chunk_idx


def create_dataloader(
    chunks_dir: Path,
    batch_size: int = 1024,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from HDF5 chunks.

    Args:
        chunks_dir: Directory with chunk files
        batch_size: Batch size
        num_workers: DataLoader workers (0 for Colab compatibility)
        shuffle: Shuffle samples within DataLoader

    Returns:
        PyTorch DataLoader
    """
    dataset = ChessDataset(chunks_dir, shuffle_chunks=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
