"""Build HDF5 training chunks from Parquet samples."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CHUNKS_DIR, CHUNK_SIZE
from .encode import encode_sample


def build_chunks(
    parquet_path: Path,
    output_dir: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    max_samples: int | None = None,
) -> list[Path]:
    """Convert Parquet samples into HDF5 training chunks.

    Each chunk contains:
        - boards: (N, 18, 8, 8) float32
        - policies: (N,) int64 - target policy index
        - legal_masks: (N, 4672) float32
        - elos: (N,) int32 - active player Elo

    Args:
        parquet_path: Path to extracted samples Parquet
        output_dir: Where to save chunks
        chunk_size: Samples per chunk
        max_samples: Limit total samples

    Returns:
        List of created chunk file paths
    """
    output_dir = output_dir or CHUNKS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    if max_samples:
        df = df.head(max_samples)

    total = len(df)
    chunk_paths = []
    chunk_idx = 0

    boards_buf = []
    policies_buf = []
    elos_buf = []

    for i, row in tqdm(df.iterrows(), total=total, desc="Encoding samples"):
        try:
            board_tensor, policy_idx, _ = encode_sample(
                row["fen"], row["move_uci"]
            )
            boards_buf.append(board_tensor)
            policies_buf.append(policy_idx)
            elos_buf.append(row["active_elo"])
        except (ValueError, IndexError) as e:
            continue

        if len(boards_buf) >= chunk_size:
            path = _save_chunk(
                output_dir, chunk_idx,
                boards_buf, policies_buf, elos_buf
            )
            chunk_paths.append(path)
            chunk_idx += 1
            boards_buf, policies_buf, elos_buf = [], [], []

    # Save remaining
    if boards_buf:
        path = _save_chunk(
            output_dir, chunk_idx,
            boards_buf, policies_buf, elos_buf
        )
        chunk_paths.append(path)

    print(f"Created {len(chunk_paths)} chunks in {output_dir}")
    return chunk_paths


def _save_chunk(
    output_dir: Path, chunk_idx: int,
    boards: list, policies: list, elos: list
) -> Path:
    """Save a single HDF5 chunk."""
    path = output_dir / f"chunk_{chunk_idx:05d}.h5"

    with h5py.File(path, "w") as f:
        f.create_dataset("boards", data=np.stack(boards), dtype="float32")
        f.create_dataset("policies", data=np.array(policies), dtype="int64")
        f.create_dataset("elos", data=np.array(elos), dtype="int32")

    return path
