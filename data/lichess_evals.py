"""Interface to Lichess evaluation dataset on HuggingFace.

Dataset: ~342M pre-computed Stockfish evaluations of Lichess positions.
"""

from pathlib import Path

import pandas as pd

from .config import LICHESS_EVAL_DATASET


def load_eval_dataset(cache_dir: Path | None = None, split: str = "train"):
    """Load the Lichess evaluation dataset from HuggingFace.

    Uses streaming to avoid downloading the full dataset at once.

    Args:
        cache_dir: Local cache directory
        split: Dataset split

    Returns:
        HuggingFace IterableDataset
    """
    from datasets import load_dataset

    ds = load_dataset(
        LICHESS_EVAL_DATASET,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return ds


def build_eval_lookup(
    parquet_path: Path,
    eval_cache_path: Path | None = None,
    max_evals: int | None = None,
) -> pd.DataFrame:
    """Join extracted samples with Lichess evaluations.

    Matches on FEN (position after removing move counters for better matching).

    Args:
        parquet_path: Path to extracted samples Parquet
        eval_cache_path: Path to cache the eval lookup table
        max_evals: Limit evaluations to process

    Returns:
        DataFrame with eval column added to samples
    """
    samples = pd.read_parquet(parquet_path)

    # Normalize FEN for matching (remove half/full move counters)
    samples["fen_key"] = samples["fen"].apply(_normalize_fen)

    # Stream through eval dataset and build lookup
    eval_lookup = {}
    ds = load_eval_dataset()

    count = 0
    for record in ds:
        fen = record.get("fen", "")
        evals = record.get("evals", [])

        if not fen or not evals:
            continue

        key = _normalize_fen(fen)
        # Take the deepest eval available
        best_eval = _extract_best_eval(evals)
        if best_eval is not None:
            eval_lookup[key] = best_eval

        count += 1
        if max_evals and count >= max_evals:
            break

        if count % 1_000_000 == 0:
            print(f"Processed {count:,} evaluations...")

    # Join
    samples["stockfish_eval"] = samples["fen_key"].map(eval_lookup)
    samples = samples.drop(columns=["fen_key"])

    matched = samples["stockfish_eval"].notna().sum()
    print(f"Matched {matched:,} / {len(samples):,} positions with evals")

    if eval_cache_path:
        samples.to_parquet(eval_cache_path, index=False)

    return samples


def _normalize_fen(fen: str) -> str:
    """Remove half-move and full-move counters from FEN for matching."""
    parts = fen.split()
    return " ".join(parts[:4])  # board, turn, castling, en passant


def _extract_best_eval(evals: list) -> float | None:
    """Extract the best (deepest) evaluation from eval records.

    Returns centipawn evaluation, converting mate scores to large values.
    """
    if not evals:
        return None

    # evals is typically a list of {depth, pvs: [{cp, mate, ...}]}
    best = None
    best_depth = -1

    for ev in evals:
        depth = ev.get("depth", 0)
        pvs = ev.get("pvs", [])
        if not pvs or depth <= best_depth:
            continue

        pv = pvs[0]  # principal variation
        if "cp" in pv:
            best = float(pv["cp"])
            best_depth = depth
        elif "mate" in pv:
            mate_in = pv["mate"]
            # Convert mate to large centipawn value
            best = 10000.0 if mate_in > 0 else -10000.0
            best_depth = depth

    return best
