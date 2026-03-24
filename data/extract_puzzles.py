"""Download and extract Lichess puzzles as training samples."""

import csv
import io
from pathlib import Path

import chess
import pandas as pd
import requests
import zstandard as zstd
from tqdm import tqdm

from .config import ELO_MAX, ELO_MIN, PROCESSED_DIR, RAW_DIR

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def download_puzzles(output_dir: Path | None = None) -> Path:
    """Download the Lichess puzzle database (CSV, zst-compressed).

    Returns:
        Path to downloaded .csv.zst file
    """
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lichess_db_puzzle.csv.zst"

    if output_path.exists():
        print(f"Already downloaded: {output_path}")
        return output_path

    print(f"Downloading {PUZZLE_URL}...")
    response = requests.get(PUZZLE_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="puzzles"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Saved to {output_path}")
    return output_path


def extract_puzzles_to_parquet(
    csv_zst_path: Path,
    output_path: Path | None = None,
    max_puzzles: int | None = None,
    min_rating: int = ELO_MIN,
    max_rating: int = ELO_MAX,
) -> Path:
    """Extract puzzle positions into the same Parquet format as game samples.

    Each puzzle contributes one sample: the position after the setup move,
    with the solution move as the target. Puzzle rating is used as active_elo.

    Args:
        csv_zst_path: Path to lichess_db_puzzle.csv.zst
        output_path: Output .parquet path
        max_puzzles: Limit number of puzzles to process
        min_rating: Minimum puzzle rating to include
        max_rating: Maximum puzzle rating to include

    Returns:
        Path to output Parquet file
    """
    if output_path is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DIR / "lichess_puzzles.parquet"

    samples = []
    skipped = 0

    with open(csv_zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        csv_reader = csv.reader(text_stream)

        # Skip header
        header = next(csv_reader)

        for row in tqdm(csv_reader, desc="Extracting puzzles", unit="puzzles"):
            if max_puzzles and len(samples) >= max_puzzles:
                break

            try:
                # CSV columns: PuzzleId, FEN, Moves, Rating, RatingDeviation,
                #              Popularity, NbPlays, Themes, GameUrl, OpeningTags
                puzzle_id, fen, moves_str, rating_str = row[0], row[1], row[2], row[3]
                rating = int(rating_str)

                if rating < min_rating or rating > max_rating:
                    skipped += 1
                    continue

                moves = moves_str.split()
                if len(moves) < 2:
                    skipped += 1
                    continue

                # First move is the setup (opponent's last move)
                # Second move is the puzzle solution (what we train on)
                board = chess.Board(fen)
                setup_move = chess.Move.from_uci(moves[0])
                board.push(setup_move)

                solution_move = chess.Move.from_uci(moves[1])
                if solution_move not in board.legal_moves:
                    skipped += 1
                    continue

                samples.append({
                    "fen": board.fen(),
                    "move_uci": solution_move.uci(),
                    "active_elo": rating,
                    "white_elo": rating,
                    "black_elo": rating,
                    "ply": 20,  # Puzzles are typically midgame
                })

            except (ValueError, IndexError, chess.InvalidMoveError):
                skipped += 1
                continue

    df = pd.DataFrame(samples)
    df.to_parquet(output_path, index=False)
    print(f"Extracted {len(samples)} puzzles (skipped {skipped}) -> {output_path}")
    return output_path
