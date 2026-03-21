"""Extract training samples from filtered PGN games."""

from pathlib import Path

import chess
import chess.pgn
import pandas as pd
from tqdm import tqdm

from .config import PROCESSED_DIR
from .filter_pgn import stream_filtered_games


def extract_from_game(game: chess.pgn.Game) -> list[dict]:
    """Extract (FEN, move, elos) samples from a single game.

    Args:
        game: A parsed chess game

    Returns:
        List of sample dicts with keys: fen, move_uci, white_elo, black_elo, ply
    """
    headers = game.headers
    white_elo = int(headers.get("WhiteElo", "0"))
    black_elo = int(headers.get("BlackElo", "0"))

    samples = []
    board = game.board()
    node = game

    ply = 0
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move

        # Skip first few moves (opening book territory)
        if ply >= 6:
            samples.append({
                "fen": board.fen(),
                "move_uci": move.uci(),
                "active_elo": white_elo if board.turn == chess.WHITE else black_elo,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "ply": ply,
            })

        board.push(move)
        node = next_node
        ply += 1

    return samples


def extract_samples_to_parquet(
    pgn_path: Path,
    output_path: Path | None = None,
    max_games: int | None = None,
    batch_size: int = 50_000,
) -> Path:
    """Process a PGN file and save extracted samples as Parquet.

    Args:
        pgn_path: Path to .pgn.zst file
        output_path: Output .parquet path. Defaults to processed/{stem}.parquet
        max_games: Limit number of games to process
        batch_size: Number of samples to accumulate before writing

    Returns:
        Path to output Parquet file
    """
    if output_path is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        stem = pgn_path.stem.replace(".pgn", "")
        output_path = PROCESSED_DIR / f"{stem}.parquet"

    all_samples = []
    game_count = 0

    for game in tqdm(
        stream_filtered_games(pgn_path, max_games),
        desc="Extracting samples",
        unit="games",
    ):
        samples = extract_from_game(game)
        all_samples.extend(samples)
        game_count += 1

        # Periodic save for large files
        if len(all_samples) >= batch_size:
            _save_parquet(all_samples, output_path, append=game_count > batch_size)
            all_samples = []

    # Save remaining
    if all_samples:
        _save_parquet(all_samples, output_path, append=output_path.exists())

    print(f"Extracted {game_count} games -> {output_path}")
    return output_path


def _save_parquet(samples: list[dict], path: Path, append: bool = False):
    """Save samples to Parquet file."""
    df = pd.DataFrame(samples)
    if append and path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)
