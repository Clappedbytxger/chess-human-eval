"""Stream-filter PGN files for Rapid/Classical games."""

import io
from pathlib import Path

import chess.pgn
import zstandard as zstd

from .config import TIME_CONTROLS, ELO_MIN, ELO_MAX


def classify_time_control(tc_string: str) -> str | None:
    """Classify a Lichess TimeControl header into a category.

    Args:
        tc_string: e.g. "600+0", "180+2", "900+10"

    Returns:
        "bullet", "blitz", "rapid", "classical", or None
    """
    if not tc_string or tc_string == "-":
        return None

    try:
        parts = tc_string.split("+")
        base = int(parts[0])
        inc = int(parts[1]) if len(parts) > 1 else 0
        total = base + 40 * inc  # estimated game duration

        if total < 180:
            return "bullet"
        elif total < 480:
            return "blitz"
        elif total < 1500:
            return "rapid"
        else:
            return "classical"
    except (ValueError, IndexError):
        return None


def stream_filtered_games(pgn_path: Path, max_games: int | None = None):
    """Stream-parse a .pgn.zst file and yield Rapid/Classical games.

    Args:
        pgn_path: Path to .pgn.zst file
        max_games: Optional limit on number of games to yield

    Yields:
        chess.pgn.Game objects that pass the filter
    """
    dctx = zstd.ZstdDecompressor()
    count = 0

    with open(pgn_path, "rb") as compressed:
        with dctx.stream_reader(compressed) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

                headers = game.headers

                # Filter by time control
                tc = classify_time_control(headers.get("TimeControl", ""))
                if tc not in TIME_CONTROLS:
                    continue

                # Filter by Elo range
                try:
                    white_elo = int(headers.get("WhiteElo", "0"))
                    black_elo = int(headers.get("BlackElo", "0"))
                except ValueError:
                    continue

                if not (ELO_MIN <= white_elo <= ELO_MAX and
                        ELO_MIN <= black_elo <= ELO_MAX):
                    continue

                # Skip games with too few moves
                if game.end().ply() < 10:
                    continue

                yield game
                count += 1

                if max_games and count >= max_games:
                    return
