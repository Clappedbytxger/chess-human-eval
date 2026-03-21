"""Stockfish integration for move evaluation."""

import sys
from pathlib import Path

import chess
import chess.engine


# Default Stockfish path (project-local binary)
_DEFAULT_PATH = Path(__file__).parent.parent / "engines" / "stockfish" / "stockfish-windows-x86-64.exe"
if sys.platform != "win32":
    _DEFAULT_PATH = Path(__file__).parent.parent / "engines" / "stockfish" / "stockfish"


class StockfishService:
    """Wrapper around python-chess Stockfish engine.

    Evaluates positions and individual moves with configurable depth/time.
    Designed to be initialized once and reused across requests.
    """

    def __init__(
        self,
        stockfish_path: Path | str | None = None,
        depth: int = 16,
        time_limit: float | None = None,
        threads: int = 2,
        hash_mb: int = 128,
    ):
        """Initialize Stockfish engine.

        Args:
            stockfish_path: Path to Stockfish binary. Auto-detected if None.
            depth: Search depth (default 16, good balance of speed/accuracy).
            time_limit: Time limit per analysis in seconds. Overrides depth if set.
            threads: Number of search threads.
            hash_mb: Hash table size in MB.
        """
        path = Path(stockfish_path) if stockfish_path else _DEFAULT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Stockfish not found at {path}. "
                f"Download it to engines/stockfish/ or pass stockfish_path."
            )

        self.engine = chess.engine.SimpleEngine.popen_uci(str(path))
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        self.depth = depth
        self.time_limit = time_limit

    def _get_limit(self) -> chess.engine.Limit:
        if self.time_limit:
            return chess.engine.Limit(time=self.time_limit)
        return chess.engine.Limit(depth=self.depth)

    def evaluate_position(self, board: chess.Board) -> float:
        """Get centipawn evaluation of a position from White's perspective.

        Returns:
            Evaluation in centipawns. Positive = good for White.
            Mate scores are converted to +/-10000.
        """
        info = self.engine.analyse(board, self._get_limit())
        return self._score_to_cp(info["score"], chess.WHITE)

    def evaluate_moves(
        self, board: chess.Board, moves: list[chess.Move]
    ) -> dict[str, float]:
        """Evaluate multiple moves from White's perspective.

        Plays each move, evaluates the resulting position, and returns
        the eval from White's perspective (positive = good for White).

        Args:
            board: Current position.
            moves: List of moves to evaluate.

        Returns:
            Dict mapping move UCI string to centipawn eval (White's perspective).
        """
        evals = {}
        for move in moves:
            if move not in board.legal_moves:
                continue
            board.push(move)
            cp = self.evaluate_position(board)
            board.pop()
            evals[move.uci()] = cp
        return evals

    def evaluate_all_legal(self, board: chess.Board) -> dict[str, float]:
        """Evaluate all legal moves in a position from White's perspective.

        For positions with many legal moves, this can be slow.
        Prefer evaluate_moves() with a filtered list of top candidate moves.
        """
        return self.evaluate_moves(board, list(board.legal_moves))

    def _score_to_cp(self, score: chess.engine.PovScore, turn: chess.Color) -> float:
        """Convert engine score to centipawns from side-to-move perspective."""
        relative = score.pov(turn)
        cp = relative.score(mate_score=10000)
        return float(cp) if cp is not None else 0.0

    def close(self):
        """Shut down the engine process."""
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
