"""Global configuration for data pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"

# Elo brackets: (lower, upper) inclusive
ELO_MIN = 800
ELO_MAX = 2800
ELO_BRACKET_SIZE = 100
ELO_BRACKETS = [
    (lo, lo + ELO_BRACKET_SIZE - 1)
    for lo in range(ELO_MIN, ELO_MAX, ELO_BRACKET_SIZE)
]
NUM_ELO_BRACKETS = len(ELO_BRACKETS)  # 20

# Board encoding
NUM_PLANES = 18
BOARD_SIZE = 8

# Policy encoding (Leela-style)
NUM_QUEEN_MOVES = 56   # 7 distances x 8 directions
NUM_KNIGHT_MOVES = 8
NUM_UNDERPROMOTIONS = 9  # 3 piece types x 3 directions
POLICY_PLANES = NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES + NUM_UNDERPROMOTIONS  # 73
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * POLICY_PLANES  # 4672

# Data pipeline
CHUNK_SIZE = 100_000  # positions per HDF5 chunk
TIME_CONTROLS = {"rapid", "classical"}

# Lichess download
LICHESS_DB_URL = "https://database.lichess.org/standard"
LICHESS_EVAL_DATASET = "lichess/chess-evaluations"

# Piece mapping
PIECE_TYPES = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,  # white
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,  # black
}
