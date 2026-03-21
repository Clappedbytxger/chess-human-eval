"""Encode FEN positions and moves into tensor representations."""

import chess
import numpy as np

from model.board_encoder import BoardEncoder
from model.policy_head import move_to_policy_index, get_legal_move_mask


# Singleton encoder
_encoder = BoardEncoder()


def encode_fen(fen: str) -> np.ndarray:
    """Encode a FEN string into an 18x8x8 numpy array."""
    return _encoder.encode(fen)


def encode_move(move_uci: str, fen: str) -> int:
    """Encode a UCI move string into a policy index.

    Args:
        move_uci: e.g. "e2e4", "g1f3"
        fen: The position FEN (needed for board orientation)

    Returns:
        Policy index (0-4671)
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    flip = not board.turn
    return move_to_policy_index(move, flip=flip)


def encode_sample(fen: str, move_uci: str) -> tuple[np.ndarray, int, np.ndarray]:
    """Encode a complete training sample.

    Args:
        fen: Board position
        move_uci: Move played in UCI notation

    Returns:
        (board_tensor, policy_index, legal_mask)
    """
    board = chess.Board(fen)
    board_tensor = _encoder.encode_board(board)
    move = chess.Move.from_uci(move_uci)
    flip = not board.turn
    policy_index = move_to_policy_index(move, flip=flip)
    legal_mask = get_legal_move_mask(board)

    return board_tensor, policy_index, legal_mask
