"""Policy head: predicts move probabilities (8x8x73 Leela-style)."""

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Direction vectors for queen-like moves (N, NE, E, SE, S, SW, W, NW)
QUEEN_DIRECTIONS = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
]

# Knight move offsets
KNIGHT_OFFSETS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

# Underpromotion pieces and directions
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMO_DIRS = [-1, 0, 1]  # left-capture, straight, right-capture


class PolicyHead(nn.Module):
    """Outputs move probability distribution over 4672 possible moves.

    Architecture: Conv 1x1 → 73 filters → flatten → mask illegal → softmax
    """

    def __init__(self, in_channels: int = 128, policy_planes: int = 73):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, policy_planes, 1)
        self.policy_planes = policy_planes

    def forward(self, features: torch.Tensor,
                legal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute move probabilities.

        Args:
            features: (batch, 128, 8, 8) from backbone
            legal_mask: (batch, 4672) binary mask, 1 = legal move

        Returns:
            (batch, 4672) log-probabilities over moves
        """
        out = self.conv(features)  # (batch, 73, 8, 8)
        out = out.reshape(out.size(0), -1)  # (batch, 4672)

        if legal_mask is not None:
            # Set illegal moves to -inf before softmax
            out = out.masked_fill(~legal_mask.bool(), float("-inf"))

        return F.log_softmax(out, dim=-1)


def move_to_policy_index(move: chess.Move, flip: bool = False) -> int:
    """Convert a chess.Move to a policy index (0-4671).

    The policy tensor is 8x8x73, indexed as:
    index = from_row * 8 * 73 + from_col * 73 + plane

    Args:
        move: The chess move
        flip: Whether to flip the board (for black's perspective)
    """
    from_sq = move.from_square
    to_sq = move.to_square

    from_row = chess.square_rank(from_sq)
    from_col = chess.square_file(from_sq)
    to_row = chess.square_rank(to_sq)
    to_col = chess.square_file(to_sq)

    if flip:
        from_row = 7 - from_row
        to_row = 7 - to_row

    dr = to_row - from_row
    dc = to_col - from_col

    # Check underpromotions first
    if move.promotion and move.promotion != chess.QUEEN:
        piece_idx = UNDERPROMO_PIECES.index(move.promotion)
        dir_idx = UNDERPROMO_DIRS.index(dc)
        plane = 56 + 8 + piece_idx * 3 + dir_idx
        return from_row * 8 * 73 + from_col * 73 + plane

    # Knight moves
    if (dr, dc) in KNIGHT_OFFSETS:
        knight_idx = KNIGHT_OFFSETS.index((dr, dc))
        plane = 56 + knight_idx
        return from_row * 8 * 73 + from_col * 73 + plane

    # Queen-like moves (including queen promotions treated as normal moves)
    distance = max(abs(dr), abs(dc))
    if distance == 0:
        raise ValueError(f"Invalid move: {move}")

    direction = (
        dr // distance if dr != 0 else 0,
        dc // distance if dc != 0 else 0,
    )
    dir_idx = QUEEN_DIRECTIONS.index(direction)
    plane = (distance - 1) * 8 + dir_idx

    return from_row * 8 * 73 + from_col * 73 + plane


def policy_index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """Convert a policy index back to a chess.Move.

    All coordinates are computed in encoded (possibly flipped) space first,
    then un-flipped at the end to get actual board coordinates.

    Args:
        index: Policy index (0-4671)
        board: Current board state (needed for promotion detection)

    Returns:
        chess.Move or None if invalid
    """
    flip = not board.turn

    # Extract coordinates in encoded space
    from_row = index // (8 * 73)
    remainder = index % (8 * 73)
    from_col = remainder // 73
    plane = remainder % 73

    if plane < 56:
        # Queen-like move
        distance = plane // 8 + 1
        dir_idx = plane % 8
        dr, dc = QUEEN_DIRECTIONS[dir_idx]
        to_row = from_row + dr * distance
        to_col = from_col + dc * distance
    elif plane < 64:
        # Knight move
        knight_idx = plane - 56
        dr, dc = KNIGHT_OFFSETS[knight_idx]
        to_row = from_row + dr
        to_col = from_col + dc
    else:
        # Underpromotion
        underpromo_idx = plane - 64
        piece_idx = underpromo_idx // 3
        dir_idx = underpromo_idx % 3
        dc = UNDERPROMO_DIRS[dir_idx]
        # In encoded space, pawns always promote "upward" (row increases)
        to_row = from_row + 1
        to_col = from_col + dc

    # Bounds check (in encoded space)
    if not (0 <= to_row < 8 and 0 <= to_col < 8):
        return None

    # Un-flip to actual board coordinates
    actual_from_row = (7 - from_row) if flip else from_row
    actual_to_row = (7 - to_row) if flip else to_row

    from_sq = chess.square(from_col, actual_from_row)
    to_sq = chess.square(to_col, actual_to_row)

    # Determine promotion
    promotion = None
    if plane >= 64:
        promotion = UNDERPROMO_PIECES[piece_idx]
    else:
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if actual_to_row == 7 or actual_to_row == 0:
                promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def get_legal_move_mask(board: chess.Board) -> np.ndarray:
    """Create a binary mask of legal moves in policy space.

    Args:
        board: Current board state

    Returns:
        numpy array of shape (4672,) with 1 for legal moves
    """
    mask = np.zeros(4672, dtype=np.float32)
    flip = not board.turn

    for move in board.legal_moves:
        try:
            idx = move_to_policy_index(move, flip=flip)
            if 0 <= idx < 4672:
                mask[idx] = 1.0
        except (ValueError, IndexError):
            continue

    return mask
