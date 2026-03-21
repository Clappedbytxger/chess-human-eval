"""Encode chess positions (FEN) into 18x8x8 tensors."""

import chess
import numpy as np
import torch


class BoardEncoder:
    """Converts FEN strings to neural network input tensors.

    18 planes x 8x8:
        0-5:   White pieces (P, N, B, R, Q, K)
        6-11:  Black pieces (p, n, b, r, q, k)
        12:    Side to move (1 = white's perspective)
        13-16: Castling rights (WK, WQ, BK, BQ)
        17:    En passant square

    Always encoded from the perspective of the side to move
    (board is flipped for black).
    """

    PIECE_INDICES = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }

    def encode(self, fen: str) -> np.ndarray:
        """Encode a FEN string into an 18x8x8 numpy array."""
        board = chess.Board(fen)
        return self.encode_board(board)

    def encode_board(self, board: chess.Board) -> np.ndarray:
        """Encode a chess.Board into an 18x8x8 numpy array."""
        planes = np.zeros((18, 8, 8), dtype=np.float32)
        flip = not board.turn  # flip if black to move

        # Pieces (planes 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            row = chess.square_rank(square)
            col = chess.square_file(square)
            if flip:
                row = 7 - row

            plane_offset = 0 if piece.color == chess.WHITE else 6
            if flip:
                # Swap colors when viewing from black's perspective
                plane_offset = 6 if piece.color == chess.WHITE else 0

            plane_idx = plane_offset + self.PIECE_INDICES[piece.piece_type]
            planes[plane_idx, row, col] = 1.0

        # Side to move (plane 12) - always 1 since we encode from mover's POV
        planes[12, :, :] = 1.0

        # Castling rights (planes 13-16)
        if flip:
            planes[13, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
            planes[14, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
            planes[15, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
            planes[16, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        else:
            planes[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
            planes[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
            planes[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
            planes[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

        # En passant (plane 17)
        if board.ep_square is not None:
            row = chess.square_rank(board.ep_square)
            col = chess.square_file(board.ep_square)
            if flip:
                row = 7 - row
            planes[17, row, col] = 1.0

        return planes

    def encode_batch(self, fens: list[str]) -> torch.Tensor:
        """Encode a batch of FEN strings into a tensor."""
        arrays = [self.encode(fen) for fen in fens]
        return torch.from_numpy(np.stack(arrays))
