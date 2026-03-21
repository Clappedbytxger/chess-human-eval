"""Core formula: Human-adjusted position evaluation.

human_eval = Σ (probability_human_plays_move_i × stockfish_eval_of_move_i)
"""

import chess
import torch
import numpy as np

from model.chess_net import ChessNet
from model.board_encoder import BoardEncoder
from model.policy_head import get_legal_move_mask, policy_index_to_move


def compute_human_eval(
    model: ChessNet,
    fen: str,
    elo: int,
    stockfish_evals: dict[str, float] | None = None,
    top_k: int = 10,
    device: str = "cpu",
) -> dict:
    """Compute human-adjusted evaluation for a position.

    Args:
        model: Trained ChessNet model
        fen: Board position in FEN
        elo: Player Elo rating
        stockfish_evals: Dict mapping UCI moves to centipawn evals.
                         If None, only returns move probabilities.
        top_k: Number of top moves to consider
        device: Torch device

    Returns:
        Dict with:
            - human_eval: weighted evaluation (if stockfish_evals provided)
            - engine_eval: best engine eval
            - top_moves: list of {move, probability, eval, contribution}
            - difficulty: "easy" / "tricky" / "deadly"
    """
    board = chess.Board(fen)
    encoder = BoardEncoder()

    # Encode inputs
    board_tensor = torch.from_numpy(encoder.encode_board(board)).unsqueeze(0).to(device)
    elo_tensor = torch.tensor([elo], dtype=torch.float32).to(device)
    legal_mask = torch.from_numpy(get_legal_move_mask(board)).unsqueeze(0).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        policy_logprobs, value_pred = model(board_tensor, elo_tensor, legal_mask)

    # Get top-k moves
    probs = torch.exp(policy_logprobs[0])
    top_values, top_indices = probs.topk(min(top_k, probs.size(0)))

    top_moves = []
    for prob, idx in zip(top_values.cpu().numpy(), top_indices.cpu().numpy()):
        move = policy_index_to_move(int(idx), board)
        if move is None or move not in board.legal_moves:
            continue

        move_uci = move.uci()
        move_info = {
            "move": board.san(move),
            "move_uci": move_uci,
            "probability": float(prob),
        }

        if stockfish_evals and move_uci in stockfish_evals:
            move_info["eval"] = stockfish_evals[move_uci]
            move_info["contribution"] = float(prob) * stockfish_evals[move_uci]

        top_moves.append(move_info)

    result = {
        "fen": fen,
        "elo": elo,
        "top_moves": top_moves,
        "value_head_eval": float(value_pred[0, 0].cpu()),
    }

    if stockfish_evals:
        # Human eval = weighted sum of (probability × eval) for top moves
        human_eval = sum(m.get("contribution", 0) for m in top_moves)
        best_eval = max(stockfish_evals.values()) if stockfish_evals else 0

        result["human_eval"] = human_eval
        result["engine_eval"] = best_eval
        result["eval_gap"] = best_eval - human_eval
        result["difficulty"] = _classify_difficulty(best_eval, human_eval)

    return result


def compute_elo_curve(
    model: ChessNet,
    fen: str,
    stockfish_evals: dict[str, float],
    elo_range: tuple[int, int] = (800, 2800),
    elo_step: int = 100,
    device: str = "cpu",
) -> list[dict]:
    """Compute human eval across all Elo levels for a position.

    Returns:
        List of {elo, human_eval, difficulty} dicts
    """
    curve = []
    for elo in range(elo_range[0], elo_range[1] + 1, elo_step):
        result = compute_human_eval(
            model, fen, elo, stockfish_evals, device=device
        )
        curve.append({
            "elo": elo,
            "human_eval": result.get("human_eval", 0),
            "difficulty": result.get("difficulty", "unknown"),
        })
    return curve


def _classify_difficulty(engine_eval: float, human_eval: float) -> str:
    """Classify position difficulty based on eval gap.

    Args:
        engine_eval: Best possible eval (centipawns)
        human_eval: Expected human eval

    Returns:
        "easy", "tricky", or "deadly"
    """
    gap = abs(engine_eval - human_eval)

    if gap < 30:
        return "easy"
    elif gap < 100:
        return "tricky"
    else:
        return "deadly"
