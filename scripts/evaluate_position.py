"""CLI tool for evaluating chess positions with the trained model."""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.chess_net import ChessNet
from training.checkpoint import CheckpointManager
from evaluation.human_eval import compute_human_eval, compute_elo_curve


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a chess position using the human-adjusted model"
    )
    parser.add_argument("fen", help="FEN string of the position to evaluate")
    parser.add_argument("--elo", type=int, default=1500, help="Player Elo (default: 1500)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top moves to show")
    parser.add_argument("--elo-curve", action="store_true", help="Show eval across all Elo levels")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--no-stockfish", action="store_true", help="Disable Stockfish blending")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = ChessNet().to(device)
    CheckpointManager.load(Path(args.checkpoint), model, device=device)

    # Get Stockfish best move for blending
    sf_best_uci = None
    if not args.no_stockfish:
        try:
            from evaluation.stockfish_service import StockfishService
            sf = StockfishService(depth=16, threads=2, hash_mb=128)
            import chess
            board = chess.Board(args.fen)
            sf_best = sf.best_move(board)
            if sf_best:
                sf_best_uci = sf_best.uci()
                print(f"Stockfish best move: {sf_best_uci}")
            sf.close()
        except FileNotFoundError:
            print("Stockfish not found -- running without blending")

    # Evaluate
    result = compute_human_eval(
        model, args.fen, args.elo,
        stockfish_best_uci=sf_best_uci,
        top_k=args.top_k, device=device,
    )

    # Display results
    print(f"\nPosition: {args.fen}")
    print(f"Elo: {args.elo}")
    print(f"Value Head Eval: {result['value_head_eval']:.3f}")
    print(f"\nTop {args.top_k} predicted moves:")
    print(f"{'Move':<10} {'Probability':>12}")
    print("-" * 24)
    for move in result["top_moves"]:
        print(f"{move['move']:<10} {move['probability']:>11.1%}")

    if args.elo_curve:
        print("\nElo Curve (requires Stockfish evals - showing probabilities only):")
        curve = compute_elo_curve(
            model, args.fen, stockfish_evals={}, device=device
        )
        for point in curve:
            print(f"  Elo {point['elo']}: human_eval={point['human_eval']:.1f}")


if __name__ == "__main__":
    main()
