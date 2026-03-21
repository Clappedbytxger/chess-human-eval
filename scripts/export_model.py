"""Export trained model to ONNX format for deployment."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.chess_net import ChessNet
from training.checkpoint import CheckpointManager


def export_onnx(checkpoint_path: str, output_path: str = "chess_model.onnx"):
    """Export model to ONNX format.

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_path: Output .onnx file path
    """
    model = ChessNet()
    CheckpointManager.load(Path(checkpoint_path), model)
    model.eval()

    # Dummy inputs
    board = torch.randn(1, 18, 8, 8)
    elo = torch.tensor([1500.0])
    legal_mask = torch.ones(1, 4672)

    torch.onnx.export(
        model,
        (board, elo, legal_mask),
        output_path,
        input_names=["board", "elo", "legal_mask"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "elo": {0: "batch"},
            "legal_mask": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", default="chess_model.onnx", help="Output path")
    args = parser.parse_args()
    export_onnx(args.checkpoint, args.output)
