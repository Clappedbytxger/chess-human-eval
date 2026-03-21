"""FastAPI backend for chess position analysis."""

import sys
from pathlib import Path

import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model.chess_net import ChessNet
from model.board_encoder import BoardEncoder
from model.policy_head import get_legal_move_mask, policy_index_to_move
from evaluation.human_eval import compute_human_eval, compute_elo_curve
from evaluation.stockfish_service import StockfishService
from training.checkpoint import CheckpointManager

app = FastAPI(title="Chess Human Eval API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and engine (loaded on startup)
model: ChessNet | None = None
stockfish: StockfishService | None = None
device: str = "cpu"

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")


@app.on_event("startup")
async def load_model():
    global model, device, stockfish
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ChessNet().to(device)
    if CHECKPOINT_PATH.exists():
        CheckpointManager.load(CHECKPOINT_PATH, model, device=device)
        print(f"Model loaded from checkpoint on {device}")
    else:
        print(f"No checkpoint found -- using untrained model for demo")
    model.eval()

    # Initialize Stockfish
    try:
        stockfish = StockfishService(depth=16, threads=2, hash_mb=128)
        print("Stockfish engine loaded")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Running without Stockfish -- human_eval will not be computed")


@app.on_event("shutdown")
async def shutdown_engine():
    global stockfish
    if stockfish:
        stockfish.close()
        stockfish = None


class AnalyzeRequest(BaseModel):
    fen: str = Field(..., description="FEN string of position to analyze")
    elo: int = Field(1500, ge=800, le=2800, description="Player Elo rating")
    top_k: int = Field(5, ge=1, le=20, description="Number of top moves")


class MoveInfo(BaseModel):
    move: str
    move_uci: str
    probability: float
    eval: float | None = None
    contribution: float | None = None


class AnalyzeResponse(BaseModel):
    fen: str
    elo: int
    top_moves: list[MoveInfo]
    human_eval: float | None = None
    engine_eval: float | None = None
    eval_gap: float | None = None
    difficulty: str | None = None
    value_head_eval: float


class EloCurveRequest(BaseModel):
    fen: str
    elo_min: int = Field(800, ge=800)
    elo_max: int = Field(2800, le=2800)
    elo_step: int = Field(100, ge=50, le=200)


class EloCurvePoint(BaseModel):
    elo: int
    human_eval: float
    difficulty: str


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_position(req: AnalyzeRequest):
    """Analyze a chess position for a given Elo rating."""
    if model is None:
        raise HTTPException(503, "Model not loaded")

    # Validate FEN
    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(400, f"Invalid FEN: {req.fen}")

    # First pass: get top moves from policy head (without Stockfish evals)
    result = compute_human_eval(
        model, req.fen, req.elo, top_k=req.top_k, device=device
    )

    # Second pass: evaluate top moves with Stockfish, then recompute
    if stockfish and result.get("top_moves"):
        top_moves_uci = [
            chess.Move.from_uci(m["move_uci"]) for m in result["top_moves"]
        ]
        sf_evals = stockfish.evaluate_moves(board, top_moves_uci)
        if sf_evals:
            result = compute_human_eval(
                model, req.fen, req.elo,
                stockfish_evals=sf_evals,
                top_k=req.top_k,
                device=device,
            )

    return AnalyzeResponse(**result)


@app.post("/api/analyze-elo-range", response_model=list[EloCurvePoint])
async def analyze_elo_range(req: EloCurveRequest):
    """Analyze a position across all Elo levels."""
    if model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        chess.Board(req.fen)
    except ValueError:
        raise HTTPException(400, f"Invalid FEN: {req.fen}")

    # Get Stockfish evals for all legal moves (computed once, reused across Elos)
    sf_evals = {}
    if stockfish:
        board = chess.Board(req.fen)
        sf_evals = stockfish.evaluate_all_legal(board)

    curve = compute_elo_curve(
        model, req.fen,
        stockfish_evals=sf_evals,
        elo_range=(req.elo_min, req.elo_max),
        elo_step=req.elo_step,
        device=device,
    )
    return [EloCurvePoint(**point) for point in curve]


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "stockfish_loaded": stockfish is not None,
        "device": device,
    }
