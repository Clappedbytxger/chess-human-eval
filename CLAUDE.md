# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural network that predicts human chess moves conditioned on Elo rating.
Core formula: `human_eval = Σ (probability_human_plays_move_i × stockfish_eval_of_move_i)`

## Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run training
python -m training.train

# Start backend (from project root)
uvicorn web.backend.main:app --reload --port 8000

# Start frontend
cd web/frontend && npm install && npm run dev

# Run data pipeline (in order)
python -m data.download           # Download Lichess PGN
python -m data.extract_samples    # PGN -> Parquet
python -m data.build_dataset      # Parquet -> HDF5 chunks

# CLI position analysis
python scripts/evaluate_position.py --fen "..." --elo 1500
```

## Architecture

### Model (`model/`)
- **ResNet + FiLM conditioning**: 10 residual blocks, 128 filters, ~3.9M params
- **FiLM** (`film.py`): Elo is discretized into 20 brackets (800-2800), embedded (dim 32), then an MLP produces per-block gamma/beta vectors that scale/shift features. Supports interpolation between brackets.
- **Board encoding** (`board_encoder.py`): FEN → 18×8×8 tensor. Always encoded from the perspective of the side to move (board is flipped for black).
- **Policy head** (`policy_head.py`): Outputs 4672 logits (8×8×73, Leela-style). Handles legal move masking at inference. `move_to_policy_index` and `policy_index_to_move` convert between chess.Move and flat index — these work in "encoded space" (flipped for black) and un-flip at the end.
- **Value head** (`value_head.py`): Scalar output via tanh, Elo-independent.
- **Entry point** (`chess_net.py`): `ChessNet.forward(board, elo, legal_mask=None) → (policy_logprobs, value)`

### Data Pipeline (`data/`)
- Lichess PGN → stream filter (Rapid/Classical) → Parquet → HDF5 chunks
- HDF5 chunks contain: boards (N,18,8,8), policies (N,), elos (N,)
- `dataset.py`: Lazy chunk loading with gc.collect() on swap to stay within RAM
- Legal masks are NOT stored in HDF5 (saves ~350MB/chunk) — computed at inference only

### Training (`training/`)
- SGD + momentum 0.9, cosine annealing, gradient clipping (max_norm=1.0)
- Combined loss: NLL(policy) + 0.5 × MSE(value)
- Value targets are currently zeros (TODO: Lichess eval dataset)
- Checkpoints save full state (model, optimizer, scheduler, epoch, step)

### Web (`web/`)
- **Backend**: FastAPI, loads model on startup, endpoints: `/api/analyze`, `/api/analyze-elo-range`, `/api/health`
- **Frontend**: React 19 + Vite + TypeScript. Uses react-chessboard, chess.js, recharts. Vite proxies `/api` to port 8000.

### Evaluation (`evaluation/`)
- `human_eval.py`: Core formula implementation — gets policy probs per Elo, combines with Stockfish evals
- Difficulty classification: easy/tricky/deadly based on eval gap

## Key Conventions

- All code, comments, commits in English
- Conventional Commits
- Board always encoded from side-to-move perspective (flip for black)
- Policy index encoding: `index = from_row * 8 * 73 + from_col * 73 + plane` (in encoded/flipped space)
- Config constants in `data/config.py` (ELO_MIN, ELO_MAX, NUM_PLANES, POLICY_SIZE, etc.)

## Lessons Learned

- **2026-03-21:** `policy_index_to_move` must compute everything in encoded space (flipped for black) and only un-flip row coordinates at the very end. Un-flipping from_row before applying direction offsets produces wrong squares.
- **2026-03-21:** Windows cp1252 encoding breaks on Unicode arrows (`→`). Use ASCII alternatives (`->`) in print statements.
- **2026-03-21:** Storing legal_masks (4672 floats per sample) in HDF5 causes OOM. Legal masks are only needed at inference, not training — compute them on-the-fly instead.
- **2026-03-21:** When swapping cached chunks, explicitly set `_cached_data = None` and `gc.collect()` before loading the new chunk to avoid holding two chunks in memory simultaneously.
