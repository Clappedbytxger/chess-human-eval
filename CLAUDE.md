# Chess Human Eval - CLAUDE.md

## Project Overview
Neural network that predicts human chess moves conditioned on Elo rating.
Core formula: `human_eval = Σ (probability_human_plays_move_i × stockfish_eval_of_move_i)`

## Architecture
- ResNet with 10 residual blocks, 128 filters, FiLM conditioning for Elo
- Board: 18 planes x 8x8 (pieces, turn, castling, en passant)
- Policy head: 8x8x73 (Leela-style move encoding)
- Value head: scalar tanh output

## Tech Stack
- Python 3.10+, PyTorch
- Data: python-chess, pandas, h5py, pyarrow
- Web: FastAPI (backend), React + Vite + TypeScript (frontend)
- Training: Google Colab (T4 GPU)

## Conventions
- All code, comments, commits in English
- Conventional Commits
- Type hints where practical

## Data
- Source: Lichess Open Database (PGN) + Lichess Eval Dataset (HuggingFace)
- Filter: Rapid + Classical only, Elo 800-2800
- Storage: HDF5 chunks (100K positions each)

## Lessons Learned
