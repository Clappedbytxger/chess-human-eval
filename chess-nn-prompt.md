# Chess Position Difficulty Evaluator – Neural Network Project

## Vision

Build a neural network that evaluates chess positions from a **human perspective** rather than an engine perspective. The core insight: Stockfish may say a position is equal (0.0), but if the only move that maintains equality is extremely hard to find for a human, the position is practically losing. This model predicts **which move a human of a given Elo rating would actually play**, then combines those move probabilities with engine evaluations to produce a "human-adjusted evaluation" – a number that reflects how difficult a position truly is for a real player.

### Core Formula

```
human_eval = Σ (probability_human_plays_move_i × stockfish_eval_of_move_i)
```

For ALL legal moves in the position, weighted by the predicted probability that a human of the specified Elo would choose that move.

**Example:** Position is objectively 0.0, but only one move holds the balance. A 1600-rated player finds that move 20% of the time and plays a losing move (+5.0 for the opponent) 80% of the time → human_eval ≈ +4.0 → this position is practically very difficult.

---

## Project Specification

### What the model learns (Multi-Task Network)

The network has **two heads**, similar to AlphaZero/Leela Chess Zero:

1. **Policy Head (Move Prediction):** Given a board position + an Elo rating, predict a probability distribution over ALL legal moves – representing how likely a human at that Elo would play each move. This is NOT the "best" move, but the most "human" move for that skill level.

2. **Value Head (Position Evaluation):** Given a board position, predict the Stockfish-style evaluation (centipawn score or win/draw/loss probabilities). This eliminates the need to run Stockfish at inference time.

The **human-adjusted evaluation** is then computed by combining both heads:
```
human_eval = Σ (policy_head_probability[move_i] × value_head_eval_after[move_i])
```

Note: The value head needs to evaluate the position AFTER each candidate move. This is a design decision to explore – options include:
- Running the value head on the resulting positions of the top-N most likely moves
- Training the value head to directly predict per-move evaluations
- Pre-computing Stockfish evals during data preparation and only using the policy head at inference, with cached evals

**Recommend the most practical and performant approach.**

### Elo as Input Parameter

This is a **single model** that takes Elo as an additional input (NOT separate models per Elo bracket like Maia Chess). The Elo can be provided as:
- A concrete rating value (e.g., 1850)
- A rating range (e.g., 1800–2000)

The model should learn smooth transitions between skill levels. A 1500-rated player and a 1550-rated player should produce similar but slightly different move distributions. Consider how to best encode the Elo input (raw normalized value, embedding, binned encoding, etc.) and recommend the best approach.

### Board Representation

The standard approach for chess NNs is encoding the board as a stack of 2D planes (8×8), with separate binary planes for each piece type and color, plus additional planes for castling rights, en passant, side to move, etc. Follow established practice from Leela/Maia or propose a better encoding if justified.

---

## Training Data

### Source: Lichess Open Database

- Use the Lichess open database (https://database.lichess.org/) – freely available, billions of games in PGN format
- **Time controls:** Rapid and Classical only (these reflect actual understanding, not time-pressure mistakes)
- **Volume:** Use as much data as feasible. Start with a manageable subset for prototyping, then scale up
- **Elo range:** Include games across all rating brackets (e.g., 800–2800) so the model learns the full spectrum

### Data Pipeline

Build a robust data pipeline that:

1. **Downloads** Lichess monthly PGN archives
2. **Filters** for Rapid + Classical time controls
3. **Extracts** training samples: for each position in each game, create a sample containing:
   - Board state (FEN or equivalent)
   - The move that was actually played
   - Both players' Elo ratings
   - Game phase (opening/middlegame/endgame – optional metadata)
4. **Generates engine evaluations:** Run Stockfish on positions to get centipawn evaluations for each legal move. This is the most computationally expensive step – consider:
   - Using a reasonable Stockfish depth (depth 16-20 should suffice)
   - Parallelizing Stockfish analysis
   - Only evaluating the top-N moves plus the played move to save time
   - Using existing evaluated datasets if available (e.g., Lichess analysis data)
5. **Stores** processed data in an efficient format (HDF5, memory-mapped numpy arrays, or similar) for fast training

### Important Data Considerations

- Each position-move pair is one training sample, labeled with the Elo of the player who made the move
- Aggregate move statistics per position across many games at similar Elo levels to get empirical move probabilities
- Handle the cold-start problem: rare positions may have very few data points – consider smoothing or minimum sample thresholds
- Deduplicate common opening positions that appear in thousands of games

---

## Technical Stack

- **Language:** Python
- **ML Framework:** PyTorch
- **Training Environment:** Google Colab (free tier, NVIDIA T4 GPU, ~15 GB VRAM)
- **Chess libraries:** `python-chess` for board representation, move generation, PGN parsing
- **Engine:** Stockfish (via `python-chess` UCI interface) for generating evaluation labels
- **Data processing:** Consider `pandas`, `numpy`, `h5py` or `zarr` for efficient data storage

### Colab Constraints to Keep in Mind

- Free Colab has session time limits (~12 hours) and may disconnect
- Implement **checkpointing** so training can resume after disconnection
- Implement **data streaming** rather than loading everything into RAM
- Consider saving intermediate results to Google Drive
- The T4 has 15 GB VRAM – design batch sizes and model size accordingly

---

## Network Architecture

**Recommend the best architecture for this task**, considering:

- This is a first ML project (user knows basics: layers, backpropagation, loss functions)
- Must train on a T4 GPU in Colab within reasonable time
- Established approaches: ResNets (Leela, Maia), Transformers (newer but heavier)
- The Elo conditioning aspect – how to best inject Elo information into the network
- Multi-task learning with shared backbone + two heads (policy + value)

Provide clear reasoning for the architecture choice. Prioritize something that:
1. Is proven to work for chess
2. Trains within the compute budget
3. Is understandable and debuggable for someone learning ML
4. Can scale up later if desired

---

## Web UI

Build a web interface for using the trained model. Stack recommendation is open – choose what works best (e.g., FastAPI backend + React frontend, or Next.js fullstack, or simpler approach).

### Input
- **Interactive chessboard** where the user can:
  - Set up any position by dragging and dropping pieces
  - Paste a FEN string to load a position
  - Step through moves in a game
- **Elo slider/input** to specify the target player strength (both exact value and range)

### Output – Full Analysis View
- **Human-adjusted evaluation:** The core metric – a single number showing how the position plays out in practice for the given Elo
- **Standard engine evaluation:** What Stockfish/the value head thinks objectively
- **Gap indicator:** The difference between objective and human eval (this IS the "difficulty score")
- **Top moves with probabilities:** Show the most likely human moves at the given Elo, with:
  - Move notation
  - Probability of being played
  - Stockfish eval after that move
  - Color coding (green = good move, red = bad move)
- **Visualization:** A chart/diagram showing how the human eval changes across different Elo levels for the current position (e.g., "this position is fine for 2000+ but deadly for <1500")

### Nice-to-Have Features (lower priority)
- Compare two Elo levels side by side
- Batch analyze an entire opening line
- Find the most "tricky" positions in a given opening repertoire
- Export analysis results

---

## Project Structure

Organize the codebase cleanly but don't over-engineer. Suggested layout:

```
chess-human-eval/
├── data/              # Data downloading, processing, pipeline
├── model/             # Network architecture, training logic
├── training/          # Training scripts, configs, checkpointing  
├── evaluation/        # Model evaluation, metrics, validation
├── web/               # Web UI (frontend + backend)
├── notebooks/         # Colab notebooks for training
└── README.md
```

Feel free to adjust this structure as appropriate.

---

## Development Approach

This is a learning project – the user knows Python and JS well, understands ML basics (layers, backpropagation), but has never trained a neural network from scratch. The codebase should be:

- **Well-commented:** Explain WHY things are done, not just what. Especially for ML-specific patterns (data loading, loss functions, learning rate schedules, etc.)
- **Incremental:** Build and validate each component before moving on
- **Debuggable:** Include logging, visualizations of training progress, sanity checks on data
- **Practical:** Don't over-optimize prematurely. Get something working end-to-end first, then improve

### Suggested Milestones (loose, not rigid phases)

1. **Data pipeline working:** Can download, parse, and extract training samples from Lichess PGN files
2. **Small-scale training:** Train on a subset (~100k positions) to validate the architecture works
3. **Full training:** Scale up to millions of positions, tune hyperparameters
4. **Evaluation:** Measure accuracy – does the model actually predict human moves well? Compare to baselines
5. **Inference pipeline:** Load trained model, feed it a position + Elo, get the human-adjusted eval
6. **Web UI:** Build the interactive frontend

---

## Key References & Inspiration

- **Maia Chess** (https://maiachess.com/) – Predicts human moves at specific Elo levels. Similar goal for the policy head, but uses separate models per Elo bracket
- **Leela Chess Zero** – Open source AlphaZero-style engine with ResNet architecture and policy + value heads
- **Lichess Open Database** (https://database.lichess.org/) – Training data source
- **python-chess** library – For all chess logic (board representation, move generation, PGN parsing, UCI engine interface)
