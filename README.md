# chessvision-py

A Python library for computational chess performance analysis. Transforms 
personal PGN game archives into structured datasets and applies machine 
learning to produce actionable, causally grounded insights.

## What it does

- Parses PGN files from chess.com, lichess, and OTB tournaments
- Evaluates every position with Stockfish (resumable, cached)
- Engineers behavioral features: clock pressure, position complexity, 
  novelty index, mobility ratio, opening family
- Discovers error archetypes using HDBSCAN clustering
- Embeds playing style into continuous vector space (chess2vec)
- Forecasts ELO trajectory with LSTM
- Generates personalized training recommendations

## Installation

```bash
pip install chessvision-py
```

Requires Stockfish installed locally:
```bash
brew install stockfish  # Mac
apt install stockfish   # Linux
```

## Quick start

```python
from chessvision import parse_pgn, evaluate_games, engineer_features
from chessvision import run_archetype_analysis

# Load your games
games, moves = parse_pgn("my_games.pgn")

# Evaluate with Stockfish (resumable)
moves = evaluate_games(moves)

# Engineer features
moves = engineer_features(games, moves)

# Discover your error archetypes
errors = run_archetype_analysis(moves)
print(errors["archetype"].value_counts())
```

## Research

This package accompanies four peer-reviewed publications:

1. chessvision: Open-source computational chess analysis (JOSS)
2. Beyond blunders: A ML taxonomy of chess decision errors (JQAS)
3. Decision fatigue under competitive time pressure (CHB)
4. Causal identification of skill acquisition from behavioral panel data

## License

GPL-3. See LICENSE.
