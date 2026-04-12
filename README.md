# chessvision-py

A Python library for computational chess performance analysis. Takes your PGN game files and runs machine learning on them to tell you what kind of mistakes you make and how to improve.

## What it does

- Parses PGN files from chess.com, lichess, or OTB tournaments
- Evaluates every position with Stockfish (cached, resumable)
- Engineers behavioral features: clock pressure, position complexity, novelty index, mobility ratio
- Clusters your errors into 5 distinct archetypes using HDBSCAN
- Embeds your playing style into a vector space trained on 29.3M elite games
- Forecasts ELO trajectory using a model trained on 67,115 elite players
- Generates a personalized weekly training plan

## Installation

```bash
pip install chessvision-py
```

Requires Stockfish:
```bash
brew install stockfish   # Mac
apt install stockfish    # Linux
```

## Quick start

```python
import chessvision as cv

cv.download_models()

report = cv.analyze("my_games.pgn", player_name="YourUsername")
```

Or step by step:

```python
games, moves = cv.parse_pgn("my_games.pgn")
moves  = cv.evaluate_games(moves)
moves  = cv.engineer_features(games, moves)
errors = cv.run_archetype_analysis(moves)
report = cv.analyze_player(moves, games, errors, "YourUsername")
```

## Pre-trained models

Models are hosted on Hugging Face at `rakkshet/chessvision-models` and download automatically on first use:

- `chess2vec` — style embedding trained on 29.3M elite games (180MB)
- `population_lstm` — ELO trajectory model trained on 67,115 players (5MB)
- `error_archetypes` — HDBSCAN clustering model (50MB)

## Validation results

- chess2vec recovers known stylistic groupings without labeled data
  - Tal vs Nezhmetdinov similarity: 0.979
  - Cross-cluster separation: 0.115 cosine units
- Population LSTM: 53.2% direction accuracy on held-out players (above 50% baseline)
- 5 error archetypes discovered from 96,651 error moves across 4,685 games

## Research

This package is being developed alongside four papers currently in preparation:

1. chessvision: Open-source computational chess analysis (targeting JOSS)
2. Beyond blunders: A ML taxonomy of chess decision errors (targeting JQAS)
3. Chess style in continuous space: Unsupervised player embeddings (targeting CHB)
4. Causal identification of skill acquisition from behavioral panel data (targeting JEBO)

## License

GPL-3.0
