
# chessvision-py

A Python library for computational chess performance analysis. Transforms personal PGN game archives into structured datasets and applies machine learning to produce actionable, causally grounded insights.

## What it does

- **Parses** PGN files from chess.com, lichess, and OTB tournaments

- **Evaluates** every position with Stockfish (resumable, SQLite cached)

- **Engineers** behavioral features: clock pressure, position complexity, novelty index, mobility ratio

- **Clusters** error archetypes using HDBSCAN — 5 distinct error types discovered

- **Embeds** playing style into continuous vector space (chess2vec, trained on 29.3M elite games)

- **Forecasts** ELO trajectory with population LSTM (trained on 67,115 elite players)

- **Recommends** personalized training plans based on your error profile and style

## Installation

```bash

pip install chessvision-py

```

Requires Stockfish installed locally:

```bash

brew install stockfish   # Mac

apt install stockfish    # Linux

```

## Quick start

```python

from chessvision import parse_pgn, evaluate_games, engineer_features

from chessvision import run_archetype_analysis, analyze_player

# Parse your games

games, moves = parse_pgn("my_games.pgn")

# Evaluate with Stockfish (resumable — safe to stop and restart)

moves = evaluate_games(moves)

# Engineer behavioral features

moves = engineer_features(games, moves)

# Discover your error archetypes

errors = run_archetype_analysis(moves)

# Generate your full training report

report = analyze_player(moves, games, errors, player_name="YourUsername")

```

## Key findings from validation

- chess2vec trained on 29.3M elite games recovers known stylistic groupings without labels

- Attacker cluster (Tal/Shirov/Nezhmetdinov): mean similarity 0.965

- Positional cluster (Petrosian/Karpov/Carlsen): mean similarity 0.967

- Cross-cluster separation: 0.115 cosine units

- Population LSTM achieves 53.2% direction accuracy on 67,115 elite players

## Research papers

This package accompanies four peer-reviewed publications:

1. **chessvision**: Open-source computational chess analysis (JOSS)

2. **Beyond blunders**: A ML taxonomy of chess decision errors (JQAS)

3. **Chess style in continuous space**: Unsupervised player embeddings (Computers in Human Behavior)

4. **Causal identification of skill acquisition**: Evidence from behavioral panel data (JEBO)

## Architecture

PGN files → Parser → Stockfish Evaluator → Feature Engineer 

↓

 HDBSCAN Error Archetypes 

chess2vec Style Embeddings 

Population LSTM Forecasting 

↓ 

Personalized Recommender → Training Report

## License

GPL-3.0. See LICENSE.
