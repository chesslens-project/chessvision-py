# chessvision-py

A Python library for computational chess performance analysis. Takes your PGN game files and runs machine learning on them to tell you what kind of mistakes you make and how to improve.

## What it does

- Parses PGN files from chess.com, lichess, or OTB tournaments
- Evaluates every position with Stockfish (cached, resumable)
- Clusters your errors into 5 distinct archetypes
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

```python
cv.download_models()
```

## License

GPL-3.0
