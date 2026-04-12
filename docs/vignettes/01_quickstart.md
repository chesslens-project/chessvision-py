# Vignette 1 — Quick start (amateur player)

Get your personal chess analysis report in 5 steps.

## Requirements

- Stockfish installed: `brew install stockfish`
- Your PGN file downloaded from chess.com or lichess

## Step 1 — Install

````bash
pip install chessvision-py
````

## Step 2 — Download pre-trained models

````python
import chessvision as cv
cv.download_models()
````

## Step 3 — Run your analysis

````python
report = cv.analyze(
    pgn_path    = "my_games.pgn",
    player_name = "YourUsername",
)
````

That is it. The report prints automatically and is saved to `chessvision_report.json`.

## What you get

- Your mean centipawn loss by phase (opening / middlegame / endgame)
- Your error archetype breakdown (what kind of mistakes you make)
- Your playing style compared to grandmasters
- A weekly training schedule targeting your specific weaknesses
- Opening recommendations matched to your style
