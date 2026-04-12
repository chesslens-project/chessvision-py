# Vignette 3 — Research usage (academic)

Access raw data and model internals for custom analysis.

## Parse and evaluate a large PGN archive

```python
import chessvision as cv
import pandas as pd

games_df, moves_df = cv.parse_pgn("tournament_games/")
moves_df = cv.evaluate_games(moves_df, depth=20)
moves_df = cv.engineer_features(games_df, moves_df)

# Save for downstream analysis
moves_df.to_parquet("moves_evaluated.parquet")
games_df.to_parquet("games.parquet")
```

## Access error archetype raw data

```python
errors = cv.run_archetype_analysis(
    moves_df,
    min_cluster_size = 100,
    min_samples      = 20,
    output_dir       = "archetype_results/",
)

# errors is a tidy dataframe — one row per error move
print(errors.columns.tolist())
print(errors["archetype"].value_counts())
```

## Train chess2vec on your own corpus

```python
# After tokenizing with scripts/lichess_stream.py
from scripts.train_chess2vec import train

model = train(
    token_dir   = Path("tokens/"),
    output_dir  = Path("my_chess2vec/"),
    vector_size = 128,
    workers     = 8,
    epochs      = 10,
)
```

## Population LSTM — access training data

```python
from chessvision.elo_forecast import build_population_features

pop_df = pd.read_parquet("population_data.parquet")
pop_df = build_population_features(pop_df)

# Full feature matrix available for your own models
print(pop_df.head())
print(pop_df.dtypes)
```

## Causal inference — changepoint detection

```python
# Coming in Phase 5 — causal inference module
# See: chessvision.causal.detect_breakpoints()
