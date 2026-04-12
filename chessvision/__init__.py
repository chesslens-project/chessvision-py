"""
chessvision-py

Computational chess performance analysis.
ML-powered insights from personal PGN archives.

Quick start:
    >>> import chessvision as cv
    >>> report = cv.analyze("my_games.pgn", player_name="MyUsername")

Full pipeline:
    >>> games, moves = cv.parse_pgn("my_games.pgn")
    >>> moves = cv.evaluate_games(moves)
    >>> moves = cv.engineer_features(games, moves)
    >>> errors = cv.run_archetype_analysis(moves)
    >>> report = cv.analyze_player(moves, games, errors, "MyUsername")
"""

from .analyze   import analyze
from .parser    import parse_pgn
from .evaluator import evaluate_games
from .features  import engineer_features, feature_summary
from .archetypes import run_archetype_analysis, player_archetype_profile
from .elo_forecast import (
    build_game_features,
    train_elo_model,
    train_win_classifier,
    train_population_lstm,
    fine_tune_on_personal,
    predict_elo_trajectory,
    load_model,
)
from .recommender import (
    build_player_profile,
    generate_recommendations,
    analyze_player,
)
from .models import (
    download_models,
    register_local_models,
    list_models,
    is_downloaded,
)

__version__ = "0.1.0"
__author__  = "Rakkshet Singhaal"
__email__   = "rakkshet.singhaal@kellogg.northwestern.edu"

__all__ = [
    # Main entry point
    "analyze",
    # Pipeline steps
    "parse_pgn",
    "evaluate_games",
    "engineer_features",
    "feature_summary",
    "run_archetype_analysis",
    "player_archetype_profile",
    # ELO forecasting
    "build_game_features",
    "train_elo_model",
    "train_win_classifier",
    "train_population_lstm",
    "fine_tune_on_personal",
    "predict_elo_trajectory",
    "load_model",
    # Recommender
    "build_player_profile",
    "generate_recommendations",
    "analyze_player",
    # Model management
    "download_models",
    "register_local_models",
    "list_models",
    "is_downloaded",
]
