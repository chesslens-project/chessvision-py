from .analyze   import analyze
from .parser    import parse_pgn
from .evaluator import evaluate_games
from .features  import engineer_features, feature_summary
from .archetypes import run_archetype_analysis, player_archetype_profile
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

try:
    from .elo_forecast import (
        build_game_features,
        train_elo_model,
        train_win_classifier,
        train_population_lstm,
        fine_tune_on_personal,
        predict_elo_trajectory,
        load_model,
    )
except ImportError:
    pass

__version__ = "0.1.0"
__author__  = "Rakkshet Singhaal"
__email__   = "rakkshet.singhaal@kellogg.northwestern.edu"
