import pytest
import numpy as np
import pandas as pd
import torch
from chessvision.elo_forecast import (
    compute_accuracy_score,
    build_game_features,
    ELODataset,
    ELOForecaster,
    train_elo_model,
    GAME_FEATURES,
)


def make_mock_data(n_games=100, player="TestPlayer"):
    """Create minimal mock games and moves dataframes."""
    np.random.seed(42)
    game_ids = [f"game_{i}" for i in range(n_games)]

    games_df = pd.DataFrame({
        "game_id":    game_ids,
        "white":      [player if i % 2 == 0 else "Opponent" for i in range(n_games)],
        "black":      ["Opponent" if i % 2 == 0 else player for i in range(n_games)],
        "white_elo":  np.random.randint(1400, 1600, n_games),
        "black_elo":  np.random.randint(1400, 1600, n_games),
        "result":     np.random.choice(["1-0", "0-1", "1/2-1/2"], n_games),
        "date":       [f"2024.{(i//30)+1:02d}.{(i%30)+1:02d}" for i in range(n_games)],
    })

    move_rows = []
    for gid in game_ids:
        n_moves = np.random.randint(20, 50)
        for m in range(n_moves):
            color = "white" if m % 2 == 0 else "black"
            phase = "opening" if m < 15 else ("endgame" if m > 35 else "middlegame")
            cpl   = max(0, np.random.exponential(50))
            move_rows.append({
                "game_id":                   gid,
                "move_number":               m + 1,
                "color":                     color,
                "san":                       "e4",
                "cpl":                       cpl,
                "phase":                     phase,
                "is_blunder":                cpl >= 300,
                "is_mistake":                100 <= cpl < 300,
                "is_inaccuracy":             50 <= cpl < 100,
                "clock_pressure_index":      np.random.uniform(0, 1),
                "under_time_pressure":       np.random.choice([True, False]),
                "position_complexity_score": np.random.uniform(0, 1),
                "mobility_ratio":            np.random.uniform(0.5, 2.0),
                "moves_in_preparation":      np.random.randint(0, 15),
            })

    moves_df = pd.DataFrame(move_rows)
    return games_df, moves_df


@pytest.fixture
def mock_data():
    return make_mock_data(n_games=150)


@pytest.fixture
def game_features(mock_data):
    games_df, moves_df = mock_data
    return build_game_features(moves_df, games_df, "TestPlayer")


def test_accuracy_score_range():
    assert 0 <= compute_accuracy_score(pd.Series([0])) <= 100
    assert 0 <= compute_accuracy_score(pd.Series([100])) <= 100
    assert 0 <= compute_accuracy_score(pd.Series([500])) <= 100


def test_accuracy_score_ordering():
    low_cpl  = compute_accuracy_score(pd.Series([10]))
    high_cpl = compute_accuracy_score(pd.Series([200]))
    assert low_cpl > high_cpl


def test_build_game_features_returns_df(game_features):
    assert isinstance(game_features, pd.DataFrame)


def test_build_game_features_columns(game_features):
    for col in ["accuracy_score", "mean_cpl", "game_length",
                "player_elo", "elo_change_next10"]:
        assert col in game_features.columns


def test_build_game_features_row_count(game_features):
    assert len(game_features) > 0


def test_elo_dataset_length(game_features):
    ds = ELODataset(game_features, seq_len=10)
    assert len(ds) > 0


def test_elo_dataset_item_shape(game_features):
    feat_cols = [c for c in GAME_FEATURES if c in game_features.columns]
    ds        = ELODataset(game_features, seq_len=10)
    X, y      = ds[0]
    assert X.shape == (10, len(feat_cols))
    assert y.shape == ()


def test_elo_forecaster_output_shape():
    model = ELOForecaster(input_size=15)
    x     = torch.randn(4, 20, 15)
    out   = model(x)
    assert out.shape == (4,)


def test_elo_forecaster_forward_no_nan():
    model = ELOForecaster(input_size=15)
    x     = torch.randn(2, 10, 15)
    out   = model(x)
    assert not torch.isnan(out).any()


def test_train_elo_model_runs(game_features, tmp_path):
    model, scaler, history = train_elo_model(
        game_features,
        seq_len     = 5,
        hidden_size = 32,
        epochs      = 3,
        batch_size  = 8,
        patience    = 2,
        output_dir  = tmp_path,
    )
    assert model is not None
    assert scaler is not None
    assert "mae" in history
    assert "rmse" in history
    assert "direction" in history


def test_model_saves_files(game_features, tmp_path):
    train_elo_model(
        game_features,
        seq_len     = 5,
        hidden_size = 32,
        epochs      = 2,
        batch_size  = 8,
        output_dir  = tmp_path,
    )
    assert (tmp_path / "elo_forecaster.pt").exists()
    assert (tmp_path / "elo_scaler.joblib").exists()
    assert (tmp_path / "elo_config.json").exists()
    assert (tmp_path / "elo_history.json").exists()


def test_direction_accuracy_is_percentage(game_features, tmp_path):
    _, _, history = train_elo_model(
        game_features,
        seq_len     = 5,
        hidden_size = 32,
        epochs      = 2,
        batch_size  = 8,
        output_dir  = tmp_path,
    )
    assert 0 <= history["direction"] <= 100
