import pytest
import pandas as pd
import numpy as np
from chessvision.archetypes import (
    build_error_features,
    get_feature_matrix,
    run_archetype_analysis,
    player_archetype_profile,
)

@pytest.fixture
def mock_moves_df():
    """Minimal evaluated moves dataframe for testing."""
    np.random.seed(42)
    n = 500
    phases = np.random.choice(["opening", "middlegame", "endgame"], n)
    cpls   = np.random.exponential(80, n)

    return pd.DataFrame({
        "game_id":                  [f"game_{i//20}" for i in range(n)],
        "move_number":              np.random.randint(1, 60, n),
        "color":                    np.random.choice(["white","black"], n),
        "san":                      np.random.choice(["e4","Nf3","Bxe5","O-O"], n),
        "fen_before":               [f"fen_{i}" for i in range(n)],
        "cpl":                      cpls,
        "eval_before":              np.random.uniform(-300, 300, n),
        "eval_after":               np.random.uniform(-300, 300, n),
        "phase":                    phases,
        "is_blunder":               cpls >= 300,
        "is_mistake":               (cpls >= 100) & (cpls < 300),
        "is_inaccuracy":            (cpls >= 50)  & (cpls < 100),
        "clock_pressure_index":     np.random.uniform(0, 1, n),
        "clock_pressure_smooth":    np.random.uniform(0, 1, n),
        "under_time_pressure":      np.random.choice([True, False], n),
        "position_complexity_score": np.random.uniform(0, 1, n),
        "mobility_ratio":           np.random.uniform(0.1, 5.0, n),
        "mobility_own":             np.random.randint(5, 40, n),
        "mobility_opponent":        np.random.randint(5, 40, n),
        "novelty_move":             np.random.choice([True, False], n),
        "moves_in_preparation":     np.random.randint(0, 20, n),
        "opening_family_code":      np.random.choice(["A","B","C","D","E"], n),
        "opening_family_name":      "test opening",
        "first_move_white":         "e4",
        "first_move_black":         "e5",
    })


def test_build_error_features_filters(mock_moves_df):
    error_df = build_error_features(mock_moves_df)
    assert (error_df["cpl"] >= 50).all()


def test_build_error_features_columns(mock_moves_df):
    error_df = build_error_features(mock_moves_df)
    for col in ["cpl_norm", "eval_before_norm", "phase_encoded",
                "move_number_norm", "is_capture", "cumulative_cpl_norm"]:
        assert col in error_df.columns, f"Missing: {col}"


def test_cpl_norm_range(mock_moves_df):
    error_df = build_error_features(mock_moves_df)
    assert (error_df["cpl_norm"] >= 0).all()
    assert (error_df["cpl_norm"] <= 1).all()


def test_phase_encoded_values(mock_moves_df):
    error_df = build_error_features(mock_moves_df)
    assert set(error_df["phase_encoded"].unique()).issubset({0.0, 0.5, 1.0})


def test_get_feature_matrix_shape(mock_moves_df):
    error_df = build_error_features(mock_moves_df)
    X = get_feature_matrix(error_df)
    assert X.ndim == 2
    assert X.shape[1] == 9
    assert X.shape[0] == len(error_df)


def test_feature_matrix_no_nan(mock_moves_df):
    error_df = build_error_features(mock_moves_df)
    X = get_feature_matrix(error_df)
    assert not np.isnan(X).any()


def test_run_archetype_analysis_returns_df(mock_moves_df):
    result = run_archetype_analysis(
        mock_moves_df,
        min_cluster_size=20,
        min_samples=5,
        umap_components=5,
    )
    assert isinstance(result, pd.DataFrame)
    assert "archetype" in result.columns
    assert "cluster" in result.columns


def test_archetype_column_values(mock_moves_df):
    result = run_archetype_analysis(
        mock_moves_df,
        min_cluster_size=20,
        min_samples=5,
        umap_components=5,
    )
    valid = {
        "noise", "tactical_blindspot", "strategic_drift",
        "time_pressure_collapse", "endgame_failure",
        "preparation_boundary", "positional_confusion", "other"
    }
    assert set(result["archetype"].unique()).issubset(valid)


def test_player_profile(mock_moves_df):
    result = run_archetype_analysis(
        mock_moves_df,
        min_cluster_size=20,
        min_samples=5,
        umap_components=5,
    )
    profile = player_archetype_profile(result, "game_0")
    assert isinstance(profile, pd.DataFrame)


def test_row_count_is_error_moves_only(mock_moves_df):
    result = run_archetype_analysis(
        mock_moves_df,
        min_cluster_size=20,
        min_samples=5,
        umap_components=5,
    )
    expected = (mock_moves_df["cpl"] >= 50).sum()
    assert len(result) == expected
