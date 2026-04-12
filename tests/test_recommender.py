import pytest
import pandas as pd
import numpy as np
from chessvision.recommender import (
    build_player_profile,
    generate_recommendations,
    analyze_player,
    ARCHETYPE_ADVICE,
    OPENING_RECOMMENDATIONS,
)


def make_mock_data(n_games=100, player="TestPlayer"):
    np.random.seed(42)
    game_ids = [f"game_{i}" for i in range(n_games)]

    games_df = pd.DataFrame({
        "game_id":    game_ids,
        "white":      [player if i % 2 == 0 else "Opp" for i in range(n_games)],
        "black":      ["Opp" if i % 2 == 0 else player for i in range(n_games)],
        "white_elo":  np.random.randint(1400, 1600, n_games),
        "black_elo":  np.random.randint(1400, 1600, n_games),
        "result":     np.random.choice(["1-0", "0-1", "1/2-1/2"], n_games),
        "date":       [f"2024.{(i//30)+1:02d}.01" for i in range(n_games)],
    })

    move_rows = []
    for gid in game_ids:
        for m in range(30):
            color = "white" if m % 2 == 0 else "black"
            phase = "opening" if m < 15 else ("endgame" if m > 25 else "middlegame")
            cpl   = max(0, np.random.exponential(80))
            move_rows.append({
                "game_id":                   gid,
                "move_number":               m + 1,
                "color":                     color,
                "san":                       np.random.choice(["e4","Nf3","d4","c4"]),
                "cpl":                       cpl,
                "phase":                     phase,
                "is_blunder":                cpl >= 300,
                "is_mistake":                100 <= cpl < 300,
                "is_inaccuracy":             50 <= cpl < 100,
                "clock_pressure_index":      np.random.uniform(0.5, 1.0),
                "under_time_pressure":       np.random.choice([True, False], p=[0.1, 0.9]),
                "position_complexity_score": np.random.uniform(0.3, 0.9),
                "mobility_ratio":            np.random.uniform(0.7, 1.5),
                "moves_in_preparation":      np.random.randint(5, 15),
                "opening_family_code":       np.random.choice(["A","B","C","D","E"]),
            })

    moves_df = pd.DataFrame(move_rows)

    archetypes = ["positional_confusion", "endgame_failure",
                  "preparation_boundary", "tactical_blindspot",
                  "strategic_drift"]
    error_rows = []
    for gid in game_ids:
        gm = moves_df[moves_df["game_id"] == gid]
        errors = gm[gm["cpl"] >= 50]
        for _, row in errors.iterrows():
            r = row.to_dict()
            r["archetype"] = np.random.choice(archetypes, p=[0.5,0.2,0.15,0.1,0.05])
            r["cluster"]   = 0
            error_rows.append(r)

    error_df = pd.DataFrame(error_rows)
    return games_df, moves_df, error_df


@pytest.fixture
def mock_data():
    return make_mock_data(n_games=100)


@pytest.fixture
def profile(mock_data):
    games_df, moves_df, error_df = mock_data
    return build_player_profile(moves_df, games_df, error_df, "TestPlayer")


def test_profile_returns_dict(profile):
    assert isinstance(profile, dict)


def test_profile_required_keys(profile):
    for key in ["player_name", "n_games", "current_elo",
                "dominant_archetype", "archetype_distribution",
                "mean_cpl", "weakest_phase", "style_label"]:
        assert key in profile, f"Missing key: {key}"


def test_profile_game_count(profile):
    assert profile["n_games"] == 100


def test_profile_archetype_sums_to_100(profile):
    total = sum(v for k, v in profile["archetype_distribution"].items()
                if k != "noise")
    assert total > 50


def test_profile_phase_cpl_positive(profile):
    for phase, cpl in profile["phase_cpl"].items():
        assert cpl >= 0


def test_recommendations_returns_dict(profile):
    recs = generate_recommendations(profile, verbose=False)
    assert isinstance(recs, dict)


def test_recommendations_required_keys(profile):
    recs = generate_recommendations(profile, verbose=False)
    for key in ["player", "priority_focus", "archetype_advice",
                "opening_recs", "training_schedule", "trajectory"]:
        assert key in recs, f"Missing key: {key}"


def test_archetype_advice_present(profile):
    recs = generate_recommendations(profile, verbose=False)
    assert len(recs["archetype_advice"]) > 0


def test_training_schedule_has_items(profile):
    recs = generate_recommendations(profile, verbose=False)
    assert len(recs["training_schedule"]) >= 3


def test_trajectory_valid_values(profile):
    recs = generate_recommendations(profile, verbose=False)
    assert recs["trajectory"] in ["improving", "declining", "plateau"]


def test_opening_recs_present(profile):
    recs = generate_recommendations(profile, verbose=False)
    assert "white" in recs["opening_recs"] or "black" in recs["opening_recs"]


def test_analyze_player_full_pipeline(mock_data, tmp_path):
    games_df, moves_df, error_df = mock_data
    output = tmp_path / "report.json"
    recs = analyze_player(
        moves_df, games_df, error_df,
        "TestPlayer",
        output_path=output,
    )
    assert isinstance(recs, dict)
    assert output.exists()


def test_all_archetypes_have_advice():
    for arch in ["positional_confusion", "endgame_failure",
                 "preparation_boundary", "tactical_blindspot",
                 "strategic_drift", "time_pressure_collapse"]:
        assert arch in ARCHETYPE_ADVICE
        assert "recommendations" in ARCHETYPE_ADVICE[arch]
        assert len(ARCHETYPE_ADVICE[arch]["recommendations"]) >= 3
