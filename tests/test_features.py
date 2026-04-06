import pytest
import pandas as pd
import numpy as np
from chessvision.parser import parse_pgn
from chessvision.features import engineer_features, feature_summary
import tempfile
from pathlib import Path

SAMPLE_PGN = """[Event "Test"]
[Site "chess.com"]
[Date "2024.01.15"]
[White "RakkshetSinghaal"]
[Black "Opponent1"]
[WhiteElo "1500"]
[BlackElo "1450"]
[Result "1-0"]
[TimeControl "600+5"]
[ECO "C60"]
[Opening "Ruy Lopez"]

1. e4 { [%clk 0:10:00] } e5 { [%clk 0:10:00] }
2. Nf3 { [%clk 0:09:55] } Nc6 { [%clk 0:09:58] }
3. Bb5 { [%clk 0:09:50] } a6 { [%clk 0:09:55] }
4. Ba4 { [%clk 0:09:45] } Nf6 { [%clk 0:09:50] } 1-0
"""

SAMPLE_PGN_2 = """[Event "Test 2"]
[Site "chess.com"]
[Date "2024.01.20"]
[White "RakkshetSinghaal"]
[Black "Opponent2"]
[WhiteElo "1510"]
[BlackElo "1460"]
[Result "0-1"]
[TimeControl "600+5"]
[ECO "D00"]
[Opening "Queens Pawn"]

1. d4 { [%clk 0:10:00] } d5 { [%clk 0:10:00] }
2. c4 { [%clk 0:09:50] } e6 { [%clk 0:09:55] }
3. Nc3 { [%clk 0:09:40] } Nf6 { [%clk 0:09:50] } 0-1
"""

@pytest.fixture
def parsed_data(tmp_path):
    (tmp_path / "game1.pgn").write_text(SAMPLE_PGN)
    (tmp_path / "game2.pgn").write_text(SAMPLE_PGN_2)
    games_df, moves_df = parse_pgn(tmp_path)
    return games_df, moves_df

@pytest.fixture
def featured_data(parsed_data):
    games_df, moves_df = parsed_data
    return games_df, engineer_features(games_df, moves_df)

def test_feature_columns_exist(featured_data):
    _, moves_df = featured_data
    expected = [
        "clock_pressure_index",
        "clock_pressure_smooth",
        "under_time_pressure",
        "position_complexity_score",
        "novelty_move",
        "moves_in_preparation",
        "mobility_own",
        "mobility_opponent",
        "mobility_ratio",
        "opening_family_code",
        "opening_family_name",
        "first_move_white",
        "first_move_black",
    ]
    for col in expected:
        assert col in moves_df.columns, f"Missing column: {col}"

def test_clock_pressure_range(featured_data):
    _, moves_df = featured_data
    cpi = moves_df["clock_pressure_index"].dropna()
    assert (cpi >= 0.0).all()
    assert (cpi <= 1.0).all()

def test_clock_pressure_first_move(featured_data):
    _, moves_df = featured_data
    first_move = moves_df[moves_df["move_number"] == 1].iloc[0]
    assert first_move["clock_pressure_index"] == pytest.approx(1.0, abs=0.01)

def test_under_time_pressure_is_bool(featured_data):
    _, moves_df = featured_data
    assert moves_df["under_time_pressure"].dtype == bool

def test_complexity_range(featured_data):
    _, moves_df = featured_data
    scores = moves_df["position_complexity_score"].dropna()
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()

def test_mobility_own_positive(featured_data):
    _, moves_df = featured_data
    mob = moves_df["mobility_own"].dropna()
    assert (mob > 0).all()

def test_mobility_ratio_range(featured_data):
    _, moves_df = featured_data
    ratios = moves_df["mobility_ratio"].dropna()
    assert (ratios >= 0.1).all()
    assert (ratios <= 5.0).all()

def test_opening_family_code(featured_data):
    _, moves_df = featured_data
    codes = moves_df["opening_family_code"].dropna().unique()
    for code in codes:
        assert code in ["A", "B", "C", "D", "E"]

def test_opening_family_name_filled(featured_data):
    _, moves_df = featured_data
    has_code = moves_df["opening_family_code"].notna()
    assert moves_df.loc[has_code, "opening_family_name"].notna().all()

def test_first_move_white_is_e4_or_d4(featured_data):
    _, moves_df = featured_data
    whites = moves_df["first_move_white"].dropna().unique()
    for m in whites:
        assert m in ["e4", "d4", "c4", "Nf3", "g3", "b3", "f4"]

def test_novelty_move_is_bool(featured_data):
    _, moves_df = featured_data
    assert moves_df["novelty_move"].dtype == bool

def test_moves_in_preparation_non_negative(featured_data):
    _, moves_df = featured_data
    prep = moves_df["moves_in_preparation"].dropna()
    assert (prep >= 0).all()

def test_row_count_unchanged(parsed_data, featured_data):
    _, moves_before = parsed_data
    _, moves_after  = featured_data
    assert len(moves_before) == len(moves_after)

def test_feature_summary_runs(featured_data):
    _, moves_df = featured_data
    summary = feature_summary(moves_df)
    assert summary is not None
