import pytest
import pandas as pd
import chess
from chessvision.parser import parse_pgn
from chessvision.evaluator import (
    evaluate_games,
    _compute_cpl,
    _label_phase,
    _label_errors,
    _ensure_cache,
    _load_cache,
    _save_cache,
    BLUNDER_THRESHOLD,
    MISTAKE_THRESHOLD,
    INACCURACY_THRESHOLD,
)
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

1. e4 { [%clk 0:10:00] } e5 { [%clk 0:10:00] }
2. Nf3 { [%clk 0:09:55] } Nc6 { [%clk 0:09:58] }
3. Bb5 { [%clk 0:09:50] } a6 { [%clk 0:09:55] }
4. Ba4 { [%clk 0:09:45] } Nf6 { [%clk 0:09:50] }
5. O-O { [%clk 0:09:40] } Be7 { [%clk 0:09:45] } 1-0
"""

@pytest.fixture
def small_moves_df(tmp_path):
    (tmp_path / "test.pgn").write_text(SAMPLE_PGN)
    _, moves_df = parse_pgn(tmp_path)
    return moves_df

@pytest.fixture
def evaluated_df(small_moves_df):
    return evaluate_games(
        small_moves_df,
        stockfish_path="stockfish",
        depth=10,
        sample=1,
        cache=False,
    )

def test_evaluate_returns_dataframe(evaluated_df):
    assert isinstance(evaluated_df, pd.DataFrame)

def test_eval_columns_present(evaluated_df):
    for col in ["eval_before", "eval_after", "cpl", "phase",
                "is_blunder", "is_mistake", "is_inaccuracy"]:
        assert col in evaluated_df.columns, f"Missing: {col}"

def test_cpl_non_negative(evaluated_df):
    cpl = evaluated_df["cpl"].dropna()
    assert (cpl >= 0).all(), "CPL should never be negative"

def test_phase_labels_valid(evaluated_df):
    phases = evaluated_df["phase"].dropna().unique()
    for p in phases:
        assert p in ["opening", "middlegame", "endgame"]

def test_opening_phase_correct(evaluated_df):
    early = evaluated_df[evaluated_df["move_number"] <= 15]
    assert (early["phase"] == "opening").all()

def test_error_labels_are_bool(evaluated_df):
    for col in ["is_blunder", "is_mistake", "is_inaccuracy"]:
        assert evaluated_df[col].dtype == bool, f"{col} should be bool"

def test_blunder_threshold(evaluated_df):
    blunders = evaluated_df[evaluated_df["is_blunder"]]
    assert (blunders["cpl"] >= BLUNDER_THRESHOLD).all()

def test_mistake_threshold(evaluated_df):
    mistakes = evaluated_df[evaluated_df["is_mistake"]]
    assert (mistakes["cpl"] >= MISTAKE_THRESHOLD).all()
    assert (mistakes["cpl"] < BLUNDER_THRESHOLD).all()

def test_inaccuracy_threshold(evaluated_df):
    inaccuracies = evaluated_df[evaluated_df["is_inaccuracy"]]
    assert (inaccuracies["cpl"] >= INACCURACY_THRESHOLD).all()
    assert (inaccuracies["cpl"] < MISTAKE_THRESHOLD).all()

def test_row_count_preserved(small_moves_df, evaluated_df):
    assert len(small_moves_df) == len(evaluated_df)

def test_cache_save_and_load():
    _ensure_cache()
    test_evals = {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": 49}
    _save_cache(test_evals)
    loaded = _load_cache(list(test_evals.keys()))
    assert loaded == test_evals

def test_sample_limits_games(small_moves_df):
    result = evaluate_games(
        small_moves_df,
        depth=8,
        sample=1,
        cache=False,
    )
    assert len(result["game_id"].unique()) <= 1
