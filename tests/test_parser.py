import pandas as pd
import pytest
import tempfile
import os
from chessvision.parser import parse_pgn

SAMPLE_PGN = """[Event "Test Game"]
[Site "chess.com"]
[Date "2024.01.15"]
[White "Player1"]
[Black "Player2"]
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

@pytest.fixture
def sample_pgn_file(tmp_path):
    pgn_file = tmp_path / "test_game.pgn"
    pgn_file.write_text(SAMPLE_PGN)
    return pgn_file

def test_parse_returns_two_dataframes(sample_pgn_file):
    games_df, moves_df = parse_pgn(sample_pgn_file)
    assert isinstance(games_df, pd.DataFrame)
    assert isinstance(moves_df, pd.DataFrame)

def test_parse_correct_game_count(sample_pgn_file):
    games_df, moves_df = parse_pgn(sample_pgn_file)
    assert len(games_df) == 1

def test_parse_correct_move_count(sample_pgn_file):
    games_df, moves_df = parse_pgn(sample_pgn_file)
    assert len(moves_df) == 8  # 4 moves each side

def test_game_metadata_columns(sample_pgn_file):
    games_df, _ = parse_pgn(sample_pgn_file)
    for col in ["game_id", "white", "black", "white_elo", "result", "platform"]:
        assert col in games_df.columns

def test_move_columns(sample_pgn_file):
    _, moves_df = parse_pgn(sample_pgn_file)
    for col in ["game_id", "move_number", "color", "san", "uci", "fen_before", "clock_remaining"]:
        assert col in moves_df.columns

def test_clock_parsed_correctly(sample_pgn_file):
    _, moves_df = parse_pgn(sample_pgn_file)
    assert moves_df["clock_remaining"].iloc[0] == 600.0

def test_elo_parsed_as_int(sample_pgn_file):
    games_df, _ = parse_pgn(sample_pgn_file)
    assert games_df["white_elo"].iloc[0] == 1500

def test_platform_detected(sample_pgn_file):
    games_df, _ = parse_pgn(sample_pgn_file)
    assert games_df["platform"].iloc[0] == "chess.com"

def test_directory_parsing(tmp_path, sample_pgn_file):
    games_df, moves_df = parse_pgn(tmp_path)
    assert len(games_df) == 1

def test_empty_game_rejected(tmp_path):
    empty_pgn = tmp_path / "empty.pgn"
    empty_pgn.write_text('[Event "Empty"]\n[Result "*"]\n\n*\n')
    games_df, moves_df = parse_pgn(tmp_path)
    assert len(games_df) == 0

def test_game_id_is_consistent(sample_pgn_file):
    games_df1, _ = parse_pgn(sample_pgn_file)
    games_df2, _ = parse_pgn(sample_pgn_file)
    assert games_df1["game_id"].iloc[0] == games_df2["game_id"].iloc[0]
