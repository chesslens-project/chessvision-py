import pytest
import gzip
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import chess
import chess.pgn
import io

from scripts.lichess_stream import (
    find_pgn_zst_files,
    extract_year_month,
    tokenize_game,
    _accept_game,
    _get_phase,
    _parse_eval,
    _parse_clock_bucket,
    process_file,
)


SAMPLE_PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org"]
[Date "2026.03.01"]
[White "PlayerA"]
[Black "PlayerB"]
[WhiteElo "1500"]
[BlackElo "1480"]
[Result "1-0"]
[Variant "Standard"]
[TimeControl "180+2"]

1. e4 { [%eval 0.21] [%clk 0:03:00] } e5 { [%eval 0.18] [%clk 0:03:00] }
2. Nf3 { [%eval 0.25] [%clk 0:02:58] } Nc6 { [%eval 0.22] [%clk 0:02:59] }
3. Bb5 { [%eval 0.31] [%clk 0:02:56] } a6 { [%eval 0.28] [%clk 0:02:57] }
4. Ba4 { [%eval 0.33] [%clk 0:02:54] } Nf6 { [%eval 0.30] [%clk 0:02:55] }
5. O-O { [%eval 0.35] [%clk 0:02:52] } Be7 { [%eval 0.32] [%clk 0:02:53] } 1-0
"""


@pytest.fixture
def sample_game():
    return chess.pgn.read_game(io.StringIO(SAMPLE_PGN))


@pytest.fixture
def sample_zst_file(tmp_path):
    """Create a minimal .pgn.zst file for testing."""
    import zstandard as zstd
    pgn_path = tmp_path / "lichess_db_standard_rated_2026-03.pgn.zst"
    cctx     = zstd.ZstdCompressor()
    with open(pgn_path, "wb") as f:
        f.write(cctx.compress(SAMPLE_PGN.encode("utf-8")))
    return pgn_path


def test_find_single_file(sample_zst_file):
    files = find_pgn_zst_files(sample_zst_file)
    assert len(files) == 1
    assert files[0] == sample_zst_file


def test_find_files_in_directory(sample_zst_file):
    files = find_pgn_zst_files(sample_zst_file.parent)
    assert len(files) >= 1


def test_find_files_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_pgn_zst_files(tmp_path)


def test_extract_year_month(sample_zst_file):
    year, month = extract_year_month(sample_zst_file)
    assert year  == 2026
    assert month == 3


def test_extract_year_month_unknown():
    p = Path("unknown_file.pgn.zst")
    year, month = extract_year_month(p)
    assert year  == 0
    assert month == 0


def test_accept_game_valid(sample_game):
    assert _accept_game(sample_game) is True


def test_accept_game_no_moves():
    game = chess.pgn.Game()
    assert _accept_game(game) is False


def test_accept_game_bot():
    game = chess.pgn.read_game(io.StringIO(
        SAMPLE_PGN.replace("[White ", '[WhiteTitle "BOT"]\n[White ')
    ))
    assert _accept_game(game) is False


def test_accept_game_elo_too_low():
    low = SAMPLE_PGN.replace("1500", "500").replace("1480", "490")
    game = chess.pgn.read_game(io.StringIO(low))
    assert _accept_game(game) is False


def test_tokenize_game_returns_list(sample_game):
    tokens = tokenize_game(sample_game)
    assert isinstance(tokens, list)
    assert len(tokens) >= 5


def test_token_format(sample_game):
    tokens = tokenize_game(sample_game)
    for token in tokens:
        parts = token.split("_")
        assert len(parts) >= 4
        assert parts[1] in ["opening", "middlegame", "endgame"]
        assert parts[2] in ["white", "black"]


def test_get_phase_opening():
    board = chess.Board()
    assert _get_phase(board, 5) == "opening"


def test_get_phase_middlegame():
    board = chess.Board()
    assert _get_phase(board, 20) == "middlegame"


def test_parse_eval_equal():
    assert _parse_eval("[%eval 0.15]", True) == "equal"


def test_parse_eval_winning():
    assert _parse_eval("[%eval 1.50]", True) == "winning"


def test_parse_eval_mate():
    assert _parse_eval("[%eval #3]", True) == "winning_easily"


def test_parse_eval_empty():
    assert _parse_eval("", True) == "unknown"


def test_parse_clock_plenty():
    assert _parse_clock_bucket("[%clk 0:06:00]") == "plenty"


def test_parse_clock_critical():
    assert _parse_clock_bucket("[%clk 0:00:10]") == "critical"


def test_parse_clock_missing():
    assert _parse_clock_bucket("") == "noclock"


def test_process_file_creates_output(sample_zst_file, tmp_path):
    output_dir = tmp_path / "tokens"
    stats      = process_file(sample_zst_file, output_dir)
    assert stats["games_written"] >= 0
    token_files = list(output_dir.glob("*.txt.gz"))
    assert len(token_files) == 1


def test_process_file_skips_if_exists(sample_zst_file, tmp_path):
    output_dir = tmp_path / "tokens"
    stats1     = process_file(sample_zst_file, output_dir)
    stats2     = process_file(sample_zst_file, output_dir)
    assert stats1["games_written"] == stats2["games_written"]


def test_process_file_meta_json(sample_zst_file, tmp_path):
    output_dir = tmp_path / "tokens"
    process_file(sample_zst_file, output_dir)
    meta_files = list(output_dir.glob("*.meta.json"))
    assert len(meta_files) == 1
    with open(meta_files[0]) as f:
        meta = json.load(f)
    assert "games_written" in meta
    assert "tokens_written" in meta
    assert "started" in meta
    assert "finished" in meta
