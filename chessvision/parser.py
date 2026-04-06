import chess
import chess.pgn
import pandas as pd
import hashlib
import re
from pathlib import Path
from typing import Union, Tuple


def parse_pgn(path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse one PGN file or a directory of PGN files into two tidy dataframes.

    Parameters
    ----------
    path : str or Path
        A single .pgn file or a folder containing .pgn files.

    Returns
    -------
    games_df : pd.DataFrame
        One row per game with metadata (players, ELO, result, time control).
    moves_df : pd.DataFrame
        One row per move with full context (SAN, UCI, FEN, clock time).
    """
    path = Path(path)

    if path.is_dir():
        pgn_files = sorted(path.glob("*.pgn")) + sorted(path.glob("*.PGN"))
        if not pgn_files:
            raise ValueError(f"No PGN files found in: {path}")
    elif path.is_file():
        pgn_files = [path]
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    all_games, all_moves = [], []
    total_rejected = 0

    for pgn_file in pgn_files:
        games, moves, rejected = _parse_file(pgn_file)
        all_games.extend(games)
        all_moves.extend(moves)
        total_rejected += rejected

    games_df = pd.DataFrame(all_games)
    moves_df = pd.DataFrame(all_moves)

    print(f"Parsed {len(games_df)} games and {len(moves_df)} moves "
          f"from {len(pgn_files)} file(s). Rejected: {total_rejected} games.")

    return games_df, moves_df


def _parse_file(pgn_file: Path) -> Tuple[list, list, int]:
    """Parse a single PGN file. Returns (games, moves, rejected_count)."""
    games, moves = [], []
    rejected = 0

    with open(pgn_file, "r", encoding="utf-8", errors="replace") as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                if game.next() is None:
                    rejected += 1
                    continue

                game_id = _make_game_id(game)
                headers = dict(game.headers)
                tc = _parse_time_control(headers.get("TimeControl", ""))

                game_row = {
                    "game_id":          game_id,
                    "source_file":      pgn_file.name,
                    "white":            headers.get("White", "?"),
                    "black":            headers.get("Black", "?"),
                    "white_elo":        _safe_int(headers.get("WhiteElo")),
                    "black_elo":        _safe_int(headers.get("BlackElo")),
                    "result":           headers.get("Result", "?"),
                    "date":             headers.get("Date", "?"),
                    "time_control":     headers.get("TimeControl", "?"),
                    "initial_seconds":  tc["initial"],
                    "increment_seconds": tc["increment"],
                    "eco":              headers.get("ECO", "?"),
                    "opening":          headers.get("Opening", "?"),
                    "termination":      headers.get("Termination", "?"),
                    "platform":         _detect_platform(headers),
                }

                move_rows = _extract_moves(game, game_id, tc["initial"])

                if not move_rows:
                    rejected += 1
                    continue

                game_row["total_moves"] = len(move_rows)
                games.append(game_row)
                moves.extend(move_rows)

            except Exception:
                rejected += 1
                continue

    return games, moves, rejected


def _extract_moves(game, game_id: str, initial_seconds) -> list:
    """Walk through a game and extract one dict per move."""
    rows = []
    board = game.board()
    node = game
    move_number = 0

    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        move_number += 1

        color = "white" if board.turn == chess.WHITE else "black"
        fen_before = board.fen()
        san = board.san(move)
        clock = _parse_clock(next_node.comment)

        rows.append({
            "game_id":         game_id,
            "move_number":     move_number,
            "color":           color,
            "san":             san,
            "uci":             move.uci(),
            "fen_before":      fen_before,
            "clock_remaining": clock,
            "initial_seconds": initial_seconds,
            "clock_fraction":  (clock / initial_seconds)
                               if (clock is not None and initial_seconds)
                               else None,
            # filled later by evaluator.py
            "eval_before":     None,
            "eval_after":      None,
            "cpl":             None,
            "phase":           None,
            "is_blunder":      None,
            "is_mistake":      None,
            "is_inaccuracy":   None,
        })

        board.push(move)
        node = next_node

    return rows


def _make_game_id(game) -> str:
    """Stable 12-char ID based on first 10 moves + date + white player."""
    board = game.board()
    moves = []
    for move in game.mainline_moves():
        moves.append(move.uci())
        if len(moves) >= 10:
            break
    raw = "".join(moves) + game.headers.get("Date", "") + game.headers.get("White", "")
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _parse_clock(comment: str) -> float:
    """Extract seconds remaining from a PGN clock comment."""
    if not comment:
        return None
    match = re.search(r'\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]', comment)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return None


def _parse_time_control(tc: str) -> dict:
    """Parse a time control string like '600+5' into seconds."""
    if not tc or tc in ("-", "?", ""):
        return {"initial": None, "increment": None}
    try:
        if "+" in tc:
            parts = tc.split("+")
            return {"initial": int(parts[0]), "increment": int(parts[1])}
        elif "/" in tc:
            return {"initial": None, "increment": None}
        else:
            return {"initial": int(tc), "increment": 0}
    except (ValueError, IndexError):
        return {"initial": None, "increment": None}


def _detect_platform(headers: dict) -> str:
    """Infer platform from PGN headers."""
    site = headers.get("Site", "").lower()
    if "chess.com" in site:
        return "chess.com"
    if "lichess" in site:
        return "lichess"
    return "unknown"


def _safe_int(value) -> int:
    """Convert to int safely, returning None on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None
