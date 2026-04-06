import chess
import chess.engine
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional


CACHE_PATH = Path.home() / ".chessvision_cache.db"

BLUNDER_THRESHOLD    = 300
MISTAKE_THRESHOLD    = 100
INACCURACY_THRESHOLD = 50


def evaluate_games(
    moves_df: pd.DataFrame,
    stockfish_path: str = "stockfish",
    depth: int = 15,
    sample: Optional[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Evaluate positions in moves_df with Stockfish.

    Parameters
    ----------
    moves_df   : move-level dataframe from parse_pgn()
    stockfish_path : path to Stockfish binary (default: 'stockfish' on PATH)
    depth      : Stockfish search depth (15 is a good default)
    sample     : if set, only evaluate this many games (useful for testing)
    cache      : use SQLite cache to avoid re-evaluating known positions

    Returns
    -------
    moves_df with eval_before, eval_after, cpl, phase,
    is_blunder, is_mistake, is_inaccuracy filled in.
    """
    df = moves_df.copy()

    if sample is not None:
        game_ids = df["game_id"].unique()[:sample]
        df = df[df["game_id"].isin(game_ids)].copy()

    unique_fens = df["fen_before"].unique().tolist()

    if cache:
        cached = _load_cache(unique_fens)
        fens_to_evaluate = [f for f in unique_fens if f not in cached]
    else:
        cached = {}
        fens_to_evaluate = unique_fens

    print(f"Positions: {len(unique_fens)} total | "
          f"{len(cached)} cached | {len(fens_to_evaluate)} to evaluate")

    if fens_to_evaluate:
        new_evals = _run_stockfish(fens_to_evaluate, stockfish_path, depth)
        if cache:
            _save_cache(new_evals)
        cached.update(new_evals)

    df["eval_before"] = df["fen_before"].map(cached)
    df["eval_after"]  = df["fen_before"].map(
        dict(zip(df["fen_before"], df["fen_before"].shift(-1).map(cached)))
    )

    df = _compute_cpl(df)
    df = _label_phase(df)
    df = _label_errors(df)

    print(f"Evaluation complete. {df['is_blunder'].sum():.0f} blunders found.")
    return df


def _run_stockfish(fens: list, stockfish_path: str, depth: int) -> dict:
    """Evaluate a list of FEN positions. Returns {fen: centipawn_score}."""
    results = {}
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    for i, fen in enumerate(fens):
        if i % 100 == 0:
            print(f"  Evaluating position {i}/{len(fens)}...", end="\r")
        try:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white().score(mate_score=10000)
            results[fen] = score
        except Exception:
            results[fen] = None

    engine.quit()
    print(f"  Evaluated {len(results)} positions.       ")
    return results


def _compute_cpl(df: pd.DataFrame) -> pd.DataFrame:
    """Compute centipawn loss per move."""
    def cpl_for_row(row):
        if row["eval_before"] is None or row["eval_after"] is None:
            return None
        if row["color"] == "white":
            return max(0, row["eval_before"] - row["eval_after"])
        else:
            return max(0, -row["eval_before"] + row["eval_after"])

    df["cpl"] = df.apply(cpl_for_row, axis=1)
    return df


def _label_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each move's game phase based on move number and material.
    Opening: move <= 15
    Endgame: move > 15 AND fewer than 14 pawns+pieces remain
    Middlegame: everything else
    """
    def phase(row):
        if row["move_number"] <= 15:
            return "opening"
        fen = row["fen_before"]
        try:
            board = chess.Board(fen)
            material = sum(1 for p in board.piece_map().values()
                          if p.piece_type != chess.KING)
            if material <= 12:
                return "endgame"
        except Exception:
            pass
        return "middlegame"

    df["phase"] = df.apply(phase, axis=1)
    return df


def _label_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Label blunders, mistakes, and inaccuracies by CPL thresholds."""
    df["is_blunder"]    = df["cpl"].apply(
        lambda x: bool(x is not None and x >= BLUNDER_THRESHOLD))
    df["is_mistake"]    = df["cpl"].apply(
        lambda x: bool(x is not None and MISTAKE_THRESHOLD <= x < BLUNDER_THRESHOLD))
    df["is_inaccuracy"] = df["cpl"].apply(
        lambda x: bool(x is not None and INACCURACY_THRESHOLD <= x < MISTAKE_THRESHOLD))
    return df


def _load_cache(fens: list) -> dict:
    """Load previously computed evaluations from SQLite."""
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evals
        (fen TEXT PRIMARY KEY, score INTEGER)
    """)
    placeholders = ",".join("?" * len(fens))
    rows = conn.execute(
        f"SELECT fen, score FROM evals WHERE fen IN ({placeholders})", fens
    ).fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def _save_cache(evals: dict) -> None:
    """Save new evaluations to SQLite cache."""
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evals
        (fen TEXT PRIMARY KEY, score INTEGER)
    """)
    conn.executemany(
        "INSERT OR IGNORE INTO evals (fen, score) VALUES (?, ?)",
        [(fen, score) for fen, score in evals.items()]
    )
    conn.commit()
    conn.close()
