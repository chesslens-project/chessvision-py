import chess
import chess.engine
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

CACHE_PATH = Path.home() / ".chessvision_cache.db"

BLUNDER_THRESHOLD    = 300
MISTAKE_THRESHOLD    = 100
INACCURACY_THRESHOLD = 50

PHASE_OPENING_MAX_MOVE   = 15
PHASE_ENDGAME_MAX_MATERIAL = 12


def evaluate_games(
    moves_df: pd.DataFrame,
    stockfish_path: str = "stockfish",
    depth: int = 15,
    batch_size: int = 500,
    sample: Optional[int] = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Evaluate every position in moves_df with Stockfish.

    Resumable: uses a SQLite cache keyed on FEN. If interrupted,
    restart with the same call — already-evaluated positions are
    loaded from cache instantly.

    Parameters
    ----------
    moves_df      : move-level dataframe from parse_pgn()
    stockfish_path: path to Stockfish binary
    depth         : search depth (15 = good balance of speed/quality)
    batch_size    : positions evaluated per Stockfish session
    sample        : if set, only evaluate this many games (for testing)
    cache         : use SQLite cache (always True in production)

    Returns
    -------
    moves_df with eval_before, eval_after, cpl, phase,
    is_blunder, is_mistake, is_inaccuracy filled in.
    """
    df = moves_df.copy()

    if sample is not None:
        game_ids = df["game_id"].unique()[:sample]
        df = df[df["game_id"].isin(game_ids)].copy()
        print(f"Sample mode: evaluating {sample} games "
              f"({len(df)} moves).")

    _ensure_cache()
    unique_fens = df["fen_before"].dropna().unique().tolist()

    if cache:
        cached      = _load_cache(unique_fens)
        to_evaluate = [f for f in unique_fens if f not in cached]
    else:
        cached      = {}
        to_evaluate = unique_fens

    total     = len(unique_fens)
    n_cached  = len(cached)
    n_new     = len(to_evaluate)

    print(f"\nPositions:  {total:,} total")
    print(f"  cached:   {n_cached:,}  ({n_cached/max(total,1)*100:.1f}%)")
    print(f"  new:      {n_new:,}  ({n_new/max(total,1)*100:.1f}%)")

    if n_new > 0:
        est_minutes = n_new / 60 / 60 * depth
        print(f"  Estimated time at depth {depth}: "
              f"~{est_minutes:.0f} min  "
              f"(run overnight if >60 min)\n")
        new_evals = _run_stockfish_batched(
            to_evaluate, stockfish_path, depth, batch_size, cache
        )
        cached.update(new_evals)
    else:
        print("  All positions already cached — loading instantly.\n")

    print("Mapping evaluations to moves...")
    df = _map_evals_to_moves(df, cached)
    df = _compute_cpl(df)
    df = _label_phase(df)
    df = _label_errors(df)
    _print_summary(df)

    return df


def _run_stockfish_batched(
    fens: list,
    stockfish_path: str,
    depth: int,
    batch_size: int,
    cache: bool,
) -> dict:
    """
    Evaluate FENs in batches. Saves to cache after each batch
    so progress is never lost if interrupted.
    """
    results = {}
    batches = [fens[i:i+batch_size] for i in range(0, len(fens), batch_size)]

    print(f"Evaluating {len(fens):,} positions in "
          f"{len(batches)} batches of {batch_size}...")

    with tqdm(total=len(fens), unit="pos", desc="Stockfish") as pbar:
        for batch_num, batch in enumerate(batches):
            batch_results = {}
            try:
                engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                engine.configure({"Threads": 2, "Hash": 128})

                for fen in batch:
                    try:
                        board = chess.Board(fen)
                        info  = engine.analyse(
                            board,
                            chess.engine.Limit(depth=depth),
                            multipv=1,
                        )
                        score = info["score"].white().score(mate_score=10000)
                        batch_results[fen] = score
                    except Exception:
                        batch_results[fen] = None
                    pbar.update(1)

                engine.quit()

            except Exception as e:
                print(f"\nEngine error in batch {batch_num}: {e}")
                print("Progress saved — restart the script to continue.")
                if cache:
                    _save_cache(batch_results)
                results.update(batch_results)
                return results

            if cache:
                _save_cache(batch_results)
            results.update(batch_results)

    return results


def _map_evals_to_moves(df: pd.DataFrame, cached: dict) -> pd.DataFrame:
    """Map FEN evaluations onto the move dataframe."""
    df["eval_before"] = df["fen_before"].map(cached)

    fens_after = (
        df.groupby("game_id")["fen_before"]
        .shift(-1)
    )
    df["eval_after"] = fens_after.map(cached)

    return df


def _compute_cpl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Centipawn loss (CPL) per move.

    For White: CPL = eval_before - eval_after  (positive = worse)
    For Black: CPL = eval_after  - eval_before (positive = worse)
    Both clipped at 0 (a good move has CPL 0, never negative).
    """
    white_mask = df["color"] == "white"
    black_mask = df["color"] == "black"

    df["cpl"] = np.nan

    df.loc[white_mask, "cpl"] = (
        df.loc[white_mask, "eval_before"] -
        df.loc[white_mask, "eval_after"]
    ).clip(lower=0)

    df.loc[black_mask, "cpl"] = (
        df.loc[black_mask, "eval_after"] -
        df.loc[black_mask, "eval_before"]
    ).clip(lower=0)

    return df


def _label_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each move's game phase.

    opening    : move_number <= 15
    endgame    : total non-king material on board <= 12 pieces
    middlegame : everything else
    """
    phases = []
    for _, row in df.iterrows():
        if row["move_number"] <= PHASE_OPENING_MAX_MOVE:
            phases.append("opening")
            continue
        try:
            board    = chess.Board(row["fen_before"])
            material = sum(
                1 for p in board.piece_map().values()
                if p.piece_type != chess.KING
            )
            phases.append("endgame" if material <= PHASE_ENDGAME_MAX_MATERIAL
                          else "middlegame")
        except Exception:
            phases.append("middlegame")

    df["phase"] = phases
    return df


def _label_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each move by CPL thresholds (Lichess standard).

    blunder    : CPL >= 300
    mistake    : 100 <= CPL < 300
    inaccuracy : 50  <= CPL < 100
    """
    cpl = df["cpl"]
    df["is_blunder"]    = cpl >= BLUNDER_THRESHOLD
    df["is_mistake"]    = (cpl >= MISTAKE_THRESHOLD) & (cpl < BLUNDER_THRESHOLD)
    df["is_inaccuracy"] = (cpl >= INACCURACY_THRESHOLD) & (cpl < MISTAKE_THRESHOLD)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable evaluation summary."""
    total    = len(df)
    evaluated = df["cpl"].notna().sum()
    blunders  = df["is_blunder"].sum()
    mistakes  = df["is_mistake"].sum()
    inaccuracies = df["is_inaccuracy"].sum()
    mean_cpl  = df["cpl"].mean()

    print(f"\nEvaluation complete:")
    print(f"  Moves evaluated : {evaluated:,} / {total:,}")
    print(f"  Mean CPL        : {mean_cpl:.1f}")
    print(f"  Blunders        : {blunders:,}  ({blunders/max(evaluated,1)*100:.1f}%)")
    print(f"  Mistakes        : {mistakes:,}  ({mistakes/max(evaluated,1)*100:.1f}%)")
    print(f"  Inaccuracies    : {inaccuracies:,}  ({inaccuracies/max(evaluated,1)*100:.1f}%)")
    print()

    if "phase" in df.columns:
        phase_cpl = df.groupby("phase")["cpl"].mean().round(1)
        print("  Mean CPL by phase:")
        for phase, val in phase_cpl.items():
            print(f"    {phase:<12}: {val}")
    print()


def _ensure_cache() -> None:
    """Create the SQLite cache table if it does not exist."""
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evals (
            fen   TEXT PRIMARY KEY,
            score INTEGER
        )
    """)
    conn.commit()
    conn.close()


def _load_cache(fens: list) -> dict:
    """Load previously computed evaluations from SQLite in chunks of 900."""
    if not fens:
        return {}
    results = {}
    chunk_size = 900
    conn = sqlite3.connect(CACHE_PATH)
    for i in range(0, len(fens), chunk_size):
        chunk = fens[i:i+chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT fen, score FROM evals WHERE fen IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            results[row[0]] = row[1]
    conn.close()
    return results


def _save_cache(evals: dict) -> None:
    """Persist new evaluations to SQLite cache."""
    if not evals:
        return
    conn = sqlite3.connect(CACHE_PATH)
    conn.executemany(
        "INSERT OR IGNORE INTO evals (fen, score) VALUES (?, ?)",
        [(fen, score) for fen, score in evals.items()],
    )
    conn.commit()
    conn.close()
