"""
analyze.py

Main entry point for chessvision.
One function call produces the full analysis report.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union

from .parser    import parse_pgn
from .evaluator import evaluate_games
from .features  import engineer_features
from .archetypes import run_archetype_analysis
from .recommender import analyze_player
from .models    import get_cached_path, is_downloaded, download_models


def analyze(
    pgn_path:    Union[str, Path],
    player_name: Optional[str] = None,
    stockfish_path: str = "stockfish",
    depth:       int   = 15,
    output_path: Optional[Union[str, Path]] = None,
    verbose:     bool  = True,
) -> dict:
    """
    Full chessvision analysis pipeline — one function call.

    Takes a PGN file (or folder of PGN files) and returns a
    comprehensive training report with error archetypes, style
    profile, and personalized recommendations.

    Parameters
    ----------
    pgn_path      : path to .pgn file or folder of .pgn files
    player_name   : name of the player to analyze
                    (auto-detected if only one player appears frequently)
    stockfish_path: path to Stockfish binary (default: 'stockfish' on PATH)
    depth         : Stockfish search depth (default: 15)
    output_path   : if set, saves JSON report here
    verbose       : print progress and report

    Returns
    -------
    dict with full analysis results

    Example
    -------
    >>> import chessvision as cv
    >>> report = cv.analyze("my_games.pgn", player_name="MyUsername")
    """
    pgn_path = Path(pgn_path)

    if verbose:
        print("=" * 60)
        print("  ChessVision Analysis Pipeline")
        print("=" * 60)
        print()

    # ── Step 1: Parse ────────────────────────────────────────────
    if verbose:
        print("[1/5] Parsing PGN files...")
    games_df, moves_df = parse_pgn(pgn_path)

    # Auto-detect player name
    if player_name is None:
        player_name = _detect_player(games_df)
        if verbose:
            print(f"  Auto-detected player: {player_name}")

    # ── Step 2: Evaluate ─────────────────────────────────────────
    if verbose:
        print("\n[2/5] Evaluating positions with Stockfish...")
        print("  (This step is cached — safe to stop and restart)")
    moves_df = evaluate_games(
        moves_df,
        stockfish_path = stockfish_path,
        depth          = depth,
        cache          = True,
    )

    # ── Step 3: Feature engineering ──────────────────────────────
    if verbose:
        print("\n[3/5] Engineering behavioral features...")
    moves_df = engineer_features(games_df, moves_df)

    # ── Step 4: Error archetypes ──────────────────────────────────
    if verbose:
        print("\n[4/5] Discovering error archetypes...")

    # Use pre-trained model if available
    arch_model_path = get_cached_path("error_archetypes", "hdbscan_model.joblib")
    error_df = run_archetype_analysis(
        moves_df,
        min_cluster_size = 200,
        min_samples      = 50,
    )

    # ── Step 5: Recommendations ──────────────────────────────────
    if verbose:
        print("\n[5/5] Generating personalized recommendations...")

    chess2vec_path = get_cached_path(
        "chess2vec", "chess2vec.wordvectors"
    )

    report = analyze_player(
        moves_df       = moves_df,
        games_df       = games_df,
        error_df       = error_df,
        player_name    = player_name,
        chess2vec_path = chess2vec_path,
        output_path    = Path(output_path) if output_path else None,
    )

    return report


def _detect_player(games_df: pd.DataFrame) -> str:
    """Auto-detect the focal player — the one who appears most often."""
    all_players = pd.concat([games_df["white"], games_df["black"]])
    return all_players.value_counts().index[0]
