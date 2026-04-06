import chess
import chess.pgn
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple


def engineer_features(
    games_df: pd.DataFrame,
    moves_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add all behavioral and positional features to moves_df.

    Parameters
    ----------
    games_df : game-level dataframe from parse_pgn()
    moves_df : move-level dataframe from parse_pgn()

    Returns
    -------
    moves_df with five new feature columns added.
    """
    print("Engineering features...")
    df = moves_df.copy()

    print("  [1/5] Clock pressure index...")
    df = _add_clock_pressure(df)

    print("  [2/5] Position complexity score...")
    df = _add_complexity(df)

    print("  [3/5] Novelty move index...")
    df = _add_novelty_index(df, games_df)

    print("  [4/5] Mobility ratio...")
    df = _add_mobility(df)

    print("  [5/5] Opening family...")
    df = _add_opening_family(df, games_df)

    print(f"Done. Features added to {len(df)} moves.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1: Clock Pressure Index
# ─────────────────────────────────────────────────────────────────────────────

def _add_clock_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    clock_pressure_index : float [0, 1]
        Fraction of initial time remaining at this move.
        0 = no time left (maximum pressure), 1 = full time remaining.

    under_time_pressure : bool
        True if clock_pressure_index < 0.15 (last 15% of time).

    clock_pressure_smooth : float
        5-move rolling mean of clock_pressure_index within each game+color.
        Smoother signal for ML models.
    """
    df["clock_pressure_index"] = df["clock_fraction"]

    df["under_time_pressure"] = (
        df["clock_pressure_index"].notna() &
        (df["clock_pressure_index"] < 0.15)
    )

    df["clock_pressure_smooth"] = (
        df.groupby(["game_id", "color"])["clock_pressure_index"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2: Position Complexity Score
# ─────────────────────────────────────────────────────────────────────────────

def _add_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    position_complexity_score : float [0, 1]
        Ratio of non-forced legal moves at the position.
        Computed from the FEN using python-chess.

        A "forced" move is one where ALL alternatives would have been
        significantly worse (CPL > 150 relative to the best move).
        When no Stockfish eval is available, we use legal move count
        as a proxy for complexity.

        0 = only one reasonable move (simple/forced)
        1 = many equally viable moves (complex/rich)
    """
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="    complexity", leave=False):
        score = _complexity_from_fen(row["fen_before"], row.get("cpl"))
        scores.append(score)

    df["position_complexity_score"] = scores
    return df


def _complexity_from_fen(fen: str, cpl) -> float:
    """Estimate complexity from legal move count as a proxy."""
    try:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        n = len(legal_moves)
        if n == 0:
            return 0.0
        # Normalize: 1 move = 0.0, 40+ moves = 1.0
        return min(1.0, (n - 1) / 39.0)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3: Novelty Move Index
# ─────────────────────────────────────────────────────────────────────────────

def _add_novelty_index(
    df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    novelty_move : bool
        True at the first move in a game where the position (FEN)
        has never been seen in ANY of the player's previous games.
        This marks the moment the player "left preparation."

    moves_in_preparation : int
        The move number of the novelty - 1.
        How many moves the player was "in book" for this game.
    """
    # Build a chronological list of games per player
    # We need to know which side the focal player was on
    # We'll compute novelty for both white and black perspectives

    # Map game_id -> date for ordering
    date_map = games_df.set_index("game_id")["date"].to_dict()
    df["_date"] = df["game_id"].map(date_map)

    # Get all unique players in the dataset
    all_whites = games_df[["game_id", "white", "date"]].rename(
        columns={"white": "player"})
    all_blacks = games_df[["game_id", "black", "date"]].rename(
        columns={"black": "player"})
    player_games = pd.concat([all_whites, all_blacks]).drop_duplicates("game_id")

    # For each game, find the "focal player" — the one who appears most
    focal_player = _find_focal_player(games_df)

    novelty_flags = {}   # (game_id, move_number) -> bool
    prep_depth    = {}   # game_id -> int

    if focal_player:
        focal_games = games_df[
            (games_df["white"] == focal_player) |
            (games_df["black"] == focal_player)
        ].sort_values("date")

        seen_fens = set()
        for _, game_row in focal_games.iterrows():
            gid = game_row["game_id"]
            game_moves = df[df["game_id"] == gid].sort_values("move_number")
            novelty_found = False
            for _, move_row in game_moves.iterrows():
                fen = move_row["fen_before"]
                mn  = move_row["move_number"]
                if not novelty_found and fen not in seen_fens:
                    novelty_flags[(gid, mn)] = True
                    prep_depth[gid] = mn - 1
                    novelty_found = True
                else:
                    novelty_flags[(gid, mn)] = False
            if not novelty_found:
                prep_depth[gid] = len(game_moves)
            # Add all FENs from this game to seen set AFTER processing
            for fen in game_moves["fen_before"].tolist():
                seen_fens.add(fen)
    else:
        for gid in df["game_id"].unique():
            prep_depth[gid] = None

    df["novelty_move"] = df.apply(
        lambda r: novelty_flags.get((r["game_id"], r["move_number"]), False),
        axis=1
    )
    df["moves_in_preparation"] = df["game_id"].map(prep_depth)
    df.drop(columns=["_date"], inplace=True)

    return df


def _find_focal_player(games_df: pd.DataFrame) -> str:
    """Find the player who appears most frequently across all games."""
    all_players = pd.concat([
        games_df["white"],
        games_df["black"]
    ])
    if all_players.empty:
        return None
    return all_players.value_counts().index[0]


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: Mobility Ratio
# ─────────────────────────────────────────────────────────────────────────────

def _add_mobility(df: pd.DataFrame) -> pd.DataFrame:
    """
    mobility_own : int
        Number of legal moves available to the moving player
        at this position (before the move is made).

    mobility_opponent : int
        Number of legal moves available to the opponent
        at this position (after flipping the turn).

    mobility_ratio : float
        mobility_own / mobility_opponent.
        > 1.0 = more options than opponent (space/activity advantage)
        < 1.0 = fewer options than opponent (cramped/passive position)
        Clipped at [0.1, 5.0] to avoid division-by-zero extremes.
    """
    own_mob, opp_mob, ratios = [], [], []

    for fen in tqdm(df["fen_before"], desc="    mobility", leave=False):
        try:
            board = chess.Board(fen)
            own = board.legal_moves.count()

            # Flip turn to count opponent moves
            board_flipped = board.copy()
            board_flipped.turn = not board.turn
            # Remove checks that would make the flipped board illegal
            try:
                opp = board_flipped.legal_moves.count()
            except Exception:
                opp = own

            ratio = np.clip(own / max(opp, 1), 0.1, 5.0)
            own_mob.append(own)
            opp_mob.append(opp)
            ratios.append(round(ratio, 4))
        except Exception:
            own_mob.append(None)
            opp_mob.append(None)
            ratios.append(None)

    df["mobility_own"]      = own_mob
    df["mobility_opponent"] = opp_mob
    df["mobility_ratio"]    = ratios

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5: Opening Family
# ─────────────────────────────────────────────────────────────────────────────

ECO_FAMILIES = {
    "A": "flank openings",
    "B": "semi-open games",
    "C": "open games",
    "D": "closed and semi-closed games",
    "E": "indian defences",
}

def _add_opening_family(
    df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    opening_family_code : str
        ECO letter (A/B/C/D/E) from the game header.

    opening_family_name : str
        Human-readable family name for the ECO code.

    first_move_white : str
        White's first move SAN — the coarsest opening classifier.

    first_move_black : str
        Black's first move SAN.
    """
    eco_map  = games_df.set_index("game_id")["eco"].to_dict()
    open_map = games_df.set_index("game_id")["opening"].to_dict()

    df["opening_family_code"] = df["game_id"].map(eco_map).apply(
        lambda x: x[0] if isinstance(x, str) and len(x) > 0 and x != "?" else None
    )
    df["opening_family_name"] = df["opening_family_code"].map(ECO_FAMILIES)

    # First move per game per color
    first_moves = (
        df.sort_values("move_number")
        .groupby(["game_id", "color"])["san"]
        .first()
        .reset_index()
    )
    white_first = first_moves[first_moves["color"] == "white"].set_index("game_id")["san"].rename("first_move_white")
    black_first = first_moves[first_moves["color"] == "black"].set_index("game_id")["san"].rename("first_move_black")

    df = df.join(white_first, on="game_id")
    df = df.join(black_first, on="game_id")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def feature_summary(moves_df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a quick summary of all engineered features.
    Useful for sanity-checking before ML training.
    """
    feature_cols = [
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
    ]

    existing = [c for c in feature_cols if c in moves_df.columns]
    summary  = moves_df[existing].describe(include="all").T
    print("\nFeature summary:")
    print(summary[["count", "mean", "std", "min", "max"]].to_string())
    return summary
