"""
archetypes.py

Unsupervised error archetype discovery using HDBSCAN.

Takes the evaluated move dataframe and clusters error moves
into behaviorally distinct groups — replacing the naive
blunder/mistake/inaccuracy taxonomy with a data-driven one.

Expected archetypes:
    - Tactical blindspot   : sharp positions, sudden CPL spike
    - Strategic drift      : cumulative small losses, passive pieces
    - Time pressure        : low clock, CPL spikes regardless of position
    - Endgame failure      : winning positions lost in low material
    - Preparation boundary : CPL spike at specific opening move number
    - Positional confusion : complex pawn structures, gradual CPL creep
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

try:
    import hdbscan
except ImportError:
    raise ImportError("Install hdbscan: pip3 install hdbscan")

try:
    import umap
except ImportError:
    raise ImportError("Install umap-learn: pip3 install umap-learn")


ARCHETYPE_LABELS = {
    -1: "noise",
    0:  "tactical_blindspot",
    1:  "strategic_drift",
    2:  "time_pressure_collapse",
    3:  "endgame_failure",
    4:  "preparation_boundary",
    5:  "positional_confusion",
    6:  "other",
}


def build_error_features(moves_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for error clustering.
    Filters to error moves only (CPL >= 50).

    Features per move:
        cpl_norm             : normalized centipawn loss [0,1]
        eval_before_norm     : normalized position evaluation
        clock_pressure_index : time remaining fraction
        complexity           : position complexity score
        mobility_ratio       : own vs opponent mobility
        phase_encoded        : opening=0, middlegame=1, endgame=2
        move_number_norm     : normalized move number
        is_capture           : binary — was it a capture?
        cumulative_cpl_norm  : running total CPL in game (normalized)

    Returns
    -------
    error_df : filtered + feature-enriched dataframe
    """
    df = moves_df.copy()

    # Filter to errors only
    error_df = df[df["cpl"] >= 50].copy()
    print(f"Error moves (CPL >= 50): {len(error_df):,} of {len(df):,} total")

    # Normalize CPL (clip at 1000 to handle mate scores)
    error_df["cpl_norm"] = (
        error_df["cpl"].clip(upper=1000) / 1000.0
    )

    # Normalize eval_before (clip at ±1000 centipawns)
    error_df["eval_before_norm"] = (
        error_df["eval_before"].clip(-1000, 1000) / 1000.0
    )

    # Phase encoding
    phase_map = {"opening": 0.0, "middlegame": 0.5, "endgame": 1.0}
    error_df["phase_encoded"] = error_df["phase"].map(phase_map).fillna(0.5)

    # Normalize move number (cap at 80)
    error_df["move_number_norm"] = (
        error_df["move_number"].clip(upper=80) / 80.0
    )

    # Is capture — detect from SAN notation
    error_df["is_capture"] = error_df["san"].str.contains(
        "x", na=False
    ).astype(float)

    # Cumulative CPL per game (normalized)
    error_df["cumulative_cpl"] = (
        df.groupby("game_id")["cpl"]
        .transform(lambda x: x.fillna(0).cumsum())
        .loc[error_df.index]
    )
    error_df["cumulative_cpl_norm"] = (
        error_df["cumulative_cpl"].clip(upper=3000) / 3000.0
    )

    # Fill missing values
    error_df["clock_pressure_index"] = (
        error_df["clock_pressure_index"].fillna(0.5)
    )
    error_df["position_complexity_score"] = (
        error_df["position_complexity_score"].fillna(0.5)
    )
    error_df["mobility_ratio"] = (
        error_df["mobility_ratio"].fillna(1.0).clip(0.1, 5.0)
    )
    error_df["mobility_ratio_norm"] = (
        (error_df["mobility_ratio"] - 0.1) / (5.0 - 0.1)
    )

    return error_df


def get_feature_matrix(error_df: pd.DataFrame) -> np.ndarray:
    """Extract the numeric feature columns as a numpy array."""
    feature_cols = [
        "cpl_norm",
        "eval_before_norm",
        "clock_pressure_index",
        "position_complexity_score",
        "mobility_ratio_norm",
        "phase_encoded",
        "move_number_norm",
        "is_capture",
        "cumulative_cpl_norm",
    ]
    missing = [c for c in feature_cols if c not in error_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = error_df[feature_cols].values.astype(np.float32)
    print(f"Feature matrix shape: {X.shape}")
    return X


def reduce_dimensions(
    X: np.ndarray,
    n_components: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce feature dimensions with UMAP before clustering.
    HDBSCAN degrades in high dimensions — UMAP first improves results.
    """
    print(f"Reducing {X.shape[1]}D → {n_components}D with UMAP...")
    reducer = umap.UMAP(
        n_components  = n_components,
        n_neighbors   = 30,
        min_dist      = 0.0,
        metric        = "euclidean",
        random_state  = random_state,
    )
    X_reduced = reducer.fit_transform(X)
    print(f"Reduction complete. Shape: {X_reduced.shape}")
    return X_reduced, reducer


def cluster_errors(
    X_reduced: np.ndarray,
    min_cluster_size: int = 200,
    min_samples: int = 50,
) -> np.ndarray:
    """
    Run HDBSCAN on the UMAP-reduced feature matrix.

    min_cluster_size : minimum moves to form a cluster
    min_samples      : controls how conservative clustering is
                       (higher = fewer, denser clusters)
    """
    print(f"Running HDBSCAN "
          f"(min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size  = min_cluster_size,
        min_samples       = min_samples,
        metric            = "euclidean",
        cluster_selection_method = "eom",
        prediction_data   = True,
    )
    labels = clusterer.fit_predict(X_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"Found {n_clusters} clusters, {n_noise:,} noise points "
          f"({n_noise/len(labels)*100:.1f}%)")

    return labels, clusterer


def label_archetypes(
    error_df: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Assign archetype labels to clusters by inspecting
    the mean feature profile of each cluster.

    Labeling rules (in priority order):
        time_pressure_collapse : mean clock_pressure_index < 0.2
        endgame_failure        : mean phase_encoded > 0.8
        preparation_boundary   : mean move_number_norm < 0.2
        tactical_blindspot     : high cpl_norm + high complexity
        positional_confusion   : low cpl_norm + high cumulative_cpl
        strategic_drift        : middlegame + low complexity + high cumul
        other                  : everything else
    """
    df = error_df.copy()
    df["cluster"]   = labels
    df["archetype"] = "noise"

    cluster_ids = [c for c in sorted(df["cluster"].unique()) if c != -1]
    archetype_map = {}

    print("\nCluster profiles:")
    print(f"{'Cluster':>8} {'N':>7} {'CPL':>6} {'Clock':>6} "
          f"{'Phase':>6} {'Complexity':>10} {'MoveN':>6}  Archetype")
    print("-" * 75)

    for cid in cluster_ids:
        mask = df["cluster"] == cid
        sub  = df[mask]
        n    = len(sub)

        mean_cpl        = sub["cpl_norm"].mean()
        mean_clock      = sub["clock_pressure_index"].mean()
        mean_phase      = sub["phase_encoded"].mean()
        mean_complexity = sub["position_complexity_score"].mean()
        mean_move       = sub["move_number_norm"].mean()
        mean_cumul      = sub["cumulative_cpl_norm"].mean()

        # Assign archetype by feature signature
        if mean_clock < 0.2:
            archetype = "time_pressure_collapse"
        elif mean_phase > 0.75:
            archetype = "endgame_failure"
        elif mean_move < 0.2:
            archetype = "preparation_boundary"
        elif mean_cpl > 0.5 and mean_complexity > 0.6:
            archetype = "tactical_blindspot"
        elif mean_cumul > 0.5 and mean_complexity < 0.4:
            archetype = "strategic_drift"
        elif mean_cumul > 0.4 and mean_complexity > 0.4:
            archetype = "positional_confusion"
        else:
            archetype = "other"

        archetype_map[cid] = archetype
        print(f"{cid:>8} {n:>7,} {mean_cpl:>6.3f} {mean_clock:>6.3f} "
              f"{mean_phase:>6.3f} {mean_complexity:>10.3f} "
              f"{mean_move:>6.3f}  {archetype}")

    df["archetype"] = df["cluster"].map(archetype_map).fillna("noise")
    return df, archetype_map


def run_archetype_analysis(
    moves_df: pd.DataFrame,
    min_cluster_size: int = 200,
    min_samples: int = 50,
    umap_components: int = 10,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full pipeline: features → UMAP → HDBSCAN → archetypes.

    Parameters
    ----------
    moves_df         : evaluated move dataframe
    min_cluster_size : HDBSCAN minimum cluster size
    min_samples      : HDBSCAN minimum samples
    umap_components  : UMAP output dimensions
    output_dir       : if set, saves model and results here

    Returns
    -------
    error_df with cluster and archetype columns added
    """
    print("=" * 60)
    print("Phase 3B — Error Archetype Analysis")
    print("=" * 60)

    # Check input
    required = ["cpl", "phase", "move_number", "clock_pressure_index",
                "position_complexity_score", "mobility_ratio"]
    missing = [c for c in required if c not in moves_df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Run evaluate_games() and engineer_features() first."
        )

    evaluated = moves_df["cpl"].notna().sum()
    if evaluated == 0:
        raise ValueError("No evaluated moves found. Run evaluate_games() first.")

    print(f"\nInput: {len(moves_df):,} moves, {evaluated:,} evaluated")

    # Step 1: build features
    print("\n[1/4] Building error features...")
    error_df = build_error_features(moves_df)

    # Step 2: feature matrix
    print("\n[2/4] Extracting feature matrix...")
    X = get_feature_matrix(error_df)

    # Step 3: UMAP reduction
    print("\n[3/4] Dimensionality reduction...")
    X_reduced, reducer = reduce_dimensions(X, n_components=umap_components)

    # Step 4: clustering
    print("\n[4/4] Clustering...")
    labels, clusterer = cluster_errors(
        X_reduced, min_cluster_size, min_samples
    )

    # Label archetypes
    error_df, archetype_map = label_archetypes(error_df, labels)

    # Summary
    print("\nArchetype distribution:")
    arch_counts = error_df["archetype"].value_counts()
    for arch, count in arch_counts.items():
        pct = count / len(error_df) * 100
        print(f"  {arch:<28} : {count:>6,}  ({pct:.1f}%)")

    # Save if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(clusterer, output_dir / "hdbscan_model.joblib")
        joblib.dump(reducer,   output_dir / "umap_reducer.joblib")

        error_df.to_parquet(output_dir / "error_archetypes.parquet",
                            index=False)

        import json
        with open(output_dir / "archetype_map.json", "w") as f:
            json.dump({str(k): v for k, v in archetype_map.items()}, f,
                      indent=2)

        print(f"\nSaved to {output_dir}")

    return error_df


def player_archetype_profile(
    error_df: pd.DataFrame,
    player_name: str,
    games_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute archetype distribution for a specific player.
    Returns a dataframe with archetype counts and percentages.
    """
    if games_df is not None:
        player_games = games_df[
            (games_df["white"] == player_name) |
            (games_df["black"] == player_name)
        ]["game_id"].tolist()
        player_errors = error_df[error_df["game_id"].isin(player_games)]
    else:
        player_errors = error_df

    if len(player_errors) == 0:
        print(f"No error moves found for player: {player_name}")
        return pd.DataFrame()

    profile = (
        player_errors["archetype"]
        .value_counts()
        .reset_index()
    )
    profile.columns = ["archetype", "count"]
    profile["pct"]  = (profile["count"] / len(player_errors) * 100).round(1)

    print(f"\nError archetype profile for {player_name}:")
    print(f"Total error moves: {len(player_errors):,}")
    print(profile.to_string(index=False))

    return profile
