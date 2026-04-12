"""
build_population_data.py

Extracts per-player game trajectories from the Lichess Elite Database.
Produces a parquet file with one row per game per player,
suitable for training the population LSTM (Phase 3D Option A).

Usage:
    python3 scripts/build_population_data.py \
        --input  /path/to/lichess_elite_database/ \
        --output /path/to/population_data.parquet \
        --min-games 50
"""

import argparse
import chess
import chess.pgn
import io
import json
import multiprocessing as mp
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("pip3 install tqdm")
    sys.exit(1)


MIN_GAMES     = 50
CHUNK_SIZE_MB = 32


# ─────────────────────────────────────────────────────────────────────────────
# ECO family lookup
# ─────────────────────────────────────────────────────────────────────────────

ECO_FAMILIES = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def encode_result(result: str, color: str) -> float:
    """Encode game result as score for the player (1=win, 0.5=draw, 0=loss)."""
    if result == "1-0":
        return 1.0 if color == "white" else 0.0
    elif result == "0-1":
        return 0.0 if color == "white" else 1.0
    elif result == "1/2-1/2":
        return 0.5
    return np.nan


def parse_game_row(game, color: str, player_name: str) -> Optional[dict]:
    """Extract one row of features from a single game for a player."""
    h = game.headers

    result  = h.get("Result", "*")
    score   = encode_result(result, color)
    if np.isnan(score):
        return None

    try:
        w_elo = int(h.get("WhiteElo", 0))
        b_elo = int(h.get("BlackElo", 0))
    except (ValueError, TypeError):
        return None

    player_elo = w_elo if color == "white" else b_elo
    opp_elo    = b_elo if color == "white" else w_elo

    if player_elo < 100 or opp_elo < 100:
        return None

    # Count moves
    move_count = sum(1 for _ in game.mainline_moves())
    if move_count < 5:
        return None

    # ECO
    eco  = h.get("ECO", "?")
    eco_family = ECO_FAMILIES.get(eco[0], -1) if eco and eco != "?" else -1

    # Date
    date = h.get("Date", "????.??.??")
    try:
        year  = int(date[:4])
        month = int(date[5:7])
    except (ValueError, IndexError):
        year, month = 0, 0

    return {
        "player":          player_name,
        "color":           color,
        "player_elo":      player_elo,
        "opp_elo":         opp_elo,
        "elo_delta":       opp_elo - player_elo,
        "score":           score,
        "result":          result,
        "move_count":      move_count,
        "eco_family":      eco_family,
        "year":            year,
        "month":           month,
        "date_str":        date,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ─────────────────────────────────────────────────────────────────────────────

def process_chunk(chunk_text: str) -> list:
    """Parse a PGN chunk and return list of game feature dicts."""
    rows       = []
    game_texts = re.split(r'\n\n(?=\[Event )', chunk_text)

    for game_text in game_texts:
        if not game_text.strip() or "[Event " not in game_text:
            continue
        try:
            game = chess.pgn.read_game(io.StringIO(game_text))
            if game is None:
                continue
            h = game.headers

            # Skip bots and variants
            if h.get("Variant", "Standard") != "Standard":
                continue
            if h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT":
                continue

            white = h.get("White", "?")
            black = h.get("Black", "?")
            if white == "?" or black == "?":
                continue

            row_w = parse_game_row(game, "white", white)
            row_b = parse_game_row(game, "black", black)
            if row_w:
                rows.append(row_w)
            if row_b:
                rows.append(row_b)

        except Exception:
            continue

    return rows


def generate_chunks(filepath: Path):
    """Yield text chunks from a plain PGN file."""
    chunk_size  = CHUNK_SIZE_MB * 1024 * 1024
    text_buffer = ""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        while True:
            raw = f.read(chunk_size)
            if not raw:
                break
            text_buffer += raw
            boundary     = text_buffer.rfind("\n\n[Event ")
            if boundary == -1:
                continue
            yield text_buffer[:boundary]
            text_buffer = text_buffer[boundary:]
        if text_buffer.strip():
            yield text_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_population_data(
    input_dir:  Path,
    output_path: Path,
    min_games:  int = 50,
    workers:    int = 8,
) -> pd.DataFrame:
    """
    Extract per-player game trajectories from all PGN files in input_dir.

    Returns a dataframe with one row per game per player,
    filtered to players with at least min_games games.
    """
    files = sorted(input_dir.glob("*.pgn"))
    if not files:
        raise FileNotFoundError(f"No PGN files in {input_dir}")

    print(f"Found {len(files)} PGN files")
    print(f"Workers: {workers}")
    print(f"Min games per player: {min_games}")
    print()

    all_rows   = []
    start_time = time.time()

    for i, filepath in enumerate(files):
        file_size = filepath.stat().st_size / 1024**2
        print(f"[{i+1}/{len(files)}] {filepath.name} ({file_size:.0f} MB)",
              end=" ... ", flush=True)

        file_rows = []
        with mp.Pool(processes=workers) as pool:
            for rows in pool.imap_unordered(
                process_chunk,
                generate_chunks(filepath),
                chunksize=1,
            ):
                file_rows.extend(rows)

        all_rows.extend(file_rows)
        print(f"{len(file_rows):,} game records")

    print(f"\nTotal records: {len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    print(f"Unique players: {df['player'].nunique():,}")

    # Filter to players with enough games
    game_counts = df["player"].value_counts()
    qualified   = game_counts[game_counts >= min_games].index
    df          = df[df["player"].isin(qualified)].copy()

    print(f"Players with {min_games}+ games: {df['player'].nunique():,}")
    print(f"Total game records kept: {len(df):,}")

    # Sort chronologically within each player
    df = df.sort_values(["player", "year", "month", "date_str"])
    df = df.reset_index(drop=True)

    # Add cumulative game number per player
    df["game_num"] = df.groupby("player").cumcount()

    # Compute rolling ELO change (target for LSTM)
    df["elo_change_next10"] = df.groupby("player")["player_elo"].transform(
        lambda x: x.shift(-10) - x
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\nSaved to {output_path}")
    print(f"Total time: {elapsed/60:.1f} min")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build population data from Lichess Elite Database"
    )
    parser.add_argument("--input",     required=True,
                        help="Folder containing .pgn files")
    parser.add_argument("--output",    required=True,
                        help="Output .parquet file path")
    parser.add_argument("--min-games", type=int, default=50,
                        help="Minimum games per player (default: 50)")
    parser.add_argument("--workers",   type=int, default=8,
                        help="Parallel workers (default: 8)")
    args = parser.parse_args()

    df = build_population_data(
        input_dir   = Path(args.input),
        output_path = Path(args.output),
        min_games   = args.min_games,
        workers     = args.workers,
    )

    print(f"\nSummary:")
    print(f"  Players    : {df['player'].nunique():,}")
    print(f"  Games      : {len(df):,}")
    print(f"  ELO range  : {df['player_elo'].min()} - {df['player_elo'].max()}")
    print(f"  Date range : {df['date_str'].min()} - {df['date_str'].max()}")


if __name__ == "__main__":
    main()
