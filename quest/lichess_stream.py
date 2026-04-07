"""
lichess_stream.py

Streaming pipeline for chess2vec training data.
Downloads Lichess monthly PGN files one at a time,
extracts move token sequences, writes compact output,
deletes the source file, moves to next month.

Designed to run on Quest (Northwestern HPC) but works anywhere.

Usage:
    python3 lichess_stream.py --months 3 --output /path/to/tokens/
    python3 lichess_stream.py --start 2024-01 --end 2024-12 --output /path/
    python3 lichess_stream.py --all --output /path/  # runs for days
"""

import argparse
import chess
import chess.pgn
import gzip
import io
import json
import os
import re
import requests
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

try:
    import zstandard as zstd
except ImportError:
    print("Install zstandard: pip install zstandard")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Install tqdm: pip install tqdm")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL       = "https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
CHUNK_SIZE     = 1024 * 1024 * 4   # 4MB download chunks
MIN_ELO        = 800
MAX_ELO        = 3500
MIN_MOVES      = 5
CLOCK_REQUIRED = True              # only games with clock annotations

# Phase boundaries
OPENING_MAX_MOVE    = 15
ENDGAME_MAX_PIECES  = 12

# Eval bucket boundaries (centipawns from White perspective)
EVAL_BUCKETS = [
    (-10000, -300, "losing_badly"),
    (-300,   -100, "losing"),
    (-100,    -30, "slightly_worse"),
    (-30,      30, "equal"),
    (30,      100, "slightly_better"),
    (100,     300, "winning"),
    (300,   10000, "winning_easily"),
]


# ─────────────────────────────────────────────────────────────────────────────
# URL discovery — scrape the Lichess database page
# ─────────────────────────────────────────────────────────────────────────────

def get_available_months() -> list[tuple[int, int]]:
    """
    Scrape database.lichess.org to get all available (year, month) pairs.
    Returns newest first.
    """
    print("Fetching available months from database.lichess.org...")
    resp = requests.get("https://database.lichess.org/", timeout=30)
    resp.raise_for_status()

    pattern = r'lichess_db_standard_rated_(\d{4})-(\d{2})\.pgn\.zst'
    matches = re.findall(pattern, resp.text)
    months  = sorted(set((int(y), int(m)) for y, m in matches), reverse=True)
    print(f"Found {len(months)} available months ({months[-1][0]}-{months[-1][1]:02d} "
          f"to {months[0][0]}-{months[0][1]:02d})")
    return months


# ─────────────────────────────────────────────────────────────────────────────
# Streaming download + decompress
# ─────────────────────────────────────────────────────────────────────────────

def stream_pgn_games(year: int, month: int) -> Iterator[chess.pgn.Game]:
    """
    Stream games from a Lichess monthly file without writing to disk.

    Downloads .pgn.zst, decompresses on the fly with zstandard,
    yields python-chess Game objects one at a time.
    """
    url = BASE_URL.format(year=year, month=month)
    print(f"\nStreaming {year}-{month:02d} from {url}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total_bytes = int(resp.headers.get("Content-Length", 0))

    dctx       = zstd.ZstdDecompressor(max_window_size=2**31)
    pgn_buffer = io.TextIOWrapper(
        dctx.stream_reader(resp.raw, read_size=CHUNK_SIZE),
        encoding="utf-8",
        errors="replace",
    )

    games_seen     = 0
    games_accepted = 0
    bytes_bar = tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc=f"  Download {year}-{month:02d}",
        leave=False,
    )

    while True:
        try:
            game = chess.pgn.read_game(pgn_buffer)
            if game is None:
                break
            games_seen += 1

            # Update progress roughly every 10k games
            if games_seen % 10_000 == 0:
                bytes_bar.update(0)

            if _accept_game(game):
                games_accepted += 1
                yield game

        except Exception:
            continue

    bytes_bar.close()
    print(f"  Streamed {games_seen:,} games, accepted {games_accepted:,} "
          f"({games_accepted/max(games_seen,1)*100:.1f}%)")


def _accept_game(game: chess.pgn.Game) -> bool:
    """Filter to standard, rated, non-bot, clocked games in ELO range."""
    h = game.headers

    # Must be standard chess (no variants)
    if h.get("Variant", "Standard") != "Standard":
        return False

    # No bot games
    if h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT":
        return False

    # Must have moves
    if game.next() is None:
        return False

    # ELO filter
    try:
        w_elo = int(h.get("WhiteElo", 0))
        b_elo = int(h.get("BlackElo", 0))
        if not (MIN_ELO <= w_elo <= MAX_ELO and MIN_ELO <= b_elo <= MAX_ELO):
            return False
    except ValueError:
        return False

    # Clock filter (only games from April 2017+ have clocks)
    if CLOCK_REQUIRED:
        first_comment = game.next().comment if game.next() else ""
        if "%clk" not in first_comment:
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_game(game: chess.pgn.Game) -> Optional[list[str]]:
    """
    Convert a game into a list of enriched move tokens.

    Token format:
        {move}_{phase}_{color}_{eval_bucket}_{clock_bucket}

    Example:
        e4_opening_white_equal_plenty
        Nf3_opening_white_slightly_better_plenty
        Qxd5_middlegame_black_losing_low
    """
    tokens = []
    board  = game.board()
    node   = game
    move_number = 0

    while node.variations:
        next_node   = node.variations[0]
        move        = next_node.move
        move_number += 1

        try:
            color   = "white" if board.turn == chess.WHITE else "black"
            san     = board.san(move)
            phase   = _get_phase(board, move_number)
            eval_b  = _parse_eval(next_node.comment, board.turn == chess.WHITE)
            clock_b = _parse_clock_bucket(next_node.comment)

            token = f"{san}_{phase}_{color}_{eval_b}_{clock_b}"
            tokens.append(token)

            board.push(move)
            node = next_node

        except Exception:
            break

    if len(tokens) < MIN_MOVES:
        return None
    return tokens


def _get_phase(board: chess.Board, move_number: int) -> str:
    if move_number <= OPENING_MAX_MOVE:
        return "opening"
    pieces = sum(1 for p in board.piece_map().values()
                 if p.piece_type != chess.KING)
    return "endgame" if pieces <= ENDGAME_MAX_PIECES else "middlegame"


def _parse_eval(comment: str, is_white: bool) -> str:
    """Parse [%eval X] or [%eval #N] from PGN comment."""
    if not comment:
        return "unknown"

    # Mate annotation
    mate_match = re.search(r'\[%eval #(-?\d+)\]', comment)
    if mate_match:
        mate_in = int(mate_match.group(1))
        if is_white:
            return "winning_easily" if mate_in > 0 else "losing_badly"
        else:
            return "losing_badly" if mate_in > 0 else "winning_easily"

    # Centipawn annotation
    cp_match = re.search(r'\[%eval (-?\d+\.?\d*)\]', comment)
    if cp_match:
        cp = float(cp_match.group(1)) * 100  # convert from pawns to centipawns
        if not is_white:
            cp = -cp   # flip for Black
        for lo, hi, label in EVAL_BUCKETS:
            if lo <= cp < hi:
                return label
        return "unknown"

    return "unknown"


def _parse_clock_bucket(comment: str) -> str:
    """Bucket remaining clock time."""
    if not comment:
        return "noclock"
    match = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', comment)
    if not match:
        return "noclock"
    h, m, s = int(match.group(1)), int(match.group(2)), int(match.group(3))
    seconds = h * 3600 + m * 60 + s
    if seconds > 300:
        return "plenty"
    elif seconds > 60:
        return "moderate"
    elif seconds > 15:
        return "low"
    else:
        return "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def process_month(
    year: int,
    month: int,
    output_dir: Path,
    max_games: Optional[int] = None,
) -> dict:
    """
    Process one month of Lichess data.
    Writes one gzipped token file per month.
    Returns stats dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"tokens_{year}_{month:02d}.txt.gz"
    meta_path = output_dir / f"tokens_{year}_{month:02d}.meta.json"

    # Skip if already done
    if out_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Already processed {year}-{month:02d}: "
              f"{meta['games_written']:,} games. Skipping.")
        return meta

    stats = {
        "year": year, "month": month,
        "games_written": 0, "tokens_written": 0,
        "started": datetime.now().isoformat(),
    }

    with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
        for i, game in enumerate(stream_pgn_games(year, month)):
            if max_games and i >= max_games:
                break

            tokens = tokenize_game(game)
            if tokens is None:
                continue

            # One game per line, tokens space-separated
            out_f.write(" ".join(tokens) + "\n")
            stats["games_written"]  += 1
            stats["tokens_written"] += len(tokens)

            if stats["games_written"] % 100_000 == 0:
                print(f"    {stats['games_written']:,} games written...")

    stats["finished"] = datetime.now().isoformat()
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Done: {stats['games_written']:,} games, "
          f"{stats['tokens_written']:,} tokens, "
          f"{size_mb:.1f} MB written to {out_path.name}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lichess streaming pipeline for chess2vec")
    parser.add_argument("--output",  required=True, help="Output directory for token files")
    parser.add_argument("--months",  type=int, default=3,
                        help="Number of most recent months to process (default: 3)")
    parser.add_argument("--start",   help="Start month YYYY-MM (e.g. 2024-01)")
    parser.add_argument("--end",     help="End month YYYY-MM (e.g. 2024-12)")
    parser.add_argument("--all",     action="store_true",
                        help="Process all available months (runs for days)")
    parser.add_argument("--max-games", type=int,
                        help="Max games per month (for testing)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    available  = get_available_months()

    if args.all:
        months_to_process = available
    elif args.start and args.end:
        sy, sm = map(int, args.start.split("-"))
        ey, em = map(int, args.end.split("-"))
        months_to_process = [
            (y, m) for y, m in available
            if (sy, sm) <= (y, m) <= (ey, em)
        ]
    else:
        months_to_process = available[:args.months]

    print(f"\nWill process {len(months_to_process)} month(s):")
    for y, m in months_to_process:
        print(f"  {y}-{m:02d}")
    print()

    total_stats = {"total_games": 0, "total_tokens": 0, "months_done": 0}

    for year, month in months_to_process:
        try:
            stats = process_month(year, month, output_dir, args.max_games)
            total_stats["total_games"]  += stats["games_written"]
            total_stats["total_tokens"] += stats["tokens_written"]
            total_stats["months_done"]  += 1
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved — re-run to continue.")
            break
        except Exception as e:
            print(f"Error on {year}-{month:02d}: {e}. Skipping.")
            continue

    print(f"\nAll done.")
    print(f"  Months processed : {total_stats['months_done']}")
    print(f"  Total games      : {total_stats['total_games']:,}")
    print(f"  Total tokens     : {total_stats['total_tokens']:,}")
    print(f"  Token files in   : {output_dir}")


if __name__ == "__main__":
    main()
