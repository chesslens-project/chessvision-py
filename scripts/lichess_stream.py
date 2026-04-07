"""
lichess_stream.py

Streaming pipeline for chess2vec training data.
Downloads Lichess monthly PGN files one at a time,
extracts move token sequences, writes compact output,
deletes the source file, moves to next month.

Runs entirely locally on your machine.
Token files are small (~2MB per month) even though
source files are large (5-30GB per month).

Usage:
    python3 scripts/lichess_stream.py --months 3 --output data/tokens/
    python3 scripts/lichess_stream.py --start 2024-01 --end 2024-12 --output data/tokens/
    python3 scripts/lichess_stream.py --all --output data/tokens/
"""

import argparse
import chess
import chess.pgn
import gzip
import io
import json
import re
import requests
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

try:
    import zstandard as zstd
except ImportError:
    print("Install zstandard: pip3 install zstandard")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Install tqdm: pip3 install tqdm")
    sys.exit(1)

BASE_URL   = "https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
CHUNK_SIZE = 1024 * 1024 * 4  # 4MB

MIN_ELO        = 800
MAX_ELO        = 3500
MIN_MOVES      = 5
CLOCK_REQUIRED = False  # False = include pre-2017 games without clocks

OPENING_MAX_MOVE   = 15
ENDGAME_MAX_PIECES = 12

EVAL_BUCKETS = [
    (-10000, -300, "losing_badly"),
    (-300,   -100, "losing"),
    (-100,    -30, "slightly_worse"),
    (-30,      30, "equal"),
    (30,      100, "slightly_better"),
    (100,     300, "winning"),
    (300,   10000, "winning_easily"),
]


def get_available_months() -> list:
    print("Fetching available months from database.lichess.org...")
    resp = requests.get("https://database.lichess.org/", timeout=30)
    resp.raise_for_status()
    pattern = r'lichess_db_standard_rated_(\d{4})-(\d{2})\.pgn\.zst'
    matches = re.findall(pattern, resp.text)
    months  = sorted(set((int(y), int(m)) for y, m in matches), reverse=True)
    print(f"Found {len(months)} available months")
    return months


def stream_pgn_games(year: int, month: int) -> Iterator[chess.pgn.Game]:
    url  = BASE_URL.format(year=year, month=month)
    print(f"\nStreaming {year}-{month:02d} from {url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total_bytes = int(resp.headers.get("Content-Length", 0))
    dctx        = zstd.ZstdDecompressor(max_window_size=2**31)
    pgn_buffer  = io.TextIOWrapper(
        dctx.stream_reader(resp.raw, read_size=CHUNK_SIZE),
        encoding="utf-8",
        errors="replace",
    )

    games_seen = games_accepted = 0
    bar = tqdm(total=total_bytes, unit="B", unit_scale=True,
               desc=f"  {year}-{month:02d}", leave=False)

    while True:
        try:
            game = chess.pgn.read_game(pgn_buffer)
            if game is None:
                break
            games_seen += 1
            if games_seen % 10_000 == 0:
                bar.update(0)
            if _accept_game(game):
                games_accepted += 1
                yield game
        except Exception:
            continue

    bar.close()
    print(f"  Streamed {games_seen:,} games, accepted {games_accepted:,} "
          f"({games_accepted/max(games_seen,1)*100:.1f}%)")


def _accept_game(game: chess.pgn.Game) -> bool:
    h = game.headers
    if h.get("Variant", "Standard") != "Standard":
        return False
    if h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT":
        return False
    if game.next() is None:
        return False
    try:
        w = int(h.get("WhiteElo", 0))
        b = int(h.get("BlackElo", 0))
        if not (MIN_ELO <= w <= MAX_ELO and MIN_ELO <= b <= MAX_ELO):
            return False
    except ValueError:
        return False
    if CLOCK_REQUIRED:
        first = game.next().comment if game.next() else ""
        if "%clk" not in first:
            return False
    return True


def tokenize_game(game: chess.pgn.Game) -> Optional[list]:
    tokens = []
    board  = game.board()
    node   = game
    move_number = 0
    while node.variations:
        next_node   = node.variations[0]
        move        = next_node.move
        move_number += 1
        try:
            color  = "white" if board.turn == chess.WHITE else "black"
            san    = board.san(move)
            phase  = _get_phase(board, move_number)
            eval_b = _parse_eval(next_node.comment, board.turn == chess.WHITE)
            clk_b  = _parse_clock_bucket(next_node.comment)
            tokens.append(f"{san}_{phase}_{color}_{eval_b}_{clk_b}")
            board.push(move)
            node = next_node
        except Exception:
            break
    return tokens if len(tokens) >= MIN_MOVES else None


def _get_phase(board: chess.Board, move_number: int) -> str:
    if move_number <= OPENING_MAX_MOVE:
        return "opening"
    pieces = sum(1 for p in board.piece_map().values()
                 if p.piece_type != chess.KING)
    return "endgame" if pieces <= ENDGAME_MAX_PIECES else "middlegame"


def _parse_eval(comment: str, is_white: bool) -> str:
    if not comment:
        return "unknown"
    mate = re.search(r'\[%eval #(-?\d+)\]', comment)
    if mate:
        n = int(mate.group(1))
        return ("winning_easily" if (is_white and n > 0) or
                (not is_white and n < 0) else "losing_badly")
    cp = re.search(r'\[%eval (-?\d+\.?\d*)\]', comment)
    if cp:
        val = float(cp.group(1)) * 100
        if not is_white:
            val = -val
        for lo, hi, label in EVAL_BUCKETS:
            if lo <= val < hi:
                return label
    return "unknown"


def _parse_clock_bucket(comment: str) -> str:
    if not comment:
        return "noclock"
    m = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', comment)
    if not m:
        return "noclock"
    secs = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))
    if secs > 300:   return "plenty"
    if secs > 60:    return "moderate"
    if secs > 15:    return "low"
    return "critical"


def process_month(year: int, month: int, output_dir: Path,
                  max_games: Optional[int] = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path  = output_dir / f"tokens_{year}_{month:02d}.txt.gz"
    meta_path = output_dir / f"tokens_{year}_{month:02d}.meta.json"

    if out_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Already done {year}-{month:02d}: "
              f"{meta['games_written']:,} games. Skipping.")
        return meta

    stats = {"year": year, "month": month, "games_written": 0,
             "tokens_written": 0, "started": datetime.now().isoformat()}

    with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
        for i, game in enumerate(stream_pgn_games(year, month)):
            if max_games and i >= max_games:
                break
            tokens = tokenize_game(game)
            if tokens is None:
                continue
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
          f"{size_mb:.1f} MB")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",    required=True)
    parser.add_argument("--months",    type=int, default=3)
    parser.add_argument("--start",     help="YYYY-MM")
    parser.add_argument("--end",       help="YYYY-MM")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--max-games", type=int)
    args = parser.parse_args()

    output_dir = Path(args.output)
    available  = get_available_months()

    if args.all:
        months = available
    elif args.start and args.end:
        sy, sm = map(int, args.start.split("-"))
        ey, em = map(int, args.end.split("-"))
        months = [(y, m) for y, m in available
                  if (sy, sm) <= (y, m) <= (ey, em)]
    else:
        months = available[:args.months]

    print(f"\nWill process {len(months)} month(s)")
    total = {"games": 0, "tokens": 0, "done": 0}

    for year, month in months:
        try:
            s = process_month(year, month, output_dir, args.max_games)
            total["games"]  += s["games_written"]
            total["tokens"] += s["tokens_written"]
            total["done"]   += 1
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run to resume from where you left off.")
            break
        except Exception as e:
            print(f"Error on {year}-{month:02d}: {e}. Skipping.")

    print(f"\nDone. {total['done']} months, "
          f"{total['games']:,} games, {total['tokens']:,} tokens")


if __name__ == "__main__":
    main()
