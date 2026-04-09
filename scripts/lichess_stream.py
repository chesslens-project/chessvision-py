"""
lichess_stream.py

Tokenization pipeline for chess2vec training data.
Reads local .pgn.zst files and extracts move token sequences.

Download files manually from https://database.lichess.org/
then run this script to tokenize them.

Usage:
    # Tokenize all .pgn.zst files in a folder
    python3 scripts/lichess_stream.py \
        --input  /path/to/lichess_dataset/ \
        --output /path/to/tokens/

    # Tokenize a single file
    python3 scripts/lichess_stream.py \
        --input  /path/to/lichess_db_standard_rated_2026-03.pgn.zst \
        --output /path/to/tokens/
"""

import argparse
import chess
import chess.pgn
import gzip
import io
import json
import re
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


MIN_ELO        = 800
MAX_ELO        = 3500
MIN_MOVES      = 5
CLOCK_REQUIRED = False

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


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_pgn_zst_files(input_path: Path) -> list:
    """Find all .pgn.zst files in a path (file or directory)."""
    if input_path.is_file():
        if input_path.suffix == ".zst":
            return [input_path]
        else:
            raise ValueError(f"Not a .pgn.zst file: {input_path}")
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.pgn.zst"))
        if not files:
            raise FileNotFoundError(
                f"No .pgn.zst files found in {input_path}\n"
                f"Download from https://database.lichess.org/"
            )
        return files
    else:
        raise FileNotFoundError(f"Path does not exist: {input_path}")


def extract_year_month(filepath: Path) -> tuple:
    """Extract year and month from Lichess filename."""
    match = re.search(r"(\d{4})-(\d{2})\.pgn\.zst", filepath.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# Local file streaming
# ─────────────────────────────────────────────────────────────────────────────

def stream_local_file(filepath: Path) -> Iterator[chess.pgn.Game]:
    """
    Stream games from a local .pgn.zst file.
    No network required — reads directly from disk.
    """
    file_size   = filepath.stat().st_size
    games_seen  = 0
    games_accepted = 0

    print(f"  Reading {filepath.name} ({file_size/1024/1024/1024:.2f} GB)")

    with open(filepath, "rb") as raw_file:
        dctx       = zstd.ZstdDecompressor(max_window_size=2**31)
        pgn_buffer = io.TextIOWrapper(
            dctx.stream_reader(raw_file, read_size=1024*1024*4),
            encoding="utf-8",
            errors="replace",
        )

        bar = tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"  Processing",
            leave=True,
        )

        while True:
            try:
                game = chess.pgn.read_game(pgn_buffer)
                if game is None:
                    break
                games_seen += 1

                if games_seen % 10_000 == 0:
                    try:
                        pos = raw_file.tell()
                        bar.update(pos - bar.n)
                    except Exception:
                        pass

                if _accept_game(game):
                    games_accepted += 1
                    yield game

            except Exception:
                continue

        bar.close()

    print(f"  Done: {games_seen:,} streamed, "
          f"{games_accepted:,} accepted "
          f"({games_accepted/max(games_seen,1)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def _accept_game(game: chess.pgn.Game) -> bool:
    """Filter to standard, rated, non-bot games in ELO range."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_game(game: chess.pgn.Game) -> Optional[list]:
    """Convert a game into enriched move tokens."""
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
        return ("winning_easily"
                if (is_white and n > 0) or (not is_white and n < 0)
                else "losing_badly")
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
    if secs > 300:  return "plenty"
    if secs > 60:   return "moderate"
    if secs > 15:   return "low"
    return "critical"


# ─────────────────────────────────────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    filepath: Path,
    output_dir: Path,
    max_games: Optional[int] = None,
) -> dict:
    """
    Tokenize one local .pgn.zst file.
    Skips if output already exists (resume support).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    year, month = extract_year_month(filepath)

    if year and month:
        out_name  = f"tokens_{year}_{month:02d}.txt.gz"
        meta_name = f"tokens_{year}_{month:02d}.meta.json"
    else:
        stem      = filepath.stem.replace(".pgn", "")
        out_name  = f"tokens_{stem}.txt.gz"
        meta_name = f"tokens_{stem}.meta.json"

    out_path  = output_dir / out_name
    meta_path = output_dir / meta_name

    if out_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Already done: {out_path.name} "
              f"({meta['games_written']:,} games). Skipping.")
        return meta

    stats = {
        "source_file":    filepath.name,
        "year":           year,
        "month":          month,
        "games_written":  0,
        "tokens_written": 0,
        "started":        datetime.now().isoformat(),
    }

    with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
        for i, game in enumerate(stream_local_file(filepath)):
            if max_games and i >= max_games:
                break
            tokens = tokenize_game(game)
            if tokens is None:
                continue
            out_f.write(" ".join(tokens) + "\n")
            stats["games_written"]  += 1
            stats["tokens_written"] += len(tokens)
            if stats["games_written"] % 500_000 == 0:
                print(f"    {stats['games_written']:,} games tokenized...")

    stats["finished"] = datetime.now().isoformat()
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {stats['games_written']:,} games, "
          f"{stats['tokens_written']:,} tokens, "
          f"{size_mb:.1f} MB → {out_path.name}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tokenize local Lichess .pgn.zst files for chess2vec"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a .pgn.zst file or folder containing multiple files"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for token .txt.gz files"
    )
    parser.add_argument(
        "--max-games", type=int,
        help="Max games per file (for testing)"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Add new files to existing token collection. Already tokenized files are skipped automatically."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    files      = find_pgn_zst_files(input_path)

    existing = list(output_dir.glob("*.txt.gz")) if output_dir.exists() else []
    if existing:
        print(f"Existing token files: {len(existing)} (will be skipped)")

    print(f"Found {len(files)} file(s) to process:")
    for f in files:
        size_gb = f.stat().st_size / 1024**3
        print(f"  {f.name} ({size_gb:.1f} GB)")
    print()

    total = {"games": 0, "tokens": 0, "files": 0}

    for filepath in files:
        print(f"\nProcessing: {filepath.name}")
        stats = process_file(filepath, output_dir, args.max_games)
        total["games"]  += stats["games_written"]
        total["tokens"] += stats["tokens_written"]
        total["files"]  += 1

    print(f"\nAll done.")
    print(f"  Files processed : {total['files']}")
    print(f"  Total games     : {total['games']:,}")
    print(f"  Total tokens    : {total['tokens']:,}")
    print(f"  Token files in  : {output_dir}")


if __name__ == "__main__":
    main()