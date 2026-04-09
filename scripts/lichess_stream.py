"""
lichess_stream.py

Fast multiprocessing tokenizer for local Lichess .pgn.zst files.
Uses all available CPU cores to parse games in parallel.

Usage:
    python3 scripts/lichess_stream.py \
        --input  /path/to/lichess_dataset/ \
        --output /path/to/tokens/ \
        --workers 10
"""

import argparse
import chess
import chess.pgn
import gzip
import io
import json
import multiprocessing as mp
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import zstandard as zstd
except ImportError:
    print("Install zstandard: pip3 install zstandard")
    sys.exit(1)

MIN_ELO            = 800
MAX_ELO            = 3500
MIN_MOVES          = 5
OPENING_MAX_MOVE   = 15
ENDGAME_MAX_PIECES = 12
CHUNK_SIZE         = 1024 * 1024 * 32

EVAL_BUCKETS = [
    (-10000, -300, "losing_badly"),
    (-300,   -100, "losing"),
    (-100,    -30, "slightly_worse"),
    (-30,      30, "equal"),
    (30,      100, "slightly_better"),
    (100,     300, "winning"),
    (300,   10000, "winning_easily"),
]


def find_pgn_zst_files(input_path: Path) -> list:
    if input_path.is_file():
        if input_path.suffix == ".zst":
            return [input_path]
        raise ValueError(f"Not a .pgn.zst file: {input_path}")
    elif input_path.is_dir():
        files = sorted(input_path.glob("*.pgn.zst"))
        if not files:
            raise FileNotFoundError(f"No .pgn.zst files found in {input_path}")
        return files
    raise FileNotFoundError(f"Path does not exist: {input_path}")


def extract_year_month(filepath: Path) -> tuple:
    match = re.search(r"(\d{4})-(\d{2})\.pgn\.zst", filepath.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


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
    return True


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


def tokenize_game(game: chess.pgn.Game) -> Optional[list]:
    tokens      = []
    board       = game.board()
    node        = game
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


def _process_chunk(args):
    chunk_text, chunk_id = args
    lines      = []
    game_texts = re.split(r'\n\n(?=\[Event )', chunk_text)
    for game_text in game_texts:
        if not game_text.strip() or "[Event " not in game_text:
            continue
        try:
            game = chess.pgn.read_game(io.StringIO(game_text))
            if game is None or not _accept_game(game):
                continue
            tokens = tokenize_game(game)
            if tokens:
                lines.append(" ".join(tokens))
        except Exception:
            continue
    return chunk_id, lines


def process_file(
    filepath: Path,
    output_dir: Path,
    workers: int = 10,
    max_games: Optional[int] = None,
) -> dict:
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

    file_size = filepath.stat().st_size
    print(f"Processing: {filepath.name} "
          f"({file_size/1024**3:.1f} GB) | {workers} workers")

    start_time     = time.time()
    bytes_read     = 0
    games_written  = 0
    tokens_written = 0
    chunk_id       = 0
    text_buffer    = ""
    pending        = []

    with gzip.open(out_path, "wt", encoding="utf-8") as out_f:
        pool = mp.Pool(processes=workers)
        try:
            with open(filepath, "rb") as raw_file:
                dctx = zstd.ZstdDecompressor(max_window_size=2**31)
                with dctx.stream_reader(raw_file,
                                        read_size=CHUNK_SIZE) as reader:
                    while True:
                        raw = reader.read(CHUNK_SIZE)
                        if not raw:
                            break

                        bytes_read  += len(raw)
                        text_buffer += raw.decode("utf-8", errors="replace")
                        boundary     = text_buffer.rfind("\n\n[Event ")
                        if boundary == -1:
                            continue

                        complete    = text_buffer[:boundary]
                        text_buffer = text_buffer[boundary:]

                        if complete.strip():
                            r = pool.apply_async(
                                _process_chunk, [(complete, chunk_id)]
                            )
                            pending.append((chunk_id, r))
                            chunk_id += 1

                        # Collect any ready results immediately
                        still_pending = []
                        for cid, res in pending:
                            if res.ready():
                                try:
                                    _, lines = res.get(timeout=5)
                                    for line in lines:
                                        out_f.write(line + "\n")
                                        games_written  += 1
                                        tokens_written += len(line.split())
                                except Exception:
                                    pass
                            else:
                                still_pending.append((cid, res))
                        pending = still_pending

                        elapsed = time.time() - start_time
                        speed   = bytes_read / max(elapsed, 1) / 1024**2
                        pct     = bytes_read / file_size * 100
                        eta_h   = ((file_size - bytes_read)
                                   / max(bytes_read / max(elapsed, 1), 1)
                                   ) / 3600
                        print(
                            f"  {pct:5.1f}% | "
                            f"{speed:5.1f} MB/s | "
                            f"ETA {eta_h:.1f}h | "
                            f"{games_written:,} games",
                            end="\r", flush=True
                        )

                        if max_games and games_written >= max_games:
                            raise StopIteration

                # Drain remaining
                if text_buffer.strip():
                    r = pool.apply_async(
                        _process_chunk, [(text_buffer, chunk_id)]
                    )
                    pending.append((chunk_id, r))

                for cid, res in pending:
                    try:
                        _, lines = res.get(timeout=120)
                        for line in lines:
                            out_f.write(line + "\n")
                            games_written  += 1
                            tokens_written += len(line.split())
                    except Exception:
                        continue

        except StopIteration:
            pass
        finally:
            pool.close()
            pool.join()

    elapsed   = time.time() - start_time
    avg_speed = file_size / max(elapsed, 1) / 1024**2
    print(f"\n  Done: {games_written:,} games | "
          f"{tokens_written:,} tokens | "
          f"{avg_speed:.1f} MB/s avg | "
          f"{elapsed/3600:.1f} hrs")

    stats = {
        "source_file":    filepath.name,
        "year":           year,
        "month":          month,
        "games_written":  games_written,
        "tokens_written": tokens_written,
        "started":        datetime.now().isoformat(),
        "finished":       datetime.now().isoformat(),
        "workers":        workers,
        "avg_speed_mbs":  round(avg_speed, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--workers",   type=int, default=10)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--append",    action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    files      = find_pgn_zst_files(input_path)

    existing = list(output_dir.glob("*.txt.gz")) if output_dir.exists() else []
    if existing:
        print(f"Existing token files: {len(existing)} (will be skipped)")

    print(f"Found {len(files)} file(s) to process:")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size/1024**3:.1f} GB)")
    print(f"Workers: {args.workers}")
    print()

    total = {"games": 0, "tokens": 0, "files": 0}
    for filepath in files:
        stats = process_file(
            filepath, output_dir,
            workers   = args.workers,
            max_games = args.max_games,
        )
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
