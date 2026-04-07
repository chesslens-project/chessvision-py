"""
train_chess2vec.py

Trains a Word2Vec skip-gram model on tokenized chess move sequences.
Run after lichess_stream.py has produced token files.

Usage:
    python3 scripts/train_chess2vec.py \
        --tokens data/tokens/ \
        --output data/chess2vec_model/ \
        --workers 4 \
        --vector-size 128 \
        --epochs 10
"""

import argparse
import gzip
import json
import logging
import time
from pathlib import Path

from gensim.models import Word2Vec

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


class TokenCorpus:
    def __init__(self, token_dir: Path):
        self.token_dir = token_dir
        self.files     = sorted(token_dir.glob("tokens_*.txt.gz"))
        if not self.files:
            raise FileNotFoundError(f"No token files in {token_dir}")
        print(f"Corpus: {len(self.files)} file(s)")

    def __iter__(self):
        for gz_file in self.files:
            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                for line in f:
                    tokens = line.strip().split()
                    if tokens:
                        yield tokens

    def count_games(self) -> int:
        total = 0
        for gz_file in self.files:
            meta = gz_file.parent / gz_file.name.replace(".txt.gz", ".meta.json")
            if meta.exists():
                with open(meta) as f:
                    total += json.load(f).get("games_written", 0)
        return total


def train(token_dir: Path, output_dir: Path, vector_size: int = 128,
          window: int = 5, min_count: int = 20, workers: int = 4,
          epochs: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus  = TokenCorpus(token_dir)
    n_games = corpus.count_games()

    print(f"\nTraining chess2vec:")
    print(f"  Games       : {n_games:,}")
    print(f"  Vector size : {vector_size}")
    print(f"  Window      : {window}")
    print(f"  Workers     : {workers}")
    print(f"  Epochs      : {epochs}")
    print()

    model = Word2Vec(
        vector_size = vector_size,
        window      = window,
        min_count   = min_count,
        workers     = workers,
        sg          = 1,
        negative    = 10,
        epochs      = epochs,
        compute_loss= True,
    )

    print("Building vocabulary...")
    model.build_vocab(corpus)
    print(f"Vocabulary: {len(model.wv):,} unique tokens")

    print("Training...")
    start = time.time()
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed/60:.1f} minutes")

    model_path = output_dir / "chess2vec.model"
    wv_path    = output_dir / "chess2vec.wordvectors"
    meta_path  = output_dir / "chess2vec.meta.json"

    model.save(str(model_path))
    model.wv.save(str(wv_path))

    meta = {
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "workers": workers,
        "epochs": epochs,
        "vocab_size": len(model.wv),
        "n_games": n_games,
        "training_minutes": round(elapsed/60, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print(f"  {model_path.name}")
    print(f"  {wv_path.name}")
    print(f"  {meta_path.name}")

    print("\nValidation — similar to 'e4_opening_white_equal_plenty':")
    try:
        token = "e4_opening_white_equal_plenty"
        if token in model.wv:
            for word, score in model.wv.most_similar(token, topn=5):
                print(f"  {word:<45} {score:.3f}")
        else:
            print(f"  Token not in vocabulary (need more data)")
    except Exception as e:
        print(f"  Validation error: {e}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens",      required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--window",      type=int, default=5)
    parser.add_argument("--min-count",   type=int, default=20)
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--epochs",      type=int, default=10)
    args = parser.parse_args()

    train(
        token_dir   = Path(args.tokens),
        output_dir  = Path(args.output),
        vector_size = args.vector_size,
        window      = args.window,
        min_count   = args.min_count,
        workers     = args.workers,
        epochs      = args.epochs,
    )


if __name__ == "__main__":
    main()
