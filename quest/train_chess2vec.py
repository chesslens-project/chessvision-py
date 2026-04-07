"""
train_chess2vec.py

Trains a Word2Vec skip-gram model on tokenized chess move sequences.
Designed to run after lichess_stream.py has produced token files.

Usage:
    python3 train_chess2vec.py \
        --tokens /path/to/tokens/ \
        --output /path/to/model/ \
        --workers 16 \
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
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Token corpus reader
# ─────────────────────────────────────────────────────────────────────────────

class TokenCorpus:
    """
    Lazy reader over all .txt.gz token files.
    Streams line by line — never loads everything into memory.
    """
    def __init__(self, token_dir: Path):
        self.token_dir = token_dir
        self.files     = sorted(token_dir.glob("tokens_*.txt.gz"))
        if not self.files:
            raise FileNotFoundError(f"No token files found in {token_dir}")
        print(f"Corpus: {len(self.files)} token file(s)")

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


# ─────────────────────────────────────────────────────────────────────────────
# Training callback
# ─────────────────────────────────────────────────────────────────────────────

class EpochLogger(CallbackAny2Vec):
    def __init__(self, total_epochs: int):
        self.epoch = 0
        self.total = total_epochs
        self.start = time.time()

    def on_epoch_begin(self, model):
        self.epoch += 1
        print(f"  Epoch {self.epoch}/{self.total} started...")

    def on_epoch_end(self, model):
        elapsed = time.time() - self.start
        print(f"  Epoch {self.epoch}/{self.total} done. "
              f"Loss: {model.get_latest_training_loss():.1f}  "
              f"Elapsed: {elapsed/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    token_dir: Path,
    output_dir: Path,
    vector_size: int = 128,
    window: int = 5,
    min_count: int = 20,
    workers: int = 8,
    epochs: int = 10,
    sg: int = 1,       # 1 = skip-gram, 0 = CBOW
    negative: int = 10,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = TokenCorpus(token_dir)

    n_games = corpus.count_games()
    print(f"\nTraining chess2vec:")
    print(f"  Games          : {n_games:,}")
    print(f"  Vector size    : {vector_size}")
    print(f"  Window         : {window}")
    print(f"  Min count      : {min_count}")
    print(f"  Workers        : {workers}")
    print(f"  Epochs         : {epochs}")
    print(f"  Architecture   : {'skip-gram' if sg else 'CBOW'}")
    print()

    print("Building vocabulary...")
    model = Word2Vec(
        vector_size = vector_size,
        window      = window,
        min_count   = min_count,
        workers     = workers,
        sg          = sg,
        negative    = negative,
        epochs      = epochs,
        compute_loss= True,
        callbacks   = [EpochLogger(epochs)],
    )
    model.build_vocab(corpus)
    print(f"Vocabulary size: {len(model.wv):,} unique move tokens")
    print()

    print("Training...")
    model.train(
        corpus,
        total_examples = model.corpus_count,
        epochs         = model.epochs,
    )

    # Save everything
    model_path = output_dir / "chess2vec.model"
    wv_path    = output_dir / "chess2vec.wordvectors"
    meta_path  = output_dir / "chess2vec.meta.json"

    model.save(str(model_path))
    model.wv.save(str(wv_path))

    meta = {
        "vector_size"  : vector_size,
        "window"       : window,
        "min_count"    : min_count,
        "workers"      : workers,
        "epochs"       : epochs,
        "sg"           : sg,
        "vocab_size"   : len(model.wv),
        "n_games"      : n_games,
        "token_files"  : [str(f) for f in corpus.files],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved:")
    print(f"  {model_path}")
    print(f"  {wv_path}")
    print(f"  {meta_path}")

    # Quick validation
    _validate(model)
    return model


def _validate(model: Word2Vec):
    """Quick sanity check — do similar moves cluster together?"""
    print("\nValidation — most similar tokens to 'e4_opening_white_equal_plenty':")
    try:
        token = "e4_opening_white_equal_plenty"
        if token in model.wv:
            similar = model.wv.most_similar(token, topn=5)
            for word, score in similar:
                print(f"  {word:<45} {score:.3f}")
        else:
            print(f"  Token '{token}' not in vocabulary.")
    except Exception as e:
        print(f"  Validation error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train chess2vec Word2Vec model")
    parser.add_argument("--tokens",      required=True, help="Directory of token .txt.gz files")
    parser.add_argument("--output",      required=True, help="Output directory for model")
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--window",      type=int, default=5)
    parser.add_argument("--min-count",   type=int, default=20)
    parser.add_argument("--workers",     type=int, default=8)
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
