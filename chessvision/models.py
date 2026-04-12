"""
models.py

Model registry and auto-download system.
Pre-trained models are hosted on Hugging Face Hub.
Downloaded once, cached locally at ~/.chessvision/models/
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

CACHE_DIR     = Path.home() / ".chessvision" / "models"
MODEL_VERSION = "v0.1.0"

# Models hosted on Hugging Face
HF_REPO = "rakkshet/chessvision-models"
MODELS  = {
    "chess2vec": {
    "files":       ["chess2vec.wordvectors",
                    "chess2vec.meta.json"],
    "description": "Style embedding model trained on 29.3M elite games",
    "size_mb":     180,
    },
    "population_lstm": {
        "files":       ["population_lstm.pt",
                        "population_scaler.joblib",
                        "population_config.json",
                        "population_history.json"],
        "description": "ELO trajectory model trained on 67,115 elite players",
        "size_mb":     5,
    },
    "error_archetypes": {
        "files":       ["hdbscan_model.joblib",
                        "umap_reducer.joblib",
                        "archetype_map.json"],
        "description": "Error archetype clustering model",
        "size_mb":     50,
    },
}


def get_model_path(model_name: str) -> Path:
    """Return local cache path for a model."""
    return CACHE_DIR / MODEL_VERSION / model_name


def is_downloaded(model_name: str) -> bool:
    """Check if a model is already cached locally."""
    model_dir = get_model_path(model_name)
    if not model_dir.exists():
        return False
    expected = MODELS.get(model_name, {}).get("files", [])
    return all((model_dir / f).exists() for f in expected)


def list_models() -> None:
    """Print available models and their download status."""
    print(f"\nChessVision Models (cache: {CACHE_DIR})")
    print("-" * 60)
    for name, info in MODELS.items():
        status = "downloaded" if is_downloaded(name) else "not downloaded"
        print(f"  {name:<20} {info['size_mb']:>4} MB  [{status}]")
        print(f"    {info['description']}")
    print()


def download_models(
    models: Optional[list] = None,
    force: bool = False,
    local_dir: Optional[Path] = None,
) -> dict:
    """
    Download pre-trained chessvision models.

    Parameters
    ----------
    models   : list of model names to download (default: all)
               options: 'chess2vec', 'population_lstm', 'error_archetypes'
    force    : re-download even if already cached
    local_dir: if set, copy models from this local directory instead
               of downloading (useful for development)

    Returns
    -------
    dict mapping model_name -> local path
    """
    if models is None:
        models = list(MODELS.keys())

    paths = {}

    for model_name in models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}. "
                  f"Available: {list(MODELS.keys())}")
            continue

        model_dir = get_model_path(model_name)

        if is_downloaded(model_name) and not force:
            print(f"  {model_name}: already cached at {model_dir}")
            paths[model_name] = model_dir
            continue

        model_dir.mkdir(parents=True, exist_ok=True)

        if local_dir:
            # Copy from local directory (development mode)
            src = Path(local_dir) / model_name
            if src.exists():
                for f in MODELS[model_name]["files"]:
                    src_f = src / f
                    if src_f.exists():
                        shutil.copy2(src_f, model_dir / f)
                print(f"  {model_name}: copied from {src}")
                paths[model_name] = model_dir
            else:
                print(f"  {model_name}: source not found at {src}")
        else:
            # Download from Hugging Face
            try:
                _download_from_hf(model_name, model_dir)
                paths[model_name] = model_dir
            except Exception as e:
                print(f"  {model_name}: download failed — {e}")
                print(f"    To use local models, run: "
                      f"chessvision.download_models(local_dir='/path/to/models')")

    return paths


def _download_from_hf(model_name: str, model_dir: Path) -> None:
    """Download model files from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "Install huggingface_hub: pip install huggingface-hub"
        )

    info  = MODELS[model_name]
    total = info["size_mb"]
    print(f"  Downloading {model_name} (~{total}MB)...")

    for filename in info["files"]:
        hf_hub_download(
            repo_id   = HF_REPO,
            filename  = f"{MODEL_VERSION}/{model_name}/{filename}",
            local_dir = str(model_dir.parent.parent),
        )
    print(f"  {model_name}: saved to {model_dir}")


def register_local_models(
    chess2vec_dir:    Optional[Path] = None,
    population_dir:   Optional[Path] = None,
    archetypes_dir:   Optional[Path] = None,
) -> None:
    """
    Register locally trained models for use with chessvision.
    Use this after training your own models with the chessvision scripts.

    Parameters
    ----------
    chess2vec_dir  : directory containing chess2vec.wordvectors
    population_dir : directory containing population_lstm.pt
    archetypes_dir : directory containing hdbscan_model.joblib
    """
    mapping = {
        "chess2vec":       chess2vec_dir,
        "population_lstm": population_dir,
        "error_archetypes": archetypes_dir,
    }

    for model_name, src_dir in mapping.items():
        if src_dir is None:
            continue
        src_dir  = Path(src_dir)
        dest_dir = get_model_path(model_name)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for f in MODELS[model_name]["files"]:
            src_f = src_dir / f
            if src_f.exists():
                shutil.copy2(src_f, dest_dir / f)

        print(f"  Registered {model_name} from {src_dir}")


def get_cached_path(model_name: str, filename: str) -> Optional[Path]:
    """Return the path to a specific cached model file, or None."""
    path = get_model_path(model_name) / filename
    return path if path.exists() else None
