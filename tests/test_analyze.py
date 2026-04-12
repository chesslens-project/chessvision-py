import pytest
import pandas as pd
import tempfile
from pathlib import Path
from chessvision.models import (
    get_model_path,
    is_downloaded,
    list_models,
    MODELS,
    CACHE_DIR,
)
from chessvision.analyze import _detect_player


def test_model_registry_has_required_keys():
    for name, info in MODELS.items():
        assert "files" in info
        assert "description" in info
        assert "size_mb" in info
        assert len(info["files"]) > 0


def test_get_model_path_returns_path():
    path = get_model_path("chess2vec")
    assert isinstance(path, Path)
    assert "chess2vec" in str(path)


def test_is_downloaded_returns_bool():
    result = is_downloaded("chess2vec")
    assert isinstance(result, bool)


def test_list_models_runs():
    list_models()


def test_detect_player_finds_most_common():
    games_df = pd.DataFrame({
        "white": ["Alice", "Alice", "Bob", "Alice"],
        "black": ["Bob",   "Carol", "Alice", "Bob"],
    })
    player = _detect_player(games_df)
    assert player == "Alice"


def test_cache_dir_is_in_home():
    assert str(Path.home()) in str(CACHE_DIR)
