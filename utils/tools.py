"""General experiment helpers."""

from __future__ import annotations

import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


class dotdict(dict):
    """Dictionary with attribute-style access."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bool_flag(value) -> bool:
    """Convert argparse int/bool/string flags to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def save_args(args: Namespace, path: str | Path):
    """Write experiment arguments to a readable JSON file."""
    path = Path(path)
    serializable = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool, type(None))):
            serializable[key] = value
        elif isinstance(value, (list, tuple)):
            serializable[key] = list(value)
        else:
            serializable[key] = str(value)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

