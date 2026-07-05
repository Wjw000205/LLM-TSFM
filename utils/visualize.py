"""Lightweight visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction(pred, true, save_path: str | Path, title: str = "Forecast"):
    """Save a single forecast plot for quick qualitative inspection."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(true.reshape(-1), label="true")
    plt.plot(pred.reshape(-1), label="pred")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

