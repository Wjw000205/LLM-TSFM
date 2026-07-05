"""Early stopping helper."""

from __future__ import annotations

from pathlib import Path

import torch


class EarlyStopping:
    """Stop training after validation loss stops improving."""

    def __init__(self, patience: int = 7, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model: torch.nn.Module, path: str | Path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, path)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self._save_checkpoint(val_loss, model, path)
        self.counter = 0

    def _save_checkpoint(self, val_loss: float, model: torch.nn.Module, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path / "checkpoint.pth")
        self.val_loss_min = val_loss

