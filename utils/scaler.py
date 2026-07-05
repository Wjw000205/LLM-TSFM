"""Leakage-free z-score scaling utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


class StandardScaler:
    """Standardize time-series variables with statistics from train data only."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data):
        """Fit per-variable mean and std on the provided train split."""
        array = self._as_numpy(data)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        self.mean = array.mean(axis=0).astype(np.float32)
        std = array.std(axis=0).astype(np.float32)
        self.std = np.where(std < self.eps, self.eps, std).astype(np.float32)
        return self

    def transform(self, data):
        """Apply z-score normalization."""
        self._check_fitted()
        if isinstance(data, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=data.dtype, device=data.device)
            std = torch.as_tensor(self.std, dtype=data.dtype, device=data.device)
            return (data - mean) / std
        array = np.asarray(data, dtype=np.float32)
        return (array - self.mean) / self.std

    def inverse_transform(self, data, feature_indices: Iterable[int] | None = None):
        """Restore normalized values to the original scale.

        ``feature_indices`` is optional and is used when a model predicts a
        subset of the variables, for example ``features=MS``.
        """
        self._check_fitted()
        if feature_indices is None:
            mean = self.mean
            std = self.std
        else:
            indices = np.asarray(list(feature_indices), dtype=np.int64)
            mean = self.mean[indices]
            std = self.std[indices]

        if isinstance(data, torch.Tensor):
            mean_tensor = torch.as_tensor(mean, dtype=data.dtype, device=data.device)
            std_tensor = torch.as_tensor(std, dtype=data.dtype, device=data.device)
            return data * std_tensor + mean_tensor
        array = np.asarray(data, dtype=np.float32)
        return array * std + mean

    def state_dict(self) -> dict[str, np.ndarray]:
        """Return serializable scaler state."""
        self._check_fitted()
        return {"mean": self.mean.copy(), "std": self.std.copy(), "eps": self.eps}

    def load_state_dict(self, state: dict[str, np.ndarray]):
        """Load scaler state created by :meth:`state_dict`."""
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.std = np.asarray(state["std"], dtype=np.float32)
        self.eps = float(state.get("eps", self.eps))
        return self

    def _check_fitted(self):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before use.")

    @staticmethod
    def _as_numpy(data) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.asarray(data, dtype=np.float32)

