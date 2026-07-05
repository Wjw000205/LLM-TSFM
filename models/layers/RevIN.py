"""Reversible instance normalization for time-series inputs."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Normalize each sample and variable, then optionally denormalize output."""

    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self._cached_mean = None
        self._cached_stdev = None
        self._cached_last = None

    def forward(self, x, mode, feature_indices: Iterable[int] | None = None):
        """Run ``mode='norm'`` or ``mode='denorm'`` on ``[B, L, C]`` tensors."""
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x, feature_indices=feature_indices)
        raise ValueError("RevIN mode must be 'norm' or 'denorm'.")

    def _normalize(self, x):
        if x.ndim != 3:
            raise ValueError("RevIN expects input shaped [B, L, C].")
        if self.subtract_last:
            self._cached_last = x[:, -1:, :].detach()
            centered = x - self._cached_last
            self._cached_mean = None
        else:
            self._cached_mean = x.mean(dim=1, keepdim=True).detach()
            centered = x - self._cached_mean
            self._cached_last = None
        self._cached_stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = centered / self._cached_stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x, feature_indices: Iterable[int] | None = None):
        if self._cached_stdev is None:
            raise RuntimeError("RevIN denorm called before norm.")
        indices = None if feature_indices is None else torch.as_tensor(list(feature_indices), device=x.device, dtype=torch.long)
        stdev = self._select_cached(self._cached_stdev, indices)
        mean = self._select_cached(self._cached_mean, indices)
        last = self._select_cached(self._cached_last, indices)
        if self.affine:
            weight = self.affine_weight if indices is None else self.affine_weight.index_select(0, indices)
            bias = self.affine_bias if indices is None else self.affine_bias.index_select(0, indices)
            x = (x - bias) / (weight + self.eps * self.eps)
        x = x * stdev
        if self.subtract_last:
            x = x + last
        else:
            x = x + mean
        return x

    @staticmethod
    def _select_cached(value, indices):
        if value is None:
            return None
        if indices is None:
            return value
        return value.index_select(-1, indices)

