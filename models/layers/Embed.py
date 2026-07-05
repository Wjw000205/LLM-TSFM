"""Minimal embeddings used by future backbones."""

from __future__ import annotations

import torch
import torch.nn as nn


class DataEmbedding(nn.Module):
    """Project raw values and time marks into a shared model dimension."""

    def __init__(self, c_in: int, d_model: int, mark_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.value_projection = nn.Linear(c_in, d_model)
        self.mark_projection = nn.Linear(mark_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None):
        value = self.value_projection(x)
        if x_mark is not None and x_mark.shape[-1] == self.mark_projection.in_features:
            value = value + self.mark_projection(x_mark)
        return self.dropout(value)

