"""Base loss functions."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return F.mse_loss(pred, true)


def mae_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return F.l1_loss(pred, true)

