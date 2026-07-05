"""Optional future-rule adapters for deterministic rule features."""

from __future__ import annotations

import torch
import torch.nn as nn


class RuleAdapter(nn.Module):
    """Apply a small future-feature correction gated by event masks."""

    def __init__(self, feature_dim: int, c_out: int, hidden_dim: int = 32):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.c_out = int(c_out)
        if self.feature_dim <= 0:
            self.mlp = None
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.c_out),
            )

    def forward(self, pred_base: torch.Tensor, future_features: torch.Tensor, future_masks: torch.Tensor):
        """Return ``pred_base + event_mask * MLP(future_features)``."""
        if self.mlp is None or future_features.shape[-1] == 0:
            return pred_base
        event_mask = _event_mask(future_masks, pred_base)
        delta = self.mlp(future_features)
        return pred_base + event_mask * delta


def apply_hard_intervention(pred: torch.Tensor, future_masks: torch.Tensor, zero_target) -> torch.Tensor:
    """Force deterministic zero-event predictions to the scaled zero target."""
    zero_mask = _zero_mask(future_masks, pred)
    if zero_mask.sum().item() == 0:
        return pred
    target = torch.as_tensor(zero_target, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
    if target.shape[-1] != pred.shape[-1]:
        raise ValueError(f"zero_target channels {target.shape[-1]} do not match predictions {pred.shape[-1]}.")
    return pred * (1.0 - zero_mask) + zero_mask * target


def _event_mask(masks: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    masks = masks.to(device=pred.device, dtype=pred.dtype)
    if masks.ndim == 4:
        return masks[:, :, 0, :]
    if masks.ndim == 3 and masks.shape[-1] >= 1:
        mask = masks[:, :, 0:1]
        return mask.expand_as(pred) if mask.shape[-1] == 1 else mask
    return torch.zeros_like(pred)


def _zero_mask(masks: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    masks = masks.to(device=pred.device, dtype=pred.dtype)
    if masks.ndim == 4:
        return masks[:, :, 1, :]
    if masks.ndim == 3 and masks.shape[-1] >= 2:
        mask = masks[:, :, 1:2]
        return mask.expand_as(pred) if mask.shape[-1] == 1 else mask
    return torch.zeros_like(pred)
