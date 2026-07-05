"""Dataset-aware losses driven by offline LLM rules."""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class DatasetAwareLoss(nn.Module):
    """MSE plus optional long-tail event penalties.

    The criterion consumes masks/features produced from rule JSON files. It
    does not call an LLM during training or inference.
    """

    def __init__(self, config: dict | Namespace | None = None, eps: float = 1e-8):
        super().__init__()
        self.config = _as_dict(config)
        self.eps = eps
        self.use_event_weighted_loss = _flag(self.config, "use_event_weighted_loss", False)
        self.use_zero_consistency_loss = _flag(self.config, "use_zero_consistency_loss", False)
        self.use_peak_shape_loss = _flag(self.config, "use_peak_shape_loss", False)
        self.use_diff_loss = _flag(self.config, "use_diff_loss", False)
        self.use_freq_loss = _flag(self.config, "use_freq_loss", False)
        self.event_weight = float(self.config.get("event_weight", 1.0))
        self.zero_weight = float(self.config.get("zero_weight", 1.0))
        self.peak_weight = float(self.config.get("peak_weight", 1.0))
        self.diff_weight = float(self.config.get("diff_weight", 1.0))
        self.freq_weight = float(self.config.get("freq_weight", 1.0))

    def forward(self, pred, true, batch_marks=None, batch_masks=None):
        """Return total loss and every component as tensors."""
        base_loss = F.mse_loss(pred, true)
        zero = pred.new_tensor(0.0)
        event_loss = zero
        zero_loss = zero
        peak_loss = zero
        diff_loss = zero
        freq_loss = zero

        masks = self._parse_masks(batch_masks, pred)
        if self.use_event_weighted_loss and masks["event"] is not None:
            event_loss = self._masked_mse(pred, true, masks["event"])
        if self.use_zero_consistency_loss and masks["zero"] is not None:
            zero_loss = self._zero_consistency(pred, masks["zero"])
        if self.use_peak_shape_loss and masks["peak"] is not None:
            peak_loss = self._peak_shape(pred, masks["peak"])
        if self.use_diff_loss and pred.shape[1] > 1:
            diff_loss = F.mse_loss(pred[:, 1:, :] - pred[:, :-1, :], true[:, 1:, :] - true[:, :-1, :])
        if self.use_freq_loss:
            fft_pred = torch.fft.rfft(pred, dim=1).abs()
            fft_true = torch.fft.rfft(true, dim=1).abs()
            freq_loss = F.mse_loss(fft_pred, fft_true)

        total = base_loss
        total = total + self.event_weight * event_loss
        total = total + self.zero_weight * zero_loss
        total = total + self.peak_weight * peak_loss
        total = total + self.diff_weight * diff_loss
        total = total + self.freq_weight * freq_loss

        return {
            "loss": total,
            "base_loss": base_loss,
            "event_loss": event_loss,
            "zero_loss": zero_loss,
            "peak_loss": peak_loss,
            "diff_loss": diff_loss,
            "freq_loss": freq_loss,
        }

    def _parse_masks(self, batch_masks, pred):
        if batch_masks is None:
            return {"event": None, "zero": None, "peak": None}
        if isinstance(batch_masks, dict):
            return {
                "event": _to_mask(batch_masks.get("event_mask"), pred),
                "zero": _to_mask(batch_masks.get("zero_mask"), pred),
                "peak": _to_mask(batch_masks.get("peak_mask"), pred),
            }

        mask = _to_mask(batch_masks, pred)
        if mask is None:
            return {"event": None, "zero": None, "peak": None}
        if mask.shape[-1] == 1:
            return {"event": mask, "zero": mask, "peak": mask}
        return {
            "event": mask[..., 0:1],
            "zero": mask[..., 1:2] if mask.shape[-1] > 1 else None,
            "peak": mask[..., 2:3] if mask.shape[-1] > 2 else None,
        }

    def _masked_mse(self, pred, true, mask):
        denom = self._mask_denominator(mask, pred.shape[-1])
        if denom.item() <= self.eps:
            return pred.new_tensor(0.0)
        return (((pred - true) ** 2) * mask).sum() / denom

    def _zero_consistency(self, pred, mask):
        denom = self._mask_denominator(mask, pred.shape[-1])
        if denom.item() <= self.eps:
            return pred.new_tensor(0.0)
        return (pred.abs() * mask).sum() / denom

    def _peak_shape(self, pred, mask):
        if pred.shape[1] < 3:
            return pred.new_tensor(0.0)
        peak_mask = mask[:, 1:-1, :]
        denom = self._mask_denominator(peak_mask, pred.shape[-1])
        if denom.item() <= self.eps:
            return pred.new_tensor(0.0)
        left = pred[:, :-2, :]
        center = pred[:, 1:-1, :]
        right = pred[:, 2:, :]
        violation = F.relu(left - center) + F.relu(right - center)
        return (violation * peak_mask).sum() / denom

    def _mask_denominator(self, mask, channels: int):
        if mask.shape[-1] == 1:
            return mask.sum() * channels + self.eps
        return mask.sum() + self.eps


def _to_mask(mask, pred):
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    return mask


def _as_dict(config) -> dict:
    if config is None:
        return {}
    if isinstance(config, Namespace):
        return vars(config)
    return dict(config)


def _flag(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)

