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
        self.use_nonevent_preservation_loss = _flag(self.config, "use_nonevent_preservation_loss", False)
        self.use_baseline_distillation = _flag(self.config, "use_baseline_distillation", False)
        self.event_weight = float(self.config.get("event_weight", 1.0))
        self.zero_weight = float(self.config.get("zero_weight", 1.0))
        self.peak_weight = float(self.config.get("peak_weight", 1.0))
        self.diff_weight = float(self.config.get("diff_weight", 1.0))
        self.freq_weight = float(self.config.get("freq_weight", 1.0))
        self.nonevent_weight = float(self.config.get("nonevent_weight", 1.0))
        self.distill_weight = float(self.config.get("distill_weight", 1.0))
        self.peak_window_size = max(1, int(self.config.get("peak_window_size", 1)))
        zero_target = self.config.get("zero_target", None)
        if zero_target is None:
            self.zero_target = None
        else:
            self.register_buffer("zero_target", torch.as_tensor(zero_target, dtype=torch.float32).view(1, 1, -1))

    def forward(self, pred, true, batch_marks=None, batch_masks=None, baseline_pred=None):
        """Return total loss and every component as tensors."""
        base_loss = F.mse_loss(pred, true)
        zero = pred.new_tensor(0.0)
        event_loss = zero
        zero_loss = zero
        peak_loss = zero
        diff_loss = zero
        freq_loss = zero
        nonevent_loss = zero
        distill_loss = zero

        masks = self._parse_masks(batch_masks, pred)
        if self.use_event_weighted_loss and masks["event"] is not None:
            event_loss = self._masked_mse(pred, true, masks["event"])
        if self.use_nonevent_preservation_loss and masks["event"] is not None:
            nonevent_loss = self._masked_mse(pred, true, self._non_event_mask(masks["event"]))
        if self.use_baseline_distillation and baseline_pred is not None and masks["event"] is not None:
            distill_loss = self._masked_mse(pred, baseline_pred.detach(), self._non_event_mask(masks["event"]))
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
        total = total + self.nonevent_weight * nonevent_loss
        total = total + self.distill_weight * distill_loss

        return {
            "loss": total,
            "base_loss": base_loss,
            "event_loss": event_loss,
            "zero_loss": zero_loss,
            "peak_loss": peak_loss,
            "diff_loss": diff_loss,
            "freq_loss": freq_loss,
            "nonevent_loss": nonevent_loss,
            "distill_loss": distill_loss,
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
        if mask.ndim == 4:
            if mask.shape[2] != 3:
                raise ValueError("4D batch_masks must be shaped [B, L, 3, C].")
            return {
                "event": mask[:, :, 0, :],
                "zero": mask[:, :, 1, :],
                "peak": mask[:, :, 2, :],
            }
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

    def _non_event_mask(self, event_mask):
        return torch.clamp(1.0 - event_mask, min=0.0, max=1.0)

    def _zero_consistency(self, pred, mask):
        denom = self._mask_denominator(mask, pred.shape[-1])
        if denom.item() <= self.eps:
            return pred.new_tensor(0.0)
        zero_target = self._zero_target(pred)
        return ((pred - zero_target).abs() * mask).sum() / denom

    def _peak_shape(self, pred, mask):
        if pred.shape[1] < 2:
            return pred.new_tensor(0.0)
        denom = self._mask_denominator(mask, pred.shape[-1])
        if denom.item() <= self.eps:
            return pred.new_tensor(0.0)
        losses = []
        length = pred.shape[1]
        for step in range(length):
            start = max(0, step - self.peak_window_size)
            end = min(length, step + self.peak_window_size + 1)
            context_parts = []
            if start < step:
                context_parts.append(pred[:, start:step, :])
            if step + 1 < end:
                context_parts.append(pred[:, step + 1 : end, :])
            if not context_parts:
                continue
            context = torch.cat(context_parts, dim=1).mean(dim=1)
            peak = pred[:, step, :]
            losses.append(F.relu(context - peak) * mask[:, step, :])
        if not losses:
            return pred.new_tensor(0.0)
        return torch.stack(losses, dim=1).sum() / denom

    def _mask_denominator(self, mask, channels: int):
        if mask.shape[-1] == 1:
            return mask.sum() * channels + self.eps
        return mask.sum() + self.eps

    def _zero_target(self, pred):
        if self.zero_target is None:
            return pred.new_zeros(1, 1, pred.shape[-1])
        zero_target = self.zero_target.to(device=pred.device, dtype=pred.dtype)
        if zero_target.shape[-1] != pred.shape[-1]:
            raise ValueError(
                f"zero_target channel count {zero_target.shape[-1]} does not match prediction channels {pred.shape[-1]}."
            )
        return zero_target


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
