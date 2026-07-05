"""Executable LLM rule-prior fusion branch."""

from __future__ import annotations

import torch
import torch.nn as nn


class RulePriorFusion(nn.Module):
    """Softly fuse compiled rule priors into forecasts at rule-triggered timestamps."""

    def __init__(
        self,
        alpha: float = 0.5,
        use_confidence: bool = False,
        rule_prior_types: str | list[str] | tuple[str, ...] = "zero_event",
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.use_confidence = bool(use_confidence)
        if isinstance(rule_prior_types, str):
            self.rule_prior_types = {item.strip() for item in rule_prior_types.split(",") if item.strip()}
        else:
            self.rule_prior_types = {str(item).strip() for item in rule_prior_types if str(item).strip()}

    def forward(self, pred_base: torch.Tensor, future_masks: torch.Tensor, zero_target, future_features=None, rule_config=None):
        """Return soft-fused predictions; currently supports compiled zero-event priors."""
        if "zero_event" not in self.rule_prior_types:
            return pred_base
        zero_mask = _zero_mask(future_masks, pred_base)
        if zero_mask.sum().item() == 0 or self.alpha == 0.0:
            return pred_base
        target = torch.as_tensor(zero_target, dtype=pred_base.dtype, device=pred_base.device).view(1, 1, -1)
        if target.shape[-1] != pred_base.shape[-1]:
            raise ValueError(f"zero_target channels {target.shape[-1]} do not match predictions {pred_base.shape[-1]}.")
        alpha = pred_base.new_tensor(self.alpha)
        return pred_base + zero_mask * alpha * (target - pred_base)


def _zero_mask(future_masks: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    masks = future_masks.to(device=pred.device, dtype=pred.dtype)
    if masks.ndim == 4:
        if masks.shape[2] <= 1:
            return torch.zeros_like(pred)
        mask = masks[:, :, 1, :]
        return mask.expand_as(pred) if mask.shape[-1] == 1 else mask
    if masks.ndim == 3:
        if masks.shape[-1] == 3:
            return masks[:, :, 1:2].expand_as(pred)
        if masks.shape[-1] >= pred.shape[-1]:
            return masks[:, :, : pred.shape[-1]]
        if masks.shape[-1] >= 2:
            return masks[:, :, 1:2].expand_as(pred)
        return torch.zeros_like(pred)
    if masks.ndim == 2:
        return torch.zeros_like(pred)
    raise ValueError("future_masks must be shaped [B,L,3,C], [B,L,M], or [B,L].")
