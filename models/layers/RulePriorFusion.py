"""Executable LLM rule-prior fusion branch."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


class RulePriorFusion(nn.Module):
    """Softly fuse compiled rule priors into forecasts at rule-triggered timestamps."""

    def __init__(
        self,
        alpha: float = 0.5,
        use_confidence: bool = False,
        rule_prior_types: str | list[str] | tuple[str, ...] = "zero_event",
        mode: str = "fixed",
        validated_rule_path: str | None = None,
        disable_invalid_rules: bool = True,
        channel_names: list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.use_confidence = bool(use_confidence)
        self.mode = str(mode)
        self.validated_rule_path = validated_rule_path
        self.disable_invalid_rules = bool(disable_invalid_rules)
        self.channel_names = list(channel_names or [])
        self.calibrated_rules = _load_calibrated_rules(validated_rule_path)
        if isinstance(rule_prior_types, str):
            self.rule_prior_types = {item.strip() for item in rule_prior_types.split(",") if item.strip()}
        else:
            self.rule_prior_types = {str(item).strip() for item in rule_prior_types if str(item).strip()}

    def forward(self, pred_base: torch.Tensor, future_masks: torch.Tensor, zero_target, future_features=None, rule_config=None):
        """Return soft-fused predictions from fixed or validated rule priors."""
        if "zero_event" not in self.rule_prior_types:
            return pred_base
        zero_mask = _zero_mask(future_masks, pred_base)
        if zero_mask.sum().item() == 0:
            return pred_base
        if self.mode == "calibrated" and self.calibrated_rules is not None:
            return self._forward_calibrated(pred_base, zero_mask, zero_target)
        if self.alpha == 0.0:
            return pred_base
        target = torch.as_tensor(zero_target, dtype=pred_base.dtype, device=pred_base.device).view(1, 1, -1)
        if target.shape[-1] != pred_base.shape[-1]:
            raise ValueError(f"zero_target channels {target.shape[-1]} do not match predictions {pred_base.shape[-1]}.")
        alpha = pred_base.new_tensor(self.alpha)
        return pred_base + zero_mask * alpha * (target - pred_base)

    def _forward_calibrated(self, pred_base: torch.Tensor, zero_mask: torch.Tensor, zero_target) -> torch.Tensor:
        channel_names = self.channel_names or [str(idx) for idx in range(pred_base.shape[-1])]
        if len(channel_names) != pred_base.shape[-1]:
            channel_names = [str(idx) for idx in range(pred_base.shape[-1])]
        target = torch.as_tensor(zero_target, dtype=pred_base.dtype, device=pred_base.device).view(-1)
        fused = pred_base.clone()
        applied = torch.zeros(pred_base.shape[-1], dtype=torch.bool, device=pred_base.device)
        for pattern in self.calibrated_rules.get("patterns", []):
            if not bool(pattern.get("enabled", False)):
                continue
            if pattern.get("type") != "zero_event":
                continue
            valid_channels = set(pattern.get("valid_channels", []))
            diagnostics = pattern.get("channel_diagnostics", {})
            for channel_idx, channel_name in enumerate(channel_names):
                if valid_channels and channel_name not in valid_channels:
                    continue
                diag = diagnostics.get(channel_name)
                if not diag:
                    continue
                alpha = float(pattern.get("calibrated_alpha", {}).get(channel_name, diag.get("best_alpha", 0.0)))
                if alpha <= 0.0:
                    continue
                prior_type = str(diag.get("best_prior_type", diag.get("selected_prior_type", "zero_target")))
                prior_value = float(diag.get("prior_value", diag.get(prior_type, 0.0)))
                base = pred_base[:, :, channel_idx]
                if prior_type == "zero_target":
                    if channel_idx >= target.numel():
                        continue
                    prior = target[channel_idx].expand_as(base)
                    updated = base + alpha * (prior - base)
                elif prior_type in {"residual_mean", "residual_median"}:
                    updated = base + alpha * prior_value
                elif prior_type == "ratio":
                    prior = base * prior_value
                    updated = base + alpha * (prior - base)
                elif prior_type == "conditional_mean":
                    prior = base.new_full(base.shape, prior_value)
                    updated = base + alpha * (prior - base)
                else:
                    continue
                mask = zero_mask[:, :, channel_idx]
                fused[:, :, channel_idx] = torch.where(mask > 0, updated, fused[:, :, channel_idx])
                applied[channel_idx] = True
        if self.disable_invalid_rules and not bool(applied.any().item()):
            return pred_base
        return fused


def _load_calibrated_rules(path: str | None) -> dict | None:
    if not path:
        return None
    rule_path = Path(path)
    if not rule_path.exists():
        return None
    return json.loads(rule_path.read_text(encoding="utf-8"))


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
