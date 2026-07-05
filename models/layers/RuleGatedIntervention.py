"""Timestamp-conditioned rule-gated intervention layer."""

from __future__ import annotations

import torch
import torch.nn as nn


class RuleGatedIntervention(nn.Module):
    """Apply event-gated residual updates to horizon hidden states."""

    def __init__(
        self,
        hidden_dim: int,
        feature_dim: int,
        intervention_hidden: int = 32,
        dropout: float = 0.0,
        intervention_scale: float = 1.0,
        init_zero: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = int(feature_dim)
        self.intervention_hidden = int(intervention_hidden)
        self.intervention_scale = float(intervention_scale)
        self.eps = eps
        self.last_gate: torch.Tensor | None = None
        self.last_delta: torch.Tensor | None = None
        self.last_event_mask: torch.Tensor | None = None

        input_dim = self.hidden_dim + self.feature_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, self.intervention_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.intervention_hidden, self.hidden_dim),
        )
        self.adapter_mlp = nn.Sequential(
            nn.Linear(input_dim, self.intervention_hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.intervention_hidden, self.hidden_dim),
        )
        if init_zero:
            last = self.adapter_mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, h: torch.Tensor, future_features: torch.Tensor | None = None, future_masks: torch.Tensor | None = None):
        """Return hidden states after local event-gated intervention."""
        self._reset_last(h)
        if future_features is None or future_masks is None:
            return h
        if future_features.numel() == 0 or future_features.shape[-1] <= 0:
            return h
        if future_masks.numel() == 0:
            return h
        if future_features.shape[:2] != h.shape[:2]:
            raise ValueError(
                f"future_features first two dims {tuple(future_features.shape[:2])} must match h {tuple(h.shape[:2])}."
            )

        event_mask = _event_mask(future_masks, h)
        if event_mask.shape[:2] != h.shape[:2]:
            raise ValueError(f"future_masks first two dims {tuple(event_mask.shape[:2])} must match h {tuple(h.shape[:2])}.")

        adapter_input = torch.cat([h, future_features.to(device=h.device, dtype=h.dtype)], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(adapter_input))
        delta = self.adapter_mlp(adapter_input)
        self.last_gate = gate
        self.last_delta = delta
        self.last_event_mask = event_mask
        return h + event_mask * gate * delta * self.intervention_scale

    def get_intervention_reg_loss(self) -> torch.Tensor:
        """Penalize gate and delta activity outside event windows."""
        if self.last_gate is None or self.last_delta is None or self.last_event_mask is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        non_event_mask = torch.clamp(1.0 - self.last_event_mask, min=0.0, max=1.0)
        gate_loss = _masked_mean(self.last_gate.square(), non_event_mask)
        delta_loss = _masked_mean(self.last_delta.square(), non_event_mask)
        return gate_loss + delta_loss

    def get_intervention_stats(self) -> dict[str, float]:
        """Return scalar diagnostics for logging and result summaries."""
        if self.last_gate is None or self.last_delta is None or self.last_event_mask is None:
            return _zero_stats()
        with torch.no_grad():
            event_mask = self.last_event_mask
            non_event_mask = torch.clamp(1.0 - event_mask, min=0.0, max=1.0)
            delta_norm = self.last_delta.norm(dim=-1, keepdim=True)
            return {
                "mean_event_gate": float(_masked_mean(self.last_gate, event_mask).detach().cpu()),
                "mean_non_event_gate": float(_masked_mean(self.last_gate, non_event_mask).detach().cpu()),
                "mean_event_delta_norm": float(_masked_mean(delta_norm, event_mask, channels=1).detach().cpu()),
                "mean_non_event_delta_norm": float(_masked_mean(delta_norm, non_event_mask, channels=1).detach().cpu()),
            }

    def _reset_last(self, h: torch.Tensor):
        zero_gate = h.new_zeros(*h.shape)
        zero_mask = h.new_zeros(h.shape[0], h.shape[1], 1)
        self.last_gate = zero_gate
        self.last_delta = zero_gate
        self.last_event_mask = zero_mask


def _event_mask(future_masks: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    mask = future_masks.to(device=h.device, dtype=h.dtype)
    if mask.ndim == 4:
        event_mask = mask[:, :, 0, :]
        return torch.clamp(event_mask.amax(dim=-1, keepdim=True), min=0.0, max=1.0)
    if mask.ndim == 3:
        return torch.clamp(mask[:, :, 0:1], min=0.0, max=1.0)
    if mask.ndim == 2:
        return torch.clamp(mask.unsqueeze(-1), min=0.0, max=1.0)
    raise ValueError("future_masks must be shaped [B,L,3,C], [B,L,M], or [B,L].")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    channels = values.shape[-1] if channels is None else channels
    denom = mask.sum() * channels
    if denom.item() <= 1e-8:
        return values.new_tensor(0.0)
    return (values * mask).sum() / (denom + 1e-8)


def _zero_stats() -> dict[str, float]:
    return {
        "mean_event_gate": 0.0,
        "mean_non_event_gate": 0.0,
        "mean_event_delta_norm": 0.0,
        "mean_non_event_delta_norm": 0.0,
    }
