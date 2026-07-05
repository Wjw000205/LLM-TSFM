"""Forecasting metrics, including event-window diagnostics."""

from __future__ import annotations

import warnings

import numpy as np


EPS = 1e-8


def mse(pred, true):
    return np.mean((pred - true) ** 2)


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def rmse(pred, true):
    return np.sqrt(mse(pred, true))


def mape(pred, true):
    return np.mean(np.abs((pred - true) / (true + EPS)))


def mspe(pred, true):
    return np.mean(np.square((pred - true) / (true + EPS)))


def metric(pred, true, masks=None) -> dict[str, float]:
    """Compute base and event-aware metrics for arrays shaped ``[N, L, C]``."""
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    result = {
        "mse": float(mse(pred, true)),
        "mae": float(mae(pred, true)),
        "rmse": float(rmse(pred, true)),
        "mape": float(mape(pred, true)),
        "mspe": float(mspe(pred, true)),
    }

    if masks is None:
        return result

    masks = np.asarray(masks, dtype=np.float32)
    event_mask = _mask_channel(masks, 0)
    zero_mask = _mask_channel(masks, 1)
    peak_mask = _mask_channel(masks, 2)

    result["event_window_mse"] = _masked_metric(pred, true, event_mask, squared=True)
    result["event_window_mae"] = _masked_metric(pred, true, event_mask, squared=False)
    result["zero_event_mse"] = _masked_metric(pred, true, zero_mask, squared=True)
    result["zero_event_mae"] = _masked_metric(pred, true, zero_mask, squared=False)
    result["peak_event_mse"] = _masked_metric(pred, true, peak_mask, squared=True)
    result["peak_event_mae"] = _masked_metric(pred, true, peak_mask, squared=False)
    result["rule_consistency_score"] = _rule_consistency_score(pred, zero_mask, peak_mask)
    result["num_event_points"] = int(event_mask.sum())
    result["num_zero_event_points"] = int(zero_mask.sum())
    result["num_peak_event_points"] = int(peak_mask.sum())
    if result["num_event_points"] == 0 and result["num_zero_event_points"] == 0 and result["num_peak_event_points"] == 0:
        warnings.warn("Event mask is empty; event-window metrics are zero by construction.", UserWarning, stacklevel=2)
    return result


def _mask_channel(masks: np.ndarray, channel: int) -> np.ndarray:
    if masks.ndim == 4:
        if masks.shape[2] <= channel:
            return np.zeros_like(masks[:, :, 0, :])
        return masks[:, :, channel, :]
    if masks.ndim == 2:
        masks = masks[..., None]
    if masks.shape[-1] <= channel:
        return np.zeros((*masks.shape[:2], 1), dtype=np.float32)
    return masks[..., channel : channel + 1]


def _masked_metric(pred, true, mask, squared: bool) -> float:
    denom = mask.sum() * pred.shape[-1] if mask.shape[-1] == 1 else mask.sum()
    if denom <= EPS:
        return 0.0
    error = pred - true
    if squared:
        error = error**2
    else:
        error = np.abs(error)
    return float((error * mask).sum() / (denom + EPS))


def _rule_consistency_score(pred, zero_mask, peak_mask) -> float:
    scores = []
    if zero_mask.sum() > EPS:
        denom = zero_mask.sum() * pred.shape[-1] if zero_mask.shape[-1] == 1 else zero_mask.sum()
        zero_mae = (np.abs(pred) * zero_mask).sum() / (denom + EPS)
        scores.append(1.0 / (1.0 + float(zero_mae)))
    if peak_mask.sum() > EPS and pred.shape[1] >= 3:
        center = pred[:, 1:-1, :]
        left = pred[:, :-2, :]
        right = pred[:, 2:, :]
        peak = peak_mask[:, 1:-1, :]
        valid = peak.sum() * pred.shape[-1] if peak.shape[-1] == 1 else peak.sum()
        if valid > EPS:
            is_peak = ((center >= left) & (center >= right)).astype(np.float32)
            scores.append(float((is_peak * peak).sum() / (valid + EPS)))
    if not scores:
        return 0.0
    return float(np.mean(scores))
