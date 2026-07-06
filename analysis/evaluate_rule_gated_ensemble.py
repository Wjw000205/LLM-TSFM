"""Evaluate a rule-gated ensemble between baseline and event-specialized predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import ensure_dir


def rule_gated_prediction(baseline_pred, event_pred, masks, alpha: float = 1.0) -> np.ndarray:
    """Use event-specialized predictions only where the event mask is active."""
    baseline = np.asarray(baseline_pred, dtype=np.float32)
    event = np.asarray(event_pred, dtype=np.float32)
    if baseline.shape != event.shape:
        raise ValueError(f"Prediction shapes must match, got {baseline.shape} and {event.shape}")
    mask = extract_event_mask(masks, baseline.shape)
    return baseline + float(alpha) * mask * (event - baseline)


def extract_event_mask(masks, prediction_shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(masks, dtype=np.float32)
    if mask.ndim == 4:
        mask = mask[:, :, 0, :]
    elif mask.ndim == 2:
        mask = mask[..., None]
    if mask.shape[:2] != prediction_shape[:2]:
        raise ValueError(f"Mask time shape {mask.shape[:2]} does not match predictions {prediction_shape[:2]}")
    if mask.shape[-1] == 1:
        mask = np.repeat(mask, prediction_shape[-1], axis=-1)
    if mask.shape != prediction_shape:
        raise ValueError(f"Mask shape {mask.shape} cannot broadcast to predictions {prediction_shape}")
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def evaluate_rule_gated_ensemble(
    baseline_result_dir: str,
    event_result_dir: str,
    output_dir: str,
    alpha: float = 1.0,
):
    baseline_dir = Path(baseline_result_dir)
    event_dir = Path(event_result_dir)
    output = ensure_dir(output_dir)

    baseline_pred = np.load(baseline_dir / "pred_normalized.npy")
    event_pred = np.load(event_dir / "pred_normalized.npy")
    true = np.load(baseline_dir / "true_normalized.npy")
    masks = load_test_masks(baseline_dir / "config.json", expected_windows=baseline_pred.shape[0])

    pred = rule_gated_prediction(baseline_pred, event_pred, masks, alpha=alpha)
    metrics_normalized = metric(pred, true, masks=masks)

    np.save(output / "pred_normalized.npy", pred)
    np.save(output / "true_normalized.npy", true)
    if (baseline_dir / "true_original.npy").exists() and (baseline_dir / "pred_original.npy").exists():
        baseline_original = np.load(baseline_dir / "pred_original.npy")
        event_original = np.load(event_dir / "pred_original.npy")
        true_original = np.load(baseline_dir / "true_original.npy")
        pred_original = rule_gated_prediction(baseline_original, event_original, masks, alpha=alpha)
        metrics_original = metric(pred_original, true_original, masks=masks)
        np.save(output / "pred_original.npy", pred_original)
        np.save(output / "true_original.npy", true_original)
    else:
        metrics_original = {}

    payload = {
        "method": "rule_gated_ensemble",
        "baseline_result_dir": str(baseline_dir),
        "event_result_dir": str(event_dir),
        "alpha": float(alpha),
        "metrics_normalized": metrics_normalized,
        "metrics_original_scale": metrics_original,
    }
    (output / "metrics_normalized.json").write_text(json.dumps(metrics_normalized, indent=2), encoding="utf-8")
    (output / "metrics_original_scale.json").write_text(json.dumps(metrics_original, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def load_test_masks(config_path: str | Path, expected_windows: int | None = None) -> np.ndarray:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    args = SimpleNamespace(**config)
    args.batch_size = int(getattr(args, "batch_size", 8))
    args.num_workers = int(getattr(args, "num_workers", 0))
    if not hasattr(args, "timeenc"):
        args.timeenc = 0
    if not hasattr(args, "freq"):
        args.freq = "t" if str(getattr(args, "data", "")).startswith("ETTm") else "h"
    if not hasattr(args, "use_shift_aware_rule"):
        args.use_shift_aware_rule = 0
    if not hasattr(args, "rule_shift_steps"):
        args.rule_shift_steps = 0

    data_provider(args, "train")
    _, loader = data_provider(args, "test")
    rows = []
    for batch in loader:
        rows.append(batch[-1].numpy()[:, -int(args.pred_len) :, :, :])
    masks = np.concatenate(rows, axis=0)
    if expected_windows is not None and masks.shape[0] != expected_windows:
        raise ValueError(f"Loaded {masks.shape[0]} mask windows, expected {expected_windows}")
    return masks.astype(np.float32)


def _jsonable(payload):
    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonable(value) for value in payload]
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate a rule-gated baseline/event-model ensemble.")
    parser.add_argument("--baseline_result_dir", required=True)
    parser.add_argument("--event_result_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    result = evaluate_rule_gated_ensemble(**vars(args))
    print(json.dumps(result["metrics_normalized"], indent=2))


if __name__ == "__main__":
    main()
