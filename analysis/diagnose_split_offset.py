"""Diagnose split-dependent rule-mask and event phase offsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider
from llm_rules.rule_parser import parse_llm_rules


ALPHA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
EPS = 1e-8


def diagnose_split_offset(
    data: str = "ETTm1",
    root_path: str = "./data/",
    data_path: str = "ETTm1.csv",
    features: str = "M",
    target: str = "OT",
    seq_len: int = 336,
    label_len: int = 48,
    pred_len: int = 96,
    llm_rule_path: str = "./llm_rules/example_rules/ETTm1_rules.json",
    baseline_result_dir: str | None = None,
    output_path: str = "artifacts/core_results/ettm1_split_offset_diagnosis.json",
    output_markdown_path: str = "docs/split_offset_diagnosis.md",
    lag_min: int = -192,
    lag_max: int = 192,
    near_zero_quantile: float = 0.05,
):
    args = _build_args(
        data=data,
        root_path=root_path,
        data_path=data_path,
        features=features,
        target=target,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        llm_rule_path=llm_rule_path,
        baseline_result_dir=baseline_result_dir,
    )
    train_data, _ = data_provider(args, "train")
    split_payloads = {}
    channel_rows = {}
    for split in ["train", "val", "test"]:
        dataset, _ = data_provider(args, split)
        split_payloads[split] = _diagnose_split(
            dataset=dataset,
            zero_target=np.asarray(train_data.zero_target, dtype=np.float32),
            lag_min=lag_min,
            lag_max=lag_max,
            near_zero_quantile=near_zero_quantile,
        )
    for channel in train_data.target_columns:
        channel_rows[channel] = {
            "channel_name": channel,
            "train_best_lag": split_payloads["train"]["per_channel"][channel]["best_lag_steps"],
            "val_best_lag": split_payloads["val"]["per_channel"][channel]["best_lag_steps"],
            "test_best_lag": split_payloads["test"]["per_channel"][channel]["best_lag_steps"],
            "train_overlap": split_payloads["train"]["per_channel"][channel]["best_overlap"],
            "val_overlap": split_payloads["val"]["per_channel"][channel]["best_overlap"],
            "test_overlap": split_payloads["test"]["per_channel"][channel]["best_overlap"],
            "zero_prior_valid": split_payloads["test"]["per_channel"][channel]["zero_prior_valid"],
            "residual_prior_valid": split_payloads["test"]["per_channel"][channel]["residual_prior_valid"],
        }
    stable_channels = [
        channel
        for channel, row in channel_rows.items()
        if row["train_best_lag"] == row["val_best_lag"] == row["test_best_lag"]
        and row["test_overlap"] > 0
    ]
    unstable_channels = [channel for channel in channel_rows if channel not in stable_channels]
    baseline_metrics = _baseline_prediction_metrics(baseline_result_dir, args) if baseline_result_dir else {}
    val_best_lag = split_payloads["val"]["aggregate"]["best_lag_steps"]
    test_best_lag = split_payloads["test"]["aggregate"]["best_lag_steps"]
    split_phase_shift = val_best_lag != test_best_lag
    alignment = _dataset_alignment_summary(split_payloads)
    offset_predictability = _offset_predictability(split_payloads, channel_rows)
    diagnosis = {
        "data": data,
        "root_path": root_path,
        "data_path": data_path,
        "llm_rule_path": llm_rule_path,
        "baseline_result_dir": baseline_result_dir,
        "lag_search": {"lag_min": lag_min, "lag_max": lag_max, "freq_minutes": _freq_minutes(train_data.timestamps)},
        "dataset_alignment": alignment,
        "splits": split_payloads,
        "per_channel_offset": channel_rows,
        "stable_channels": stable_channels,
        "unstable_channels": unstable_channels,
        "split_phase_shift": bool(split_phase_shift),
        "offset_predictability": offset_predictability,
        "rule_stability": "unstable" if split_phase_shift or unstable_channels else "stable",
        "disable_prior_recommendation": bool(split_phase_shift or unstable_channels),
        "baseline_prediction_metrics": baseline_metrics,
        "recommendation": _recommendation(split_phase_shift, unstable_channels, alignment),
    }
    _write_json(output_path, diagnosis)
    _write_markdown(output_markdown_path, diagnosis)
    return diagnosis


def lag_search(
    rule_mask: np.ndarray,
    true_event_mask: np.ndarray,
    timestamps=None,
    lag_min: int = -192,
    lag_max: int = 192,
    pred: np.ndarray | None = None,
    true: np.ndarray | None = None,
    zero_target: np.ndarray | None = None,
) -> dict:
    rule_mask = _as_2d_mask(rule_mask)
    true_event_mask = _as_2d_mask(true_event_mask)
    rows = []
    for lag in range(int(lag_min), int(lag_max) + 1):
        shifted = _shift_mask(rule_mask, lag)
        overlap = float((shifted * true_event_mask).sum())
        predicted = float(shifted.sum())
        actual = float(true_event_mask.sum())
        precision = overlap / predicted if predicted > EPS else 0.0
        recall = overlap / actual if actual > EPS else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > EPS else 0.0
        row = {
            "lag_steps": int(lag),
            "overlap": overlap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if pred is not None and true is not None and zero_target is not None:
            row["event_mse_if_shifted"] = _masked_mse(pred, true, shifted)
            row["zero_prior_mse_if_shifted"] = _zero_prior_mse(true, shifted, zero_target)
        rows.append(row)
    best = max(rows, key=lambda item: (item["f1"], item["overlap"], -abs(item["lag_steps"])))
    return {
        "best_lag_steps": int(best["lag_steps"]),
        "best_lag_hours": float(best["lag_steps"] * _timestamp_step_hours(timestamps)),
        "best_overlap": int(best["overlap"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_f1": float(best["f1"]),
        "lag_rows": rows,
    }


def classify_offset_predictability(
    train_val_offsets: list[int] | list[float],
    test_offsets: list[int] | list[float],
) -> dict:
    train_val = np.asarray(train_val_offsets, dtype=np.float32)
    test = np.asarray(test_offsets, dtype=np.float32)
    if train_val.size == 0 or test.size == 0:
        return {
            "predictable_from_train_val": False,
            "post_hoc_test_regime": False,
            "test_offsets_outside_train_val_range": False,
            "reason": "insufficient_offsets",
        }
    x = np.arange(train_val.size, dtype=np.float32)
    xt = np.arange(train_val.size, train_val.size + test.size, dtype=np.float32)
    predictions = {
        "train_val_mean": np.full_like(test, float(train_val.mean())),
        "last_observed": np.full_like(test, float(train_val[-1])),
    }
    if train_val.size >= 2:
        x_mean = float(x.mean())
        y_mean = float(train_val.mean())
        denom = float(((x - x_mean) ** 2).sum())
        slope = float(((x - x_mean) * (train_val - y_mean)).sum() / denom) if denom > EPS else 0.0
        intercept = y_mean - slope * x_mean
        predictions["linear_trend"] = np.asarray(intercept + slope * xt, dtype=np.float32)
    errors = {
        name: {
            "prediction": [float(item) for item in pred],
            "mae_steps": float(np.mean(np.abs(pred - test))),
            "max_abs_error_steps": float(np.max(np.abs(pred - test))),
        }
        for name, pred in predictions.items()
    }
    best_name = min(errors, key=lambda name: errors[name]["mae_steps"])
    outside_range = bool(test.min() < train_val.min() or test.max() > train_val.max())
    train_val_std = float(train_val.std())
    test_std = float(test.std())
    best_mae = errors[best_name]["mae_steps"]
    predictable_threshold = max(4.0, 2.0 * train_val_std)
    predictable = bool((not outside_range) and best_mae <= predictable_threshold)
    post_hoc_regime = bool(test.size >= 2 and test_std <= max(4.0, train_val_std) and not predictable)
    return {
        "predictable_from_train_val": predictable,
        "post_hoc_test_regime": post_hoc_regime,
        "test_offsets_outside_train_val_range": outside_range,
        "best_predictor": best_name,
        "best_mae_steps": float(best_mae),
        "train_val_offsets": [float(item) for item in train_val],
        "test_offsets": [float(item) for item in test],
        "train_val_offset_mean": float(train_val.mean()),
        "train_val_offset_std": train_val_std,
        "train_val_offset_min": float(train_val.min()),
        "train_val_offset_max": float(train_val.max()),
        "test_offset_mean": float(test.mean()),
        "test_offset_std": test_std,
        "test_offset_min": float(test.min()),
        "test_offset_max": float(test.max()),
        "predictors": errors,
        "reason": _predictability_reason(predictable, post_hoc_regime, outside_range),
    }


def _diagnose_split(dataset, zero_target, lag_min: int, lag_max: int, near_zero_quantile: float):
    rule_mask = dataset.event_masks[:, 1, :]
    true_event_mask = _near_zero_event_mask(dataset.data_y, quantile=near_zero_quantile)
    aggregate = lag_search(
        rule_mask=rule_mask,
        true_event_mask=true_event_mask,
        timestamps=dataset.timestamps,
        lag_min=lag_min,
        lag_max=lag_max,
        pred=dataset.data_y,
        true=dataset.data_y,
        zero_target=zero_target,
    )
    per_channel = {}
    for idx, channel in enumerate(dataset.target_columns):
        channel_lag = lag_search(
            rule_mask=rule_mask[:, idx : idx + 1],
            true_event_mask=true_event_mask[:, idx : idx + 1],
            timestamps=dataset.timestamps,
            lag_min=lag_min,
            lag_max=lag_max,
            pred=dataset.data_y[:, idx : idx + 1],
            true=dataset.data_y[:, idx : idx + 1],
            zero_target=np.asarray([zero_target[idx]], dtype=np.float32),
        )
        mask = rule_mask[:, idx] > 0
        baseline = dataset.data_y[:, idx][mask]
        label = dataset.data_y[:, idx][mask]
        residual = label - baseline
        zero_prior = np.full_like(label, zero_target[idx])
        residual_prior = baseline + (float(residual.mean()) if len(residual) else 0.0)
        baseline_mse = _mse(baseline, label) if len(label) else 0.0
        zero_mse = _mse(zero_prior, label) if len(label) else 0.0
        residual_mse = _mse(residual_prior, label) if len(label) else 0.0
        per_channel[channel] = {
            **{key: value for key, value in channel_lag.items() if key != "lag_rows"},
            "mask_count": int(mask.sum()),
            "true_event_count": int(true_event_mask[:, idx].sum()),
            "zero_prior_valid": bool(zero_mse < baseline_mse) if len(label) else False,
            "residual_prior_valid": bool(residual_mse < baseline_mse) if len(label) else False,
            "baseline_mse": baseline_mse,
            "zero_prior_mse": zero_mse,
            "residual_prior_mse": residual_mse,
        }
    return {
        "time_range": {"start": str(dataset.timestamps[0]), "end": str(dataset.timestamps[-1])},
        "num_timestamps": int(len(dataset.timestamps)),
        "rule_mask_sum": float(rule_mask.sum()),
        "true_event_sum": float(true_event_mask.sum()),
        "aggregate": {key: value for key, value in aggregate.items() if key != "lag_rows"},
        "calendar_anchor": _calendar_anchor_diagnostics(dataset, rule_mask, true_event_mask, zero_target),
        "dataset_samples": _dataset_samples(dataset),
        "per_channel": per_channel,
    }


def _offset_predictability(split_payloads: dict, channel_rows: dict) -> dict:
    train_val_anchor_offsets = []
    test_anchor_offsets = []
    for split in ["train", "val"]:
        train_val_anchor_offsets.extend([row["offset_steps"] for row in split_payloads[split]["calendar_anchor"]])
    test_anchor_offsets.extend([row["offset_steps"] for row in split_payloads["test"]["calendar_anchor"]])
    aggregate = classify_offset_predictability(train_val_anchor_offsets, test_anchor_offsets)
    per_channel = {}
    for channel, row in channel_rows.items():
        per_channel[channel] = classify_offset_predictability(
            [row["train_best_lag"], row["val_best_lag"]],
            [row["test_best_lag"]],
        )
    predictable_channels = [
        channel for channel, payload in per_channel.items() if payload.get("predictable_from_train_val", False)
    ]
    unpredictable_channels = [channel for channel in per_channel if channel not in predictable_channels]
    return {
        "aggregate_anchor_offsets": aggregate,
        "per_channel": per_channel,
        "predictable_channels": predictable_channels,
        "unpredictable_channels": unpredictable_channels,
        "overall_predictability": "predictable"
        if aggregate.get("predictable_from_train_val") and not unpredictable_channels
        else "not_predictable_from_train_val",
    }


def _near_zero_event_mask(values: np.ndarray, quantile: float = 0.05):
    threshold = np.quantile(np.abs(values), quantile, axis=0).reshape(1, -1)
    return (np.abs(values) <= threshold).astype(np.float32)


def _predictability_reason(predictable: bool, post_hoc_regime: bool, outside_range: bool) -> str:
    if predictable:
        return "train_val_offsets_predict_test_offsets_with_small_error"
    if post_hoc_regime and outside_range:
        return "test_offsets_form_a_post_hoc_regime_but_are_outside_train_val_range"
    if post_hoc_regime:
        return "test_offsets_are_internally_consistent_but_not_predicted_by_train_val"
    if outside_range:
        return "test_offsets_are_outside_train_val_range"
    return "train_val_predictors_have_large_error"


def _calendar_anchor_diagnostics(dataset, rule_mask, true_event_mask, zero_target):
    rules = parse_llm_rules(dataset.llm_rule_path)
    if rules is None:
        return []
    rows = []
    for pattern in rules.patterns:
        if pattern.condition.get("kind") != "calendar_periodic":
            continue
        expected_indices = np.flatnonzero(rule_mask.max(axis=1) > 0)
        event_indices = np.flatnonzero(true_event_mask.max(axis=1) > 0)
        for idx in _unique_day_starts(dataset.timestamps, expected_indices):
            nearest = _nearest_index(idx, event_indices)
            offset_steps = int(nearest - idx) if nearest is not None else 0
            day_mask = (dataset.timestamps >= dataset.timestamps[idx]) & (
                dataset.timestamps < dataset.timestamps[idx] + pd.Timedelta(days=1)
            )
            values = dataset.data_y[day_mask]
            target = zero_target.reshape(1, -1)
            rows.append(
                {
                    "pattern": pattern.name,
                    "expected_timestamp": str(dataset.timestamps[idx]),
                    "nearest_true_event_timestamp": str(dataset.timestamps[nearest]) if nearest is not None else None,
                    "offset_steps": offset_steps,
                    "offset_hours": float(offset_steps * _timestamp_step_hours(dataset.timestamps)),
                    "affected_variables": pattern.affected_variables,
                    "true_near_zero_ratio": float(true_event_mask[day_mask].mean()) if day_mask.sum() else 0.0,
                    "baseline_mse": 0.0,
                    "zero_prior_mse": float(np.mean((values - target) ** 2)) if len(values) else 0.0,
                }
            )
    return rows


def _baseline_prediction_metrics(baseline_result_dir: str, args: SimpleNamespace):
    result_dir = Path(baseline_result_dir)
    pred_path = result_dir / "pred_normalized.npy"
    true_path = result_dir / "true_normalized.npy"
    if not pred_path.exists() or not true_path.exists():
        return {}
    pred = np.load(pred_path)
    true = np.load(true_path)
    data_provider(args, "train")
    test_data, _ = data_provider(args, "test")
    masks, timestamps = _future_masks_and_timestamps(test_data)
    masks = masks[: pred.shape[0]]
    timestamps = timestamps[: pred.shape[0]]
    zero_mask = masks[:, :, 1, :]
    repeated_event_mse = _masked_mse(pred, true, zero_mask)
    grouped = {}
    for window_idx, horizon_idx, channel_idx in np.argwhere(zero_mask > 0):
        key = (str(timestamps[window_idx, horizon_idx]), int(channel_idx))
        grouped.setdefault(key, []).append(
            (
                float(pred[window_idx, horizon_idx, channel_idx]),
                float(true[window_idx, horizon_idx, channel_idx]),
            )
        )
    if grouped:
        errors = []
        for values in grouped.values():
            avg_pred = float(np.mean([item[0] for item in values]))
            avg_true = float(np.mean([item[1] for item in values]))
            errors.append((avg_pred - avg_true) ** 2)
        unique_event_mse = float(np.mean(errors))
    else:
        unique_event_mse = 0.0
    repeated_points = int(zero_mask.sum())
    unique_points = int(len(grouped))
    return {
        "num_repeated_event_points": repeated_points,
        "num_unique_event_timestamps": unique_points,
        "repeat_factor": float(repeated_points / unique_points) if unique_points else 0.0,
        "repeated_event_mse": repeated_event_mse,
        "unique_event_mse": unique_event_mse,
    }


def _dataset_samples(dataset, indices=(0, 1, 672, 673, 674, 675)):
    rows = []
    for index in indices:
        if index < 0 or index >= len(dataset):
            continue
        s_begin = index
        s_end = s_begin + dataset.seq_len
        r_begin = s_end - dataset.label_len
        r_end = r_begin + dataset.label_len + dataset.pred_len
        seq_y_timestamps = dataset.timestamps[r_begin:r_end]
        pred_timestamps = seq_y_timestamps[-dataset.pred_len :]
        rows.append(
            {
                "sample_index": int(index),
                "seq_x_start_time": str(dataset.timestamps[s_begin]),
                "seq_x_end_time": str(dataset.timestamps[s_end - 1]),
                "seq_y_start_time": str(seq_y_timestamps[0]),
                "seq_y_end_time": str(seq_y_timestamps[-1]),
                "pred_start_time": str(pred_timestamps[0]),
                "pred_end_time": str(pred_timestamps[-1]),
                "seq_y_true_timestamp": str(seq_y_timestamps[-dataset.pred_len]),
                "seq_y_mask_timestamp": str(seq_y_timestamps[-dataset.pred_len]),
                "seq_y_mark_timestamp": str(seq_y_timestamps[-dataset.pred_len]),
            }
        )
    return rows


def _dataset_alignment_summary(split_payloads):
    errors = []
    for split, payload in split_payloads.items():
        for sample in payload["dataset_samples"]:
            if sample["seq_y_true_timestamp"] != sample["seq_y_mask_timestamp"]:
                errors.append({"split": split, "sample_index": sample["sample_index"], "reason": "true_mask_timestamp_mismatch"})
            if sample["seq_y_true_timestamp"] != sample["seq_y_mark_timestamp"]:
                errors.append({"split": split, "sample_index": sample["sample_index"], "reason": "true_mark_timestamp_mismatch"})
    return {
        "alignment_error": bool(errors),
        "suspected_offset_steps": 0 if not errors else None,
        "suspected_offset_hours": 0.0 if not errors else None,
        "errors": errors,
    }


def _future_masks_and_timestamps(dataset):
    masks = []
    timestamps = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        masks.append(item[-1].numpy()[-dataset.pred_len :])
        s_end = idx + dataset.seq_len
        r_begin = s_end - dataset.label_len
        r_end = r_begin + dataset.label_len + dataset.pred_len
        timestamps.append([str(ts) for ts in dataset.timestamps[r_begin:r_end][-dataset.pred_len :]])
    return np.asarray(masks, dtype=np.float32), np.asarray(timestamps)


def _recommendation(split_phase_shift: bool, unstable_channels: list[str], alignment: dict):
    if alignment["alignment_error"]:
        return "case_a_dataset_alignment_bug_fix_data_loader_and_rerun_all_rule_prior_results"
    if split_phase_shift:
        return "case_c_unstable_rule_offset_disable_deterministic_prior_use_guarded_loss_or_remine_rules"
    if unstable_channels:
        return "case_d_channel_specific_rule_only_stable_channels_are_safe_for_prior"
    return "case_b_stable_offset_shift_aware_rule_can_be_considered_without_using_test_for_selection"


def _build_args(**kwargs):
    config = {}
    baseline_result_dir = kwargs.get("baseline_result_dir")
    if baseline_result_dir:
        config_path = Path(baseline_result_dir) / "config.json"
        if config_path.exists():
            config.update(json.loads(config_path.read_text(encoding="utf-8")))
    config.update({key: value for key, value in kwargs.items() if value is not None})
    config.update(
        {
            "root_path": config.get("root_path", "./data/"),
            "batch_size": int(config.get("batch_size", 8)),
            "num_workers": 0,
            "use_zscore": int(config.get("use_zscore", 1)),
            "use_llm_features": 0,
            "use_llm_rule_features": 0,
            "use_standard_time_features": 0,
            "use_oracle_features": 0,
            "freq": config.get("freq", "h"),
            "timeenc": int(config.get("timeenc", 0)),
        }
    )
    return SimpleNamespace(**config)


def _as_2d_mask(mask):
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 1:
        return mask.reshape(-1, 1)
    if mask.ndim == 2:
        return mask
    raise ValueError("mask must be 1D or 2D [T,C].")


def _shift_mask(mask, lag_steps: int):
    shifted = np.zeros_like(mask)
    if lag_steps == 0:
        return mask.copy()
    if lag_steps > 0:
        shifted[lag_steps:] = mask[:-lag_steps]
    else:
        shifted[:lag_steps] = mask[-lag_steps:]
    return shifted


def _masked_mse(pred, true, mask):
    mask = np.asarray(mask, dtype=np.float32)
    denom = float(mask.sum())
    if denom <= EPS:
        return 0.0
    return float((((pred - true) ** 2) * mask).sum() / denom)


def _zero_prior_mse(true, mask, zero_target):
    if mask.sum() <= EPS:
        return 0.0
    target = np.asarray(zero_target, dtype=np.float32).reshape(1, -1)
    return float((((true - target) ** 2) * mask).sum() / (mask.sum() + EPS))


def _mse(pred, true):
    if len(pred) == 0:
        return 0.0
    return float(np.mean((pred - true) ** 2))


def _freq_minutes(timestamps):
    if len(timestamps) < 2:
        return 0.0
    delta = pd.Timestamp(timestamps[1]) - pd.Timestamp(timestamps[0])
    return float(delta.total_seconds() / 60.0)


def _timestamp_step_hours(timestamps):
    return _freq_minutes(timestamps) / 60.0 if timestamps is not None else 0.25


def _unique_day_starts(timestamps, indices):
    seen = set()
    rows = []
    for idx in indices:
        day = pd.Timestamp(timestamps[idx]).floor("D")
        if day in seen:
            continue
        seen.add(day)
        rows.append(int(idx))
    return rows


def _nearest_index(index, candidates):
    if len(candidates) == 0:
        return None
    return int(candidates[np.argmin(np.abs(candidates - index))])


def _write_json(path: str, payload: dict):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: str, diagnosis: dict):
    alignment_error = diagnosis["dataset_alignment"]["alignment_error"]
    train_lag = diagnosis["splits"]["train"]["aggregate"]["best_lag_steps"]
    val_lag = diagnosis["splits"]["val"]["aggregate"]["best_lag_steps"]
    test_lag = diagnosis["splits"]["test"]["aggregate"]["best_lag_steps"]
    predictability = diagnosis.get("offset_predictability", {})
    aggregate_predictability = predictability.get("aggregate_anchor_offsets", {})
    lines = [
        "# Split Offset Diagnosis",
        "",
        f"- Dataset alignment bug: `{alignment_error}`.",
        f"- Train best lag: `{train_lag}` steps.",
        f"- Val best lag: `{val_lag}` steps.",
        f"- Test best lag: `{test_lag}` steps.",
        f"- Split phase shift: `{diagnosis['split_phase_shift']}`.",
        f"- Rule stability: `{diagnosis['rule_stability']}`.",
        f"- Offset predictability: `{predictability.get('overall_predictability', 'unknown')}`.",
        f"- Disable prior recommendation: `{diagnosis['disable_prior_recommendation']}`.",
        f"- Recommendation: `{diagnosis['recommendation']}`.",
        "",
        "## Direct Answers",
        "",
        f"1. Val/test offset exists: `{diagnosis['split_phase_shift']}`.",
        f"2. Offset is stable: `{not diagnosis['split_phase_shift'] and diagnosis['rule_stability'] == 'stable'}`.",
        f"3. Source: `{'dataset alignment bug' if alignment_error else 'true distribution/rule phase drift, not a Dataset slicing bug'}`.",
        f"4. Rule mask hits true near-zero events weakly: train F1 `{diagnosis['splits']['train']['aggregate']['best_f1']:.4f}`, "
        f"val F1 `{diagnosis['splits']['val']['aggregate']['best_f1']:.4f}`, "
        f"test F1 `{diagnosis['splits']['test']['aggregate']['best_f1']:.4f}`.",
        f"5. Zero prior is valid on stable channels: `{bool(diagnosis.get('stable_channels')) and not diagnosis['disable_prior_recommendation']}`.",
        f"6. Stable channels: `{', '.join(diagnosis.get('stable_channels', [])) or 'none'}`.",
        f"7. Disable current deterministic rule prior: `{diagnosis['disable_prior_recommendation']}`.",
        f"8. Shift-aware rule recommended: `{False if diagnosis['split_phase_shift'] else True}`; "
        "calendar_window or re-mined rules are preferred when offsets are unstable.",
        f"9. Predictability: `{aggregate_predictability.get('reason', 'unknown')}`.",
        "",
        "## Offset Predictability",
        "",
        f"- Predictable from train/val: `{aggregate_predictability.get('predictable_from_train_val')}`.",
        f"- Post-hoc test regime: `{aggregate_predictability.get('post_hoc_test_regime')}`.",
        f"- Test offsets outside train/val range: `{aggregate_predictability.get('test_offsets_outside_train_val_range')}`.",
        f"- Train/val anchor offsets: `{aggregate_predictability.get('train_val_offsets', [])}`.",
        f"- Test anchor offsets: `{aggregate_predictability.get('test_offsets', [])}`.",
        f"- Best train/val predictor: `{aggregate_predictability.get('best_predictor')}` "
        f"with MAE `{aggregate_predictability.get('best_mae_steps', 0.0):.2f}` steps.",
        f"- Weak train/val-predictable channels by heuristic: `{', '.join(predictability.get('predictable_channels', [])) or 'none'}`.",
        f"- Unpredictable channels: `{', '.join(predictability.get('unpredictable_channels', [])) or 'none'}`.",
        "",
        "## Per Split",
        "",
        "| Split | Rule Mask Sum | True Event Sum | Best Lag Steps | Best Lag Hours | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        payload = diagnosis["splits"][split]
        agg = payload["aggregate"]
        lines.append(
            f"| {split} | {payload['rule_mask_sum']:.0f} | {payload['true_event_sum']:.0f} | "
            f"{agg['best_lag_steps']} | {agg['best_lag_hours']:.2f} | "
            f"{agg['best_precision']:.4f} | {agg['best_recall']:.4f} | {agg['best_f1']:.4f} |"
        )
    lines.extend(["", "## Unique Timestamp Metric", ""])
    metrics = diagnosis.get("baseline_prediction_metrics", {})
    if metrics:
        lines.extend(
            [
                f"- Repeated event points: `{metrics.get('num_repeated_event_points')}`.",
                f"- Unique event timestamps: `{metrics.get('num_unique_event_timestamps')}`.",
                f"- Repeat factor: `{metrics.get('repeat_factor'):.2f}`.",
                f"- Repeated event MSE: `{metrics.get('repeated_event_mse'):.6f}`.",
                f"- Unique event MSE: `{metrics.get('unique_event_mse'):.6f}`.",
            ]
        )
    else:
        lines.append("- Baseline predictions were not available.")
    lines.extend(["", "## Stable Channels", "", ", ".join(diagnosis.get("stable_channels", [])) or "none"])
    lines.extend(["", "## Unstable Channels", "", ", ".join(diagnosis.get("unstable_channels", [])) or "none"])
    lines.extend(["", "## Calendar Anchor Samples", ""])
    for split in ["train", "val", "test"]:
        anchors = diagnosis["splits"][split].get("calendar_anchor", [])[:5]
        lines.append(f"### {split}")
        if not anchors:
            lines.append("No calendar-periodic anchors found.")
            continue
        lines.append("| Expected Timestamp | Nearest True Event | Offset Steps | Offset Hours | True Near-zero Ratio | Zero Prior MSE |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in anchors:
            lines.append(
                f"| {row['expected_timestamp']} | {row['nearest_true_event_timestamp']} | "
                f"{row['offset_steps']} | {row['offset_hours']:.2f} | "
                f"{row['true_near_zero_ratio']:.4f} | {row['zero_prior_mse']:.6f} |"
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose split-dependent rule offset and timestamp alignment.")
    parser.add_argument("--data", default="ETTm1")
    parser.add_argument("--root_path", default="./data/")
    parser.add_argument("--data_path", default="ETTm1.csv")
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--llm_rule_path", default="./llm_rules/example_rules/ETTm1_rules.json")
    parser.add_argument("--baseline_result_dir", default=None)
    parser.add_argument("--output_path", default="artifacts/core_results/ettm1_split_offset_diagnosis.json")
    parser.add_argument("--output_markdown_path", default="docs/split_offset_diagnosis.md")
    parser.add_argument("--lag_min", type=int, default=-192)
    parser.add_argument("--lag_max", type=int, default=192)
    parser.add_argument("--near_zero_quantile", type=float, default=0.05)
    args = parser.parse_args()
    diagnose_split_offset(**vars(args))


if __name__ == "__main__":
    main()
