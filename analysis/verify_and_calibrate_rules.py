"""Verify offline LLM rules on train/val and emit calibrated priors."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider
from llm_rules.rule_parser import parse_llm_rules
from models.model_factory import build_model


DEFAULT_ALPHA_GRID = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
PRIOR_TYPES = ["zero_target", "residual_mean", "residual_median", "ratio", "conditional_mean"]
EPS = 1e-8


def verify_and_calibrate_rules(
    data: str,
    root_path: str,
    data_path: str,
    features: str,
    target: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    llm_rule_path: str,
    baseline_checkpoint: str,
    baseline_result_dir: str | None = None,
    calibration_split: str = "val",
    output_rule_path: str = "./llm_rules/validated_rules/ETTm1_rules_calibrated.json",
    output_report_path: str = "./artifacts/core_results/ettm1_rule_calibration_report.json",
    min_prior_improvement: float = 0.01,
    min_event_precision: float = 0.5,
    near_zero_quantile: float = 0.05,
    calibration_objective: str = "near_zero_event",
    allowed_prior_types: str | list[str] | tuple[str, ...] | None = None,
    alpha_grid: list[float] | None = None,
):
    if calibration_split == "test":
        raise ValueError("test split must not be used for rule calibration.")
    run_args = _build_args(
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
    train_data, _ = data_provider(run_args, "train")
    calibration_data, calibration_loader = data_provider(run_args, calibration_split)
    pred, true, masks = _predict_split(run_args, calibration_loader, baseline_checkpoint)
    zero_mask = masks[:, :, 1, :]
    near_zero_thresholds = _train_near_zero_threshold(train_data, near_zero_quantile)
    actual_event_mask = _near_zero_window_mask(true, near_zero_thresholds, train_data.zero_target)
    rows = calibrate_channels(
        pred=pred,
        true=true,
        mask=zero_mask,
        channel_names=calibration_data.target_columns,
        zero_target=train_data.zero_target,
        alpha_grid=alpha_grid or DEFAULT_ALPHA_GRID,
        min_prior_improvement=min_prior_improvement,
        actual_event_mask=actual_event_mask,
        min_event_precision=min_event_precision,
        calibration_objective=calibration_objective,
        allowed_prior_types=allowed_prior_types,
    )
    source_rules = parse_llm_rules(llm_rule_path)
    false_positive_policy = {
        "actual_event_definition": "abs(y - train_zero_target) <= train_distance_quantile",
        "near_zero_quantile": near_zero_quantile,
        "min_event_precision": min_event_precision,
        "near_zero_threshold_by_channel": {
            channel: float(value) for channel, value in zip(calibration_data.target_columns, near_zero_thresholds)
        },
    }
    selected_prior_types = _parse_allowed_prior_types(allowed_prior_types)
    calibrated = _build_calibrated_rule_payload(
        source_rules=source_rules.to_dict() if source_rules is not None else {"dataset_name": data, "patterns": []},
        source_rule_path=llm_rule_path,
        calibration_split=calibration_split,
        channel_rows=rows,
        false_positive_policy=false_positive_policy,
        calibration_objective=calibration_objective,
        allowed_prior_types=selected_prior_types,
    )
    report = {
        "data": data,
        "source_rule_path": llm_rule_path,
        "baseline_checkpoint": baseline_checkpoint,
        "baseline_result_dir": baseline_result_dir,
        "calibration_split": calibration_split,
        "alpha_grid": alpha_grid or DEFAULT_ALPHA_GRID,
        "min_prior_improvement": min_prior_improvement,
        "calibration_objective": calibration_objective,
        "allowed_prior_types": selected_prior_types,
        "false_positive_policy": false_positive_policy,
        "num_windows": int(pred.shape[0]),
        "pred_shape": list(pred.shape),
        "zero_mask_sum": float(zero_mask.sum()),
        "channel_diagnostics": rows,
        "enabled_channels": _enabled_channels(rows),
        "selected_prior_types": sorted({row["best_prior_type"] for row in rows.values() if row["enabled"]}),
    }
    _write_json(output_rule_path, calibrated)
    _write_json(output_report_path, report)
    return {"calibrated_rules": calibrated, "report": report}


def calibrate_channels(
    pred: np.ndarray,
    true: np.ndarray,
    mask: np.ndarray,
    channel_names: list[str],
    zero_target: np.ndarray,
    alpha_grid: list[float] | tuple[float, ...] = DEFAULT_ALPHA_GRID,
    min_prior_improvement: float = 0.01,
    actual_event_mask: np.ndarray | None = None,
    min_event_precision: float = 0.0,
    calibration_objective: str = "near_zero_event",
    allowed_prior_types: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict]:
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    mask = _as_channel_mask(mask, pred.shape[-1])
    actual_event_mask = None if actual_event_mask is None else _as_channel_mask(actual_event_mask, pred.shape[-1])
    zero_target = np.asarray(zero_target, dtype=np.float32).reshape(-1)
    selected_prior_types = _parse_allowed_prior_types(allowed_prior_types)
    if calibration_objective not in {"near_zero_event", "rule_window_mse"}:
        raise ValueError("calibration_objective must be 'near_zero_event' or 'rule_window_mse'.")
    rows: dict[str, dict] = {}
    for channel_idx, channel_name in enumerate(channel_names):
        channel_mask = mask[:, :, channel_idx] > 0
        mask_count = int(channel_mask.sum())
        false_positive_metrics = _rule_mask_precision_metrics(
            channel_mask,
            None if actual_event_mask is None else actual_event_mask[:, :, channel_idx] > 0,
        )
        if mask_count <= 0:
            row = _disabled_row(channel_name, mask_count, "empty rule mask on calibration split")
            row.update(false_positive_metrics)
            rows[channel_name] = row
            continue
        base = pred[:, :, channel_idx][channel_mask]
        label = true[:, :, channel_idx][channel_mask]
        baseline_mse = _mse(base, label)
        candidates = {
            key: value
            for key, value in _prior_candidates(base, label, float(zero_target[channel_idx])).items()
            if key in selected_prior_types
        }
        if not candidates:
            row = _disabled_row(channel_name, mask_count, "no allowed prior types available")
            row.update(false_positive_metrics)
            rows[channel_name] = row
            continue
        metrics = {
            f"{prior_type}_mse": _mse(_apply_prior(base, prior_type, value), label)
            for prior_type, value in candidates.items()
        }
        best = {
            "prior_type": "baseline",
            "alpha": 0.0,
            "mse": baseline_mse,
            "prior_value": 0.0,
        }
        for prior_type, prior_value in candidates.items():
            for alpha in alpha_grid:
                fused = _apply_prior_alpha(base, prior_type, prior_value, float(alpha))
                mse = _mse(fused, label)
                if mse < best["mse"] - EPS:
                    best = {
                        "prior_type": prior_type,
                        "alpha": float(alpha),
                        "mse": mse,
                        "prior_value": float(prior_value),
                    }
        improvement_ratio = _safe_ratio(baseline_mse - best["mse"], baseline_mse)
        precision_gate_applied = (
            calibration_objective == "near_zero_event" and false_positive_metrics["precision_gate_applied"]
        )
        precision_gate_passed = (
            not precision_gate_applied or false_positive_metrics["event_precision"] >= float(min_event_precision)
        )
        enabled = (
            mask_count > 0
            and best["alpha"] > 0.0
            and baseline_mse > EPS
            and best["mse"] < baseline_mse * (1.0 - float(min_prior_improvement))
            and precision_gate_passed
        )
        row = {
            "channel_name": channel_name,
            "mask_count": mask_count,
            "baseline_mse": baseline_mse,
            **metrics,
            "candidate_best_prior_type": best["prior_type"],
            "candidate_best_alpha": best["alpha"],
            "candidate_best_prior_mse": best["mse"],
            "candidate_prior_value": best["prior_value"],
            "calibration_objective": calibration_objective,
            "allowed_prior_types": selected_prior_types,
            "best_prior_type": best["prior_type"] if enabled else "baseline",
            "best_alpha": best["alpha"] if enabled else 0.0,
            "best_prior_mse": best["mse"] if enabled else baseline_mse,
            "prior_value": best["prior_value"] if enabled else 0.0,
            "improvement_ratio": improvement_ratio if enabled else 0.0,
            "enabled": bool(enabled),
            **{**false_positive_metrics, "precision_gate_applied": precision_gate_applied},
        }
        if not enabled:
            if not precision_gate_passed:
                row["disable_reason"] = "rule mask precision below min_event_precision on calibration split"
            else:
                row["disable_reason"] = "no prior improves baseline on calibration split"
        rows[channel_name] = row
    return rows


def _prior_candidates(base: np.ndarray, label: np.ndarray, zero_target: float) -> dict[str, float]:
    residual = label - base
    return {
        "zero_target": float(zero_target),
        "residual_mean": float(residual.mean()),
        "residual_median": float(np.median(residual)),
        "ratio": float(np.mean(label / (base + np.where(np.abs(base) < EPS, EPS, 0.0)))),
        "conditional_mean": float(label.mean()),
    }


def _apply_prior_alpha(base: np.ndarray, prior_type: str, prior_value: float, alpha: float) -> np.ndarray:
    prior = _apply_prior(base, prior_type, prior_value)
    if prior_type in {"residual_mean", "residual_median"}:
        return base + alpha * prior_value
    return base + alpha * (prior - base)


def _apply_prior(base: np.ndarray, prior_type: str, prior_value: float) -> np.ndarray:
    if prior_type == "zero_target":
        return np.full_like(base, prior_value)
    if prior_type in {"residual_mean", "residual_median"}:
        return base + prior_value
    if prior_type == "ratio":
        return base * prior_value
    if prior_type == "conditional_mean":
        return np.full_like(base, prior_value)
    raise ValueError(f"Unsupported prior_type: {prior_type}")


def _predict_split(args: SimpleNamespace, loader, checkpoint_path: str):
    device = torch.device("cuda:0" if bool(getattr(args, "use_gpu", 0)) and torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(_model_state_dict(state), strict=False)
    model.eval()
    preds, trues, masks_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            seq_x, seq_y, *_rest, seq_y_masks = batch
            seq_x = seq_x.float().to(device)
            seq_y = seq_y.float().to(device)
            pred = model(seq_x)
            true = seq_y[:, -int(args.pred_len) :, : int(args.c_out)]
            masks = seq_y_masks[:, -int(args.pred_len) :, :]
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            masks_all.append(masks.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0), np.concatenate(masks_all, axis=0)


def _model_state_dict(state: dict) -> dict:
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if all(str(key).startswith("model.") for key in state):
        return {str(key)[6:]: value for key, value in state.items()}
    return {str(key)[6:]: value for key, value in state.items() if str(key).startswith("model.")} or state


def _build_calibrated_rule_payload(
    source_rules: dict,
    source_rule_path: str,
    calibration_split: str,
    channel_rows: dict[str, dict],
    false_positive_policy: dict | None = None,
    calibration_objective: str = "near_zero_event",
    allowed_prior_types: list[str] | None = None,
) -> dict:
    patterns = []
    valid_channels = _enabled_channels(channel_rows)
    disabled_channels = {
        channel: row.get("disable_reason", "disabled by calibration")
        for channel, row in channel_rows.items()
        if not row.get("enabled", False)
    }
    for pattern in source_rules.get("patterns", []):
        cloned = copy.deepcopy(pattern)
        cloned["enabled"] = bool(valid_channels)
        cloned["original_effect_type"] = "zero_target" if cloned.get("type") == "zero_event" else cloned.get("type")
        cloned["selected_prior_type"] = _selected_prior_summary(channel_rows)
        cloned["calibrated_alpha"] = {
            channel: row["best_alpha"] for channel, row in channel_rows.items() if row.get("enabled", False)
        }
        cloned["false_positive_policy"] = false_positive_policy or {}
        cloned["calibration_objective"] = calibration_objective
        cloned["allowed_prior_types"] = allowed_prior_types or PRIOR_TYPES
        cloned["valid_channels"] = valid_channels
        cloned["channel_diagnostics"] = channel_rows
        cloned["disabled_channels"] = disabled_channels
        patterns.append(cloned)
    return {
        "dataset_name": source_rules.get("dataset_name", "unknown"),
        "source_rule_path": source_rule_path,
        "calibration_split": calibration_split,
        "false_positive_policy": false_positive_policy or {},
        "calibration_objective": calibration_objective,
        "allowed_prior_types": allowed_prior_types or PRIOR_TYPES,
        "patterns": patterns,
    }


def _selected_prior_summary(channel_rows: dict[str, dict]) -> str:
    selected = [row["best_prior_type"] for row in channel_rows.values() if row.get("enabled", False)]
    if not selected:
        return "disabled"
    counts = {prior: selected.count(prior) for prior in sorted(set(selected))}
    return max(counts, key=counts.get)


def _parse_allowed_prior_types(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return list(PRIOR_TYPES)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    unsupported = [item for item in items if item not in PRIOR_TYPES]
    if unsupported:
        raise ValueError(f"Unsupported prior types: {unsupported}. Supported: {PRIOR_TYPES}")
    if not items:
        raise ValueError("allowed_prior_types cannot be empty.")
    return items


def _enabled_channels(channel_rows: dict[str, dict]) -> list[str]:
    return [channel for channel, row in channel_rows.items() if row.get("enabled", False)]


def _disabled_row(channel_name: str, mask_count: int, reason: str) -> dict:
    return {
        "channel_name": channel_name,
        "mask_count": mask_count,
        "baseline_mse": 0.0,
        "zero_target_mse": 0.0,
        "residual_mean_mse": 0.0,
        "residual_median_mse": 0.0,
        "ratio_mse": 0.0,
        "conditional_mean_mse": 0.0,
        "candidate_best_prior_type": "baseline",
        "candidate_best_alpha": 0.0,
        "candidate_best_prior_mse": 0.0,
        "candidate_prior_value": 0.0,
        "best_prior_type": "baseline",
        "best_alpha": 0.0,
        "best_prior_mse": 0.0,
        "prior_value": 0.0,
        "improvement_ratio": 0.0,
        "enabled": False,
        "disable_reason": reason,
        "precision_gate_applied": False,
        "event_precision": 0.0,
        "event_recall": 0.0,
        "false_positive_ratio": 0.0,
        "event_tp": 0,
        "event_fp": 0,
        "event_fn": 0,
        "event_tn": 0,
    }


def _build_args(**kwargs) -> SimpleNamespace:
    config = {}
    baseline_result_dir = kwargs.get("baseline_result_dir")
    if baseline_result_dir:
        config_path = Path(baseline_result_dir) / "config.json"
        if config_path.exists():
            config.update(json.loads(config_path.read_text(encoding="utf-8")))
    config.update({key: value for key, value in kwargs.items() if value is not None})
    config.update(
        {
            "model": config.get("model", "DLinear"),
            "root_path": config.get("root_path", "./data/"),
            "batch_size": int(config.get("batch_size", 32)),
            "num_workers": 0,
            "use_zscore": int(config.get("use_zscore", 1)),
            "use_revin": int(config.get("use_revin", 0)),
            "use_llm_features": 0,
            "use_llm_rule_features": 0,
            "use_standard_time_features": 0,
            "use_oracle_features": 0,
            "use_rule_prior_fusion": 0,
            "use_intervention_layer": 0,
            "use_hard_intervention": 0,
            "use_gpu": int(config.get("use_gpu", 0)),
            "dlinear_init_avg": int(config.get("dlinear_init_avg", 0)),
            "individual": int(config.get("individual", 0)),
            "moving_avg": int(config.get("moving_avg", 25)),
            "enc_in": int(config.get("enc_in", 7)),
            "c_out": int(config.get("c_out", 7)),
            "freq": config.get("freq", "h"),
            "timeenc": int(config.get("timeenc", 0)),
        }
    )
    return SimpleNamespace(**config)


def _as_channel_mask(mask: np.ndarray, channels: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 2:
        mask = mask[:, :, None]
    if mask.shape[-1] == 1 and channels > 1:
        return np.repeat(mask, channels, axis=-1)
    if mask.shape[-1] != channels:
        raise ValueError(f"mask channel count {mask.shape[-1]} does not match predictions {channels}.")
    return mask


def _train_near_zero_threshold(train_data, quantile: float) -> np.ndarray:
    zero_target = np.asarray(train_data.zero_target, dtype=np.float32).reshape(1, -1)
    distance = np.abs(train_data.data_y - zero_target)
    return np.quantile(distance, float(quantile), axis=0).astype(np.float32)


def _near_zero_window_mask(values: np.ndarray, thresholds: np.ndarray, zero_target: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, 1, -1)
    zero_target = np.asarray(zero_target, dtype=np.float32).reshape(1, 1, -1)
    return (np.abs(values - zero_target) <= thresholds).astype(np.float32)


def _rule_mask_precision_metrics(predicted_mask: np.ndarray, actual_event_mask: np.ndarray | None) -> dict:
    if actual_event_mask is None:
        return {
            "precision_gate_applied": False,
            "event_precision": 1.0,
            "event_recall": 0.0,
            "false_positive_ratio": 0.0,
            "event_tp": 0,
            "event_fp": 0,
            "event_fn": 0,
            "event_tn": 0,
        }
    predicted = np.asarray(predicted_mask) > 0
    actual = np.asarray(actual_event_mask) > 0
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, np.logical_not(actual)).sum())
    fn = int(np.logical_and(np.logical_not(predicted), actual).sum())
    tn = int(np.logical_and(np.logical_not(predicted), np.logical_not(actual)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "precision_gate_applied": True,
        "event_precision": precision,
        "event_recall": recall,
        "false_positive_ratio": fp / (tp + fp) if tp + fp > 0 else 0.0,
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "event_tn": tn,
    }


def _mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= EPS:
        return 0.0
    return float(numerator / denominator)


def _write_json(path: str, payload: dict):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Verify and calibrate offline LLM rule priors without using test.")
    parser.add_argument("--data", default="ETTm1")
    parser.add_argument("--root_path", default="./data/")
    parser.add_argument("--data_path", default="ETTm1.csv")
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--llm_rule_path", default="./llm_rules/example_rules/ETTm1_rules.json")
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--baseline_result_dir", default=None)
    parser.add_argument("--calibration_split", default="val", choices=["train", "val"])
    parser.add_argument("--output_rule_path", default="./llm_rules/validated_rules/ETTm1_rules_calibrated.json")
    parser.add_argument("--output_report_path", default="./artifacts/core_results/ettm1_rule_calibration_report.json")
    parser.add_argument("--min_prior_improvement", type=float, default=0.01)
    parser.add_argument("--min_event_precision", type=float, default=0.5)
    parser.add_argument("--near_zero_quantile", type=float, default=0.05)
    parser.add_argument("--calibration_objective", default="near_zero_event", choices=["near_zero_event", "rule_window_mse"])
    parser.add_argument("--allowed_prior_types", default=None)
    args = parser.parse_args()
    verify_and_calibrate_rules(**vars(args))


if __name__ == "__main__":
    main()
