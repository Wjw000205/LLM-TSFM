"""Summarize ETTm1 GPT-5.5 gated peak-transfer multi-horizon results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluate_rule_gated_ensemble import load_test_masks


DEFAULT_RUNS = [
    {
        "pred_len": 96,
        "baseline_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_myfram_gpt55_peak_only_baseline_0",
        "expert_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_myfram_gpt55_peak_only_finetune_generated_loss_guarded_0",
        "gated_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_myfram_gpt55_peak_only_gated_alpha_1p0_0",
    },
    {
        "pred_len": 192,
        "baseline_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl192_ettm1_gpt55_peak_transfer_p192_baseline_0",
        "expert_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl192_ettm1_gpt55_peak_transfer_p192_finetune_loss_0",
        "gated_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl192_ettm1_gpt55_peak_transfer_p192_gated_alpha_1p0_0",
    },
    {
        "pred_len": 336,
        "baseline_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl336_ettm1_gpt55_peak_transfer_p336_baseline_0",
        "expert_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl336_ettm1_gpt55_peak_transfer_p336_strict_ew1_nopk_lr1e6_0",
        "gated_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl336_ettm1_gpt55_peak_transfer_p336_strict_gated_ew1_nopk_lr1e6_0",
    },
    {
        "pred_len": 720,
        "baseline_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl720_ettm1_gpt55_peak_transfer_p720_baseline_0",
        "expert_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl720_ettm1_gpt55_peak_transfer_p720_strict_ew1_nopk_lr5e7_0",
        "gated_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl720_ettm1_gpt55_peak_transfer_p720_strict_gated_ew1_nopk_lr5e7_0",
    },
]


CSV_FIELDS = [
    "pred_len",
    "baseline_overall_mse",
    "expert_overall_mse",
    "gated_overall_mse",
    "baseline_event_mse",
    "expert_event_mse",
    "gated_event_mse",
    "event_reduction_pct",
    "overall_delta",
    "overall_delta_pct",
    "selected_reason",
    "selected_epoch",
    "event_weight",
    "use_peak_shape_loss",
    "selection_metric",
    "overall_mse_tolerance",
    "learning_rate",
    "status",
    "event_mask_warning",
    "event_points",
    "total_prediction_elements",
    "event_ratio",
    "baseline_non_event_mse",
    "gated_non_event_mse",
    "non_event_delta",
    "non_event_delta_pct",
    "expected_overall_delta_from_event",
    "observed_overall_delta",
    "delta_match_error",
    "baseline_dir",
    "expert_dir",
    "gated_dir",
]


def summarize_runs(runs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for run in runs or DEFAULT_RUNS:
        rows.append(_summarize_one(run))
    return rows


def _summarize_one(run: dict[str, Any]) -> dict[str, Any]:
    baseline_dir = Path(run["baseline_dir"])
    expert_dir = Path(run["expert_dir"])
    gated_dir = Path(run["gated_dir"])

    baseline_metrics = _read_json(baseline_dir / "metrics_normalized.json")
    expert_metrics = _read_json(expert_dir / "metrics_normalized.json")
    gated_metrics = _read_json(gated_dir / "metrics_normalized.json")
    expert_config = _read_json(expert_dir / "config.json")
    expert_loss = _read_json(expert_dir / "loss_config.json") if (expert_dir / "loss_config.json").exists() else {}

    true = np.load(baseline_dir / "true_normalized.npy", mmap_mode="r")
    baseline_pred = np.load(baseline_dir / "pred_normalized.npy", mmap_mode="r")
    gated_pred = np.load(gated_dir / "pred_normalized.npy", mmap_mode="r")
    masks = load_test_masks(baseline_dir / "config.json", expected_windows=true.shape[0])
    event_mask = masks[:, :, 0, :].astype(bool)

    total_elements = int(true.size)
    event_points = int(event_mask.sum())
    event_ratio = event_points / total_elements
    non_event_mask = ~event_mask
    baseline_non_event_mse = _masked_mse(baseline_pred, true, non_event_mask)
    gated_non_event_mse = _masked_mse(gated_pred, true, non_event_mask)

    baseline_overall = float(baseline_metrics["mse"])
    expert_overall = float(expert_metrics["mse"])
    gated_overall = float(gated_metrics["mse"])
    baseline_event = _metric_float(baseline_metrics, "event_window_mse")
    expert_event = _metric_float(expert_metrics, "event_window_mse")
    gated_event = _metric_float(gated_metrics, "event_window_mse")
    event_mask_warning = ""
    if event_points == 0:
        baseline_event = float("nan")
        expert_event = float("nan")
        gated_event = float("nan")
        event_mask_warning = "empty_event_mask"
    observed_delta = gated_overall - baseline_overall
    expected_delta = event_ratio * (gated_event - baseline_event)
    non_event_delta = gated_non_event_mse - baseline_non_event_mse

    selected_reason = expert_config.get("selected_reason")
    status = (
        "not_applicable_empty_mask"
        if event_points == 0
        else "guarded"
        if selected_reason == "guarded_event_mse"
        else str(selected_reason or "unknown")
    )
    return {
        "pred_len": int(run["pred_len"]),
        "baseline_overall_mse": baseline_overall,
        "expert_overall_mse": expert_overall,
        "gated_overall_mse": gated_overall,
        "baseline_event_mse": baseline_event,
        "expert_event_mse": expert_event,
        "gated_event_mse": gated_event,
        "event_reduction_pct": _pct_reduction(baseline_event, gated_event),
        "overall_delta": observed_delta,
        "overall_delta_pct": _pct_delta(baseline_overall, gated_overall),
        "selected_reason": selected_reason,
        "selected_epoch": expert_config.get("selected_epoch"),
        "event_weight": _effective_value(expert_config, expert_loss, "event_weight"),
        "use_peak_shape_loss": _effective_value(expert_config, expert_loss, "use_peak_shape_loss"),
        "selection_metric": expert_config.get("selection_metric"),
        "overall_mse_tolerance": expert_config.get("overall_mse_tolerance"),
        "learning_rate": expert_config.get("learning_rate"),
        "status": status,
        "event_mask_warning": event_mask_warning,
        "event_points": event_points,
        "total_prediction_elements": total_elements,
        "event_ratio": event_ratio,
        "baseline_non_event_mse": baseline_non_event_mse,
        "gated_non_event_mse": gated_non_event_mse,
        "non_event_delta": non_event_delta,
        "non_event_delta_pct": _pct_delta(baseline_non_event_mse, gated_non_event_mse),
        "expected_overall_delta_from_event": expected_delta,
        "observed_overall_delta": observed_delta,
        "delta_match_error": observed_delta - expected_delta,
        "baseline_dir": str(baseline_dir),
        "expert_dir": str(expert_dir),
        "gated_dir": str(gated_dir),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _masked_mse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    denom = int(mask.sum())
    if denom == 0:
        return float("nan")
    diff = np.asarray(pred - true)
    return float(np.square(diff)[mask].mean())


def _metric_float(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return float("nan")
    return float(value)


def _effective_value(config: dict[str, Any], loss: dict[str, Any], key: str):
    value = config.get(key)
    return loss.get(key) if value is None else value


def _pct_reduction(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


def _pct_delta(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return (candidate - baseline) / baseline * 100.0


def write_summary(rows: list[dict[str, Any]], output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ETTm1 gated peak-transfer multi-horizon results.")
    parser.add_argument(
        "--output_csv",
        default="artifacts/core_results/ettm1_gpt55_peak_transfer_multihorizon_summary.csv",
    )
    parser.add_argument(
        "--output_json",
        default="artifacts/core_results/ettm1_gpt55_peak_transfer_multihorizon_summary.json",
    )
    args = parser.parse_args()
    rows = summarize_runs()
    write_summary(rows, Path(args.output_csv), Path(args.output_json))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
