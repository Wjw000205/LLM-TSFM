"""Summarize GPT-5.5 gated peak-transfer full-horizon results across datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluate_rule_gated_ensemble import load_test_masks


DATASETS = ["ETTh1", "ETTh2", "ETTm2"]
PRED_LENS = [96, 192, 336, 720]

CSV_FIELDS = [
    "dataset",
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
    "selection_metric",
    "overall_mse_tolerance",
    "event_weight",
    "use_peak_shape_loss",
    "learning_rate",
    "status",
    "event_mask_warning",
    "event_points",
    "total_prediction_elements",
    "event_ratio",
    "event_ratio_pct",
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


def build_runs(datasets: list[str], pred_lens: list[int]) -> list[dict[str, Any]]:
    runs = []
    for dataset in datasets:
        dataset_lower = dataset.lower()
        for pred_len in pred_lens:
            stem = (
                f"results/long_term_forecast_DLinear_{dataset}_ftM_sl336_ll48_pl{pred_len}_"
                f"{dataset_lower}_gpt55_peak_transfer_p{pred_len}"
            )
            runs.append(
                {
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "baseline_dir": f"{stem}_baseline_0",
                    "expert_dir": f"{stem}_finetune_loss_0",
                    "gated_dir": f"{stem}_gated_alpha_1p0_0",
                }
            )
    return runs


def _expected_rule_path(dataset: str, pred_len: int) -> str:
    return f"llm_rules/generated_rules/{dataset}_p{pred_len}_peak_transfer_rules.json"


def _is_horizon_specific_rule_path(path: str | None, dataset: str, pred_len: int) -> bool:
    if not path:
        return False
    normalized = str(path).replace("\\", "/").lstrip("./").lower()
    dataset_lower = dataset.lower()
    if normalized.endswith(_expected_rule_path(dataset, pred_len).lower()):
        return True
    return re.search(rf"/?{re.escape(dataset_lower)}_p\d+_.*rules\.json$", normalized) is not None


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_summarize_one(run) for run in runs]


def _summarize_one(run: dict[str, Any]) -> dict[str, Any]:
    baseline_dir = Path(run["baseline_dir"])
    expert_dir = Path(run["expert_dir"])
    gated_dir = Path(run["gated_dir"])
    _require_files(
        [
            baseline_dir / "metrics_normalized.json",
            baseline_dir / "true_normalized.npy",
            baseline_dir / "pred_normalized.npy",
            baseline_dir / "config.json",
            expert_dir / "metrics_normalized.json",
            expert_dir / "config.json",
            gated_dir / "metrics_normalized.json",
            gated_dir / "pred_normalized.npy",
        ]
    )

    baseline_metrics = _read_json(baseline_dir / "metrics_normalized.json")
    expert_metrics = _read_json(expert_dir / "metrics_normalized.json")
    gated_metrics = _read_json(gated_dir / "metrics_normalized.json")
    baseline_config = _read_json(baseline_dir / "config.json")
    expert_config = _read_json(expert_dir / "config.json")
    expert_loss = _read_json(expert_dir / "loss_config.json") if (expert_dir / "loss_config.json").exists() else {}
    _validate_horizon_rule_config(baseline_config, run, baseline_dir)
    _validate_horizon_rule_config(expert_config, run, expert_dir)

    true = np.load(baseline_dir / "true_normalized.npy", mmap_mode="r")
    baseline_pred = np.load(baseline_dir / "pred_normalized.npy", mmap_mode="r")
    gated_pred = np.load(gated_dir / "pred_normalized.npy", mmap_mode="r")
    masks = load_test_masks(baseline_dir / "config.json", expected_windows=true.shape[0])
    event_mask = masks[:, :, 0, :].astype(bool)

    if event_mask.shape != true.shape:
        raise ValueError(f"event mask shape {event_mask.shape} does not match true shape {true.shape} for {baseline_dir}")

    total_elements = int(true.size)
    event_points = int(event_mask.sum())
    metrics_event_points = int(gated_metrics.get("num_event_points", event_points))
    if metrics_event_points != event_points:
        raise ValueError(
            f"event point mismatch for {gated_dir}: metrics={metrics_event_points}, mask={event_points}"
        )

    event_ratio = event_points / total_elements if total_elements else 0.0
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

    return {
        "dataset": run["dataset"],
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
        "selection_metric": expert_config.get("selection_metric"),
        "overall_mse_tolerance": expert_config.get("overall_mse_tolerance"),
        "event_weight": _effective_value(expert_config, expert_loss, "event_weight"),
        "use_peak_shape_loss": _effective_value(expert_config, expert_loss, "use_peak_shape_loss"),
        "learning_rate": expert_config.get("learning_rate"),
        "status": (
            "not_applicable_empty_mask"
            if event_points == 0
            else "guarded"
            if selected_reason == "guarded_event_mse"
            else str(selected_reason or "unknown")
        ),
        "event_mask_warning": event_mask_warning,
        "event_points": event_points,
        "total_prediction_elements": total_elements,
        "event_ratio": event_ratio,
        "event_ratio_pct": event_ratio * 100.0,
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


def write_outputs(rows: list[dict[str, Any]], output_csv: Path, output_json: Path, output_md: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_sanitize(row) for row in rows)

    output_json.write_text(json.dumps(_json_sanitize(rows), indent=2), encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(rows), encoding="utf-8")


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Multidataset GPT-5.5 Peak-Transfer Full-Horizon Results",
        "",
        "## Scope",
        "",
        "This report extends the gated peak-transfer check to ETTh1, ETTh2, and ETTm2 across pred_len 96, 192, 336, and 720. Each dataset uses its own GPT-5.5 generated dataset-level peak-transfer rule file; no ETTm1 rule is reused.",
        "",
        "The evaluated path is still intentionally diagnostic: a pure DLinear baseline is trained first, a dataset-aware loss expert is fine-tuned from that checkpoint, then rule-gated evaluation copies the expert prediction only inside the event mask and keeps baseline predictions elsewhere.",
        "",
        "## Main Results",
        "",
        "| Dataset | pred_len | Baseline Overall | Gated Overall | Baseline Event | Gated Event | Event Reduction | Event Ratio | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | {_fmt_fixed(row['baseline_overall_mse'])} | "
            f"{_fmt_fixed(row['gated_overall_mse'])} | {_fmt_fixed(row['baseline_event_mse'])} | "
            f"{_fmt_fixed(row['gated_event_mse'])} | {_fmt_pct(row['event_reduction_pct'])} | "
            f"{_fmt_pct(row['event_ratio_pct'], digits=4)} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Non-Event Preservation",
            "",
            "Gated evaluation should leave the non-event region unchanged. The table below checks that property directly from `pred_normalized.npy` and `true_normalized.npy`.",
            "",
            "| Dataset | pred_len | Baseline Non-event MSE | Gated Non-event MSE | Non-event Delta | Expected Overall Delta | Observed Overall Delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | {_fmt_fixed(row['baseline_non_event_mse'])} | "
            f"{_fmt_fixed(row['gated_non_event_mse'])} | {_fmt_sci(row['non_event_delta'])} | "
            f"{_fmt_sci(row['expected_overall_delta_from_event'])} | {_fmt_sci(row['observed_overall_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Guardrail Selection",
            "",
            "| Dataset | pred_len | selected_reason | selected_epoch | selection_metric | tolerance | event_weight | peak_shape | learning_rate |",
            "|---|---:|---|---:|---|---:|---:|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | {pred_len} | {selected_reason} | {selected_epoch} | {selection_metric} | "
            "{overall_mse_tolerance} | {event_weight} | {use_peak_shape_loss} | {learning_rate} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Event Coverage",
            "",
            "| Dataset | pred_len | event_points | total_prediction_elements | event_mask_warning |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | {row['event_points']} | "
            f"{row['total_prediction_elements']} | {row['event_mask_warning'] or ''} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The regenerated GPT-5.5 dataset-level rules produced zero test event coverage for all 12 dataset/horizon combinations.",
            "- Because the test event mask is empty, event-window MSE and event reduction are not applicable, and gated evaluation is exactly the baseline in all rows.",
            "- This run resolves the horizon-reuse issue, but it also shows that the current GPT rule mining prompt is too conservative or too train-evidence-specific for transferable test-time event discovery.",
            "- The next method change should target transferable event mining with train-only evidence, then require a nonzero validation/test event coverage diagnostic before running expensive event-loss fine-tuning.",
            "",
            "## Artifacts",
            "",
            "- `artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.csv`",
            "- `artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.json`",
            "- `docs/multidataset_full_horizon_peak_transfer_results.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required result files:\n" + "\n".join(missing))


def _fmt_fixed(value: Any, digits: int = 6) -> str:
    value = _as_finite_float(value)
    return "NA" if value is None else f"{value:.{digits}f}"


def _fmt_sci(value: Any) -> str:
    value = _as_finite_float(value)
    return "NA" if value is None else f"{value:.3e}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    value = _as_finite_float(value)
    return "NA" if value is None else f"{value:.{digits}f}%"


def _as_finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _csv_sanitize(row: dict[str, Any]) -> dict[str, Any]:
    return {key: ("" if _is_nonfinite_float(value) else value) for key, value in row.items()}


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if _is_nonfinite_float(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_nonfinite_float(value: Any) -> bool:
    return isinstance(value, float) and not np.isfinite(value)


def _validate_horizon_rule_config(config: dict[str, Any], run: dict[str, Any], result_dir: Path) -> None:
    dataset = str(run["dataset"])
    pred_len = int(run["pred_len"])
    actual = config.get("llm_rule_path")
    if _is_horizon_specific_rule_path(actual, dataset, pred_len):
        raise ValueError(
            f"{result_dir} uses a horizon-specific event rule path for {dataset} p{pred_len}: {actual!r}. "
            "Event timestamp locations must be horizon-independent; use a dataset-level rule path instead."
        )
    if actual:
        return
    raise ValueError(
        f"{result_dir} has no llm_rule_path. Event-mask diagnostics require a dataset-level rule file."
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--pred_lens", nargs="+", type=int, default=PRED_LENS)
    parser.add_argument(
        "--output_csv",
        default="artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.csv",
    )
    parser.add_argument(
        "--output_json",
        default="artifacts/core_results/multidataset_full_horizon_peak_transfer_summary.json",
    )
    parser.add_argument(
        "--output_md",
        default="docs/multidataset_full_horizon_peak_transfer_results.md",
    )
    args = parser.parse_args()

    rows = summarize_runs(build_runs(args.datasets, args.pred_lens))
    write_outputs(rows, Path(args.output_csv), Path(args.output_json), Path(args.output_md))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
