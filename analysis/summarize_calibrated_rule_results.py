"""Summarize calibrated rule-prior experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider


def summarize_calibrated_rule_results(
    results_root: str = "./results",
    filter: str = "ettm1_calibrated",
    baseline_experiment: str = "ettm1_calibrated_pure_dlinear",
    calibration_report_path: str = "artifacts/core_results/ettm1_rule_calibration_report.json",
    output_markdown: str = "docs/calibrated_rule_prior_results.md",
    output_csv: str = "artifacts/core_results/ettm1_calibrated_rule_prior_summary.csv",
):
    report = _read_json(Path(calibration_report_path)) if Path(calibration_report_path).exists() else {}
    rows = []
    for result_dir in sorted(Path(results_root).glob(f"*{filter}*")):
        metrics_path = result_dir / "metrics_normalized.json"
        config_path = result_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = _read_json(metrics_path)
        config = _read_json(config_path)
        non_event = _non_event_metrics(result_dir, config)
        rows.append(
            {
                "Experiment": config.get("des", result_dir.name),
                "Overall MSE": float(metrics.get("mse", 0.0)),
                "Non-event MSE": non_event["non_event_mse"],
                "Event MSE": float(metrics.get("event_window_mse", 0.0)),
                "Zero MSE": float(metrics.get("zero_event_mse", 0.0)),
                "Rule Score": float(metrics.get("rule_consistency_score", 0.0)),
                "Overall Delta": 0.0,
                "Event Reduction": 0.0,
                "Enabled Channels": ", ".join(report.get("enabled_channels", []))
                if int(config.get("use_rule_prior_fusion", 0)) and config.get("rule_prior_mode") == "calibrated"
                else "",
                "Selected Prior Types": ", ".join(report.get("selected_prior_types", []))
                if int(config.get("use_rule_prior_fusion", 0)) and config.get("rule_prior_mode") == "calibrated"
                else "",
                "Notes": _notes(config),
                "Path": str(result_dir),
            }
        )
    baseline = _find_baseline(rows, baseline_experiment)
    if baseline is not None:
        baseline_mse = baseline["Overall MSE"]
        baseline_event = baseline["Event MSE"]
        for row in rows:
            row["Overall Delta"] = _safe_ratio(row["Overall MSE"] - baseline_mse, baseline_mse)
            row["Event Reduction"] = _safe_ratio(baseline_event - row["Event MSE"], baseline_event)
    _write_csv(output_csv, rows)
    Path(output_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(output_markdown).write_text(_markdown(rows, baseline_experiment, report), encoding="utf-8")
    return {"rows": rows, "output_csv": output_csv, "output_markdown": output_markdown}


def _fieldnames():
    return [
        "Experiment",
        "Overall MSE",
        "Non-event MSE",
        "Event MSE",
        "Zero MSE",
        "Rule Score",
        "Overall Delta",
        "Event Reduction",
        "Enabled Channels",
        "Selected Prior Types",
        "Notes",
        "Path",
    ]


def _write_csv(path: str, rows: list[dict]):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: list[dict], baseline_experiment: str, report: dict) -> str:
    lines = [
        "# Calibrated Rule Prior Results",
        "",
        f"Baseline experiment: `{baseline_experiment}`",
        "",
        "| Experiment | Overall MSE | Non-event MSE | Event MSE | Zero MSE | Rule Score | Overall Delta | Event Reduction | Enabled Channels | Selected Prior Types | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {Experiment} | {Overall MSE:.6f} | {Non-event MSE:.6f} | {Event MSE:.6f} | {Zero MSE:.6f} | "
            "{Rule Score:.6f} | {Overall Delta:.2%} | {Event Reduction:.2%} | {Enabled Channels} | "
            "{Selected Prior Types} | {Notes} |".format(**row)
        )
    if not rows:
        lines.append("| No matching completed runs | 0 | 0 | 0 | 0 | 0 | 0% | 0% |  |  | Run script first |")
    lines.extend(["", "## Calibration Summary", ""])
    lines.append(f"- Calibration split: `{report.get('calibration_split', 'unknown')}`.")
    lines.append(f"- Enabled channels: `{', '.join(report.get('enabled_channels', [])) or 'none'}`.")
    lines.append(f"- Selected prior types: `{', '.join(report.get('selected_prior_types', [])) or 'none'}`.")
    if report.get("channel_diagnostics"):
        for channel, diag in report["channel_diagnostics"].items():
            status = "enabled" if diag.get("enabled") else f"disabled: {diag.get('disable_reason')}"
            lines.append(
                f"- {channel}: {status}; baseline_mse={diag.get('baseline_mse', 0.0):.6f}, "
                f"best_prior_mse={diag.get('best_prior_mse', 0.0):.6f}, "
                f"best_prior={diag.get('best_prior_type')}, alpha={diag.get('best_alpha', 0.0)}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Calibrated priors are selected only on train/val; test results are report-only.",
            "- If no channel is enabled, the calibrated rule is treated as a rejected hypothesis.",
            "- Hard intervention remains a diagnostic, not an oracle, unless the calibrated prior validity is proven.",
        ]
    )
    return "\n".join(lines) + "\n"


def _non_event_metrics(result_dir: Path, config: dict) -> dict[str, float]:
    pred_path = result_dir / "pred_normalized.npy"
    true_path = result_dir / "true_normalized.npy"
    if not pred_path.exists() or not true_path.exists():
        return {"non_event_mse": 0.0}
    pred = np.load(pred_path)
    true = np.load(true_path)
    masks = _test_masks(config)[: pred.shape[0]]
    event_mask = masks[:, :, 0, :]
    non_event = 1.0 - event_mask
    denom = non_event.sum()
    if denom <= 1e-8:
        return {"non_event_mse": 0.0}
    return {"non_event_mse": float((((pred - true) ** 2) * non_event).sum() / denom)}


def _test_masks(config: dict):
    args = SimpleNamespace(**config)
    args.num_workers = 0
    data_provider(args, "train")
    test_data, _ = data_provider(args, "test")
    return np.asarray([test_data[idx][-1].numpy()[-test_data.pred_len :] for idx in range(len(test_data))])


def _notes(config: dict) -> str:
    notes = []
    if int(config.get("use_rule_prior_fusion", 0)):
        mode = config.get("rule_prior_mode", "fixed")
        notes.append(f"{mode}_rule_prior")
    if int(config.get("use_dataset_aware_loss", 0)):
        notes.append("dataset_aware_loss")
    if int(config.get("use_nonevent_preservation_loss", 0) or 0):
        notes.append("guarded_longtail")
    if int(config.get("use_hard_intervention", 0)):
        notes.append("hard_intervention_diagnostic")
    return ", ".join(notes) if notes else "pure_dlinear"


def _find_baseline(rows: list[dict], baseline_experiment: str) -> dict | None:
    for row in rows:
        if row["Experiment"] == baseline_experiment:
            return row
    return rows[0] if rows else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Summarize calibrated rule-prior experiment folders.")
    parser.add_argument("--results_root", default="./results")
    parser.add_argument("--filter", default="ettm1_calibrated")
    parser.add_argument("--baseline_experiment", default="ettm1_calibrated_pure_dlinear")
    parser.add_argument("--calibration_report_path", default="artifacts/core_results/ettm1_rule_calibration_report.json")
    parser.add_argument("--output_markdown", default="docs/calibrated_rule_prior_results.md")
    parser.add_argument("--output_csv", default="artifacts/core_results/ettm1_calibrated_rule_prior_summary.csv")
    args = parser.parse_args()
    summarize_calibrated_rule_results(**vars(args))


if __name__ == "__main__":
    main()
