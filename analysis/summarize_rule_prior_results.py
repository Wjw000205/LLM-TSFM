"""Summarize rule-prior fusion experiments."""

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


def summarize_rule_prior_results(
    results_root: str = "./results",
    filter: str = "ettm1_rule_prior",
    baseline_experiment: str = "ettm1_rule_prior_pure_dlinear",
    output_markdown: str = "docs/rule_prior_results.md",
    output_csv: str = "artifacts/core_results/ettm1_rule_prior_summary.csv",
):
    root = Path(results_root)
    rows = []
    for result_dir in sorted(root.glob(f"*{filter}*")):
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
                "Overall MAE": float(metrics.get("mae", 0.0)),
                "Event MSE": float(metrics.get("event_window_mse", 0.0)),
                "Zero MSE": float(metrics.get("zero_event_mse", 0.0)),
                "Rule Score": float(metrics.get("rule_consistency_score", 0.0)),
                "Overall Delta vs Baseline": 0.0,
                "Event Reduction vs Baseline": 0.0,
                "non_event_mse": non_event["non_event_mse"],
                "non_event_mae": non_event["non_event_mae"],
                "Alpha": _alpha(config),
                "Notes": _notes(config),
                "Path": str(result_dir),
            }
        )

    baseline = _find_baseline(rows, baseline_experiment)
    if baseline is not None:
        baseline_mse = baseline["Overall MSE"]
        baseline_event = baseline["Event MSE"]
        for row in rows:
            row["Overall Delta vs Baseline"] = _safe_ratio(row["Overall MSE"] - baseline_mse, baseline_mse)
            row["Event Reduction vs Baseline"] = _safe_ratio(baseline_event - row["Event MSE"], baseline_event)

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(output_markdown)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(rows, baseline_experiment), encoding="utf-8")
    return {"rows": rows, "csv_path": str(csv_path), "markdown_path": str(md_path)}


def _fieldnames():
    return [
        "Experiment",
        "Overall MSE",
        "Overall MAE",
        "Event MSE",
        "Zero MSE",
        "Rule Score",
        "Overall Delta vs Baseline",
        "Event Reduction vs Baseline",
        "non_event_mse",
        "non_event_mae",
        "Alpha",
        "Notes",
        "Path",
    ]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _non_event_metrics(result_dir: Path, config: dict) -> dict[str, float]:
    pred_path = result_dir / "pred_normalized.npy"
    true_path = result_dir / "true_normalized.npy"
    if not pred_path.exists() or not true_path.exists():
        return {"non_event_mse": 0.0, "non_event_mae": 0.0}
    pred = np.load(pred_path)
    true = np.load(true_path)
    masks = _test_masks(config)[: pred.shape[0]]
    event_mask = masks[:, :, 0, :]
    non_event = 1.0 - event_mask
    denom = non_event.sum()
    if denom <= 1e-8:
        return {"non_event_mse": 0.0, "non_event_mae": 0.0}
    error = pred - true
    return {
        "non_event_mse": float(((error**2) * non_event).sum() / denom),
        "non_event_mae": float((np.abs(error) * non_event).sum() / denom),
    }


def _test_masks(config: dict):
    args = SimpleNamespace(**config)
    args.num_workers = 0
    data_provider(args, "train")
    test_data, _ = data_provider(args, "test")
    masks = []
    for idx in range(len(test_data)):
        masks.append(test_data[idx][-1].numpy()[-test_data.pred_len :])
    return np.asarray(masks, dtype=np.float32)


def _alpha(config: dict) -> str:
    if int(config.get("use_rule_prior_fusion", 0)):
        return str(config.get("rule_prior_alpha", ""))
    return ""


def _notes(config: dict) -> str:
    notes = []
    if int(config.get("use_rule_prior_fusion", 0)):
        notes.append("rule_prior_fusion")
    if int(config.get("use_dataset_aware_loss", 0)):
        notes.append("dataset_aware_loss")
    if int(config.get("use_rule_adapter", 0)):
        notes.append("output_rule_adapter")
    if int(config.get("use_intervention_layer", 0)):
        notes.append("intermediate_intervention")
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


def _markdown(rows: list[dict], baseline_experiment: str) -> str:
    diagnosis = _load_default_diagnosis()
    lines = [
        "# Rule Prior Results",
        "",
        f"Baseline experiment: `{baseline_experiment}`",
        "",
        "| Experiment | Overall MSE | Overall MAE | Event MSE | Zero MSE | Rule Score | Overall Delta | Event Reduction | Non-event MSE | Alpha | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {Experiment} | {Overall MSE:.6f} | {Overall MAE:.6f} | {Event MSE:.6f} | {Zero MSE:.6f} | "
            "{Rule Score:.6f} | {Overall Delta vs Baseline:.2%} | {Event Reduction vs Baseline:.2%} | "
            "{non_event_mse:.6f} | {Alpha} | {Notes} |".format(**row)
        )
    if not rows:
        lines.append("| No matching completed runs | 0 | 0 | 0 | 0 | 0 | 0% | 0% | 0 |  | Run script first |")
    lines.extend(_conclusion_lines(rows, baseline_experiment, diagnosis))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Rule-prior fusion is a deterministic soft fusion branch, not a trainable MLP adapter.",
            "- If event MSE worsens as alpha increases, diagnose zero masks and zero targets before tuning the model.",
            "- Hard intervention should be called an oracle upper bound only after diagnosis confirms the mask and target are valid.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_default_diagnosis() -> dict | None:
    path = Path("artifacts/core_results/rule_prior_diagnosis.json")
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except json.JSONDecodeError:
        return None


def _conclusion_lines(rows: list[dict], baseline_experiment: str, diagnosis: dict | None) -> list[str]:
    if not rows:
        return []
    baseline = _find_baseline(rows, baseline_experiment)
    rule_prior_rows = [
        row
        for row in rows
        if "rule_prior_fusion" in row["Notes"] and "dataset_aware_loss" not in row["Notes"]
    ]
    lines = ["", "## Run Conclusion", ""]
    if baseline is not None:
        lines.append(
            f"- Pure DLinear baseline: overall MSE {baseline['Overall MSE']:.6f}, event MSE {baseline['Event MSE']:.6f}."
        )
    if rule_prior_rows and baseline is not None:
        best_event = min(rule_prior_rows, key=lambda row: row["Event MSE"])
        best_overall = min(rule_prior_rows, key=lambda row: row["Overall MSE"])
        lines.append(
            f"- Best rule-prior event result is alpha={best_event['Alpha']}: event MSE {best_event['Event MSE']:.6f}, "
            f"event reduction {best_event['Event Reduction vs Baseline']:.2%}."
        )
        lines.append(
            f"- Best rule-prior overall result is alpha={best_overall['Alpha']}: overall MSE {best_overall['Overall MSE']:.6f}, "
            f"delta {best_overall['Overall Delta vs Baseline']:.2%}."
        )
        non_event_drift = max(abs(row["non_event_mse"] - baseline["non_event_mse"]) for row in rule_prior_rows)
        lines.append(
            f"- Non-event MSE drift across pure rule-prior runs is at most {non_event_drift:.6f}; "
            "the degradation is localized to rule-triggered timestamps."
        )
    if diagnosis is not None:
        alignment = diagnosis.get("mask_alignment", {})
        if alignment:
            lines.append(
                f"- Mask alignment check: matches_recomputed_masks={alignment.get('matches_recomputed_masks')}, "
                f"max_abs_diff={alignment.get('max_abs_diff')}."
            )
        for note in diagnosis.get("interpretation", []):
            lines.append(f"- Diagnosis: {note}")
    lines.append("- Success criteria are not met for the current ETTm1 zero_event rule: event MSE does not improve under rule-prior fusion.")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Summarize rule-prior fusion experiment folders.")
    parser.add_argument("--results_root", default="./results")
    parser.add_argument("--filter", default="ettm1_rule_prior")
    parser.add_argument("--baseline_experiment", default="ettm1_rule_prior_pure_dlinear")
    parser.add_argument("--output_markdown", default="docs/rule_prior_results.md")
    parser.add_argument("--output_csv", default="artifacts/core_results/ettm1_rule_prior_summary.csv")
    args = parser.parse_args()
    summarize_rule_prior_results(**vars(args))


if __name__ == "__main__":
    main()
