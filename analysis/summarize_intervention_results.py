"""Summarize timestamp-conditioned intervention experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize_intervention_results(
    results_root: str = "./results",
    filter: str = "ettm1_intervention",
    baseline_experiment: str = "ettm1_intervention_pure_dlinear",
    output_markdown: str = "docs/intervention_results.md",
    output_csv: str = "artifacts/core_results/ettm1_intervention_summary.csv",
):
    root = Path(results_root)
    rows = []
    for result_dir in sorted(root.glob(f"*{filter}*")):
        metrics_path = result_dir / "metrics_normalized.json"
        config_path = result_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        stats = _read_json(result_dir / "intervention_stats.json")
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
                "Mean Event Gate": float(stats.get("mean_event_gate", 0.0)),
                "Mean Non-event Gate": float(stats.get("mean_non_event_gate", 0.0)),
                "Notes": _notes(config),
                "Path": str(result_dir),
            }
        )

    baseline = _find_baseline(rows, baseline_experiment)
    if baseline is not None:
        baseline_mse = baseline["Overall MSE"]
        baseline_event_mse = baseline["Event MSE"]
        for row in rows:
            row["Overall Delta vs Baseline"] = _safe_ratio(row["Overall MSE"] - baseline_mse, baseline_mse)
            row["Event Reduction vs Baseline"] = _safe_ratio(baseline_event_mse - row["Event MSE"], baseline_event_mse)

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
        "Mean Event Gate",
        "Mean Non-event Gate",
        "Notes",
        "Path",
    ]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_baseline(rows: list[dict], baseline_experiment: str) -> dict | None:
    for row in rows:
        if row["Experiment"] == baseline_experiment:
            return row
    return rows[0] if rows else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def _notes(config: dict) -> str:
    notes = []
    if int(config.get("use_rule_adapter", 0)):
        notes.append("output_rule_adapter")
    if int(config.get("use_intervention_layer", 0)):
        notes.append("intermediate_intervention")
    if int(config.get("use_dataset_aware_loss", 0)):
        notes.append("dataset_aware_loss")
    if int(config.get("use_intervention_reg", 0)):
        notes.append("intervention_reg")
    if int(config.get("use_hard_intervention", 0)):
        notes.append("oracle_like_hard_intervention")
    return ", ".join(notes) if notes else "pure_dlinear"


def _markdown(rows: list[dict], baseline_experiment: str) -> str:
    lines = [
        "# Intervention Results",
        "",
        f"Baseline experiment: `{baseline_experiment}`",
        "",
        "| Experiment | Overall MSE | Overall MAE | Event MSE | Zero MSE | Rule Score | Overall Delta | Event Reduction | Mean Event Gate | Mean Non-event Gate | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {Experiment} | {Overall MSE:.6f} | {Overall MAE:.6f} | {Event MSE:.6f} | {Zero MSE:.6f} | "
            "{Rule Score:.6f} | {Overall Delta vs Baseline:.2%} | {Event Reduction vs Baseline:.2%} | "
            "{Mean Event Gate:.6f} | {Mean Non-event Gate:.6f} | {Notes} |".format(**row)
        )
    if not rows:
        lines.append("| No matching completed runs | 0 | 0 | 0 | 0 | 0 | 0% | 0% | 0 | 0 | Run script first |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Current ETTm1 results show the intermediate intervention preserves or improves overall MSE, but it does not yet improve event-window MSE against pure DLinear.",
            "- Intervention regularization suppresses non-event gate activity, which confirms the regularizer is active, but this run still needs better event supervision or rule quality to improve event MSE.",
            "- In this run, event-window improvements come from `dataset_aware_loss` and `output_rule_adapter`, but both materially hurt overall MSE.",
            "- `output_rule_adapter` is post-prediction residual correction and remains an ablation.",
            "- `intermediate_intervention` is the main timestamp-conditioned rule-gated method.",
            "- `hard_intervention_oracle` is an oracle-like ablation, not a deployable method; with the current ETTm1 zero-event mask it is not an empirical upper bound.",
            "- `dataset_aware_loss` is a diagnostic baseline because it can reduce event MSE while hurting overall MSE.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize timestamp-conditioned intervention experiments.")
    parser.add_argument("--results_root", default="./results")
    parser.add_argument("--filter", default="ettm1_intervention")
    parser.add_argument("--baseline_experiment", default="ettm1_intervention_pure_dlinear")
    parser.add_argument("--output_markdown", default="docs/intervention_results.md")
    parser.add_argument("--output_csv", default="artifacts/core_results/ettm1_intervention_summary.csv")
    args = parser.parse_args()
    summarize_intervention_results(**vars(args))


if __name__ == "__main__":
    main()
