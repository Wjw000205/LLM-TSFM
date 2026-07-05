"""Summarize core innovation experiment result folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize_core_results(
    results_root: str = "./results",
    filter: str = "ettm1_core",
    output_markdown: str = "docs/core_innovation_results.md",
    output_csv: str = "artifacts/core_results/ettm1_core_summary.csv",
):
    """Collect metrics from result folders and write CSV plus Markdown tables."""
    root = Path(results_root)
    rows = []
    for result_dir in sorted(root.glob(f"*{filter}*")):
        metrics_path = result_dir / "metrics_normalized.json"
        original_path = result_dir / "metrics_original_scale.json"
        config_path = result_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        original = json.loads(original_path.read_text(encoding="utf-8")) if original_path.exists() else {}
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "Experiment": config.get("des", result_dir.name),
                "Overall MSE": metrics.get("mse", 0.0),
                "Overall MAE": metrics.get("mae", 0.0),
                "Original MSE": original.get("mse", 0.0),
                "Original MAE": original.get("mae", 0.0),
                "Event MSE": metrics.get("event_window_mse", 0.0),
                "Zero MSE": metrics.get("zero_event_mse", 0.0),
                "Rule Score": metrics.get("rule_consistency_score", 0.0),
                "Event Points": metrics.get("num_event_points", 0),
                "Zero Points": metrics.get("num_zero_event_points", 0),
                "Notes": _notes(config),
                "Path": str(result_dir),
            }
        )

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    md_path = Path(output_markdown)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_markdown(rows, filter), encoding="utf-8")
    return {"rows": rows, "csv_path": csv_path, "markdown_path": md_path}


def _fieldnames():
    return [
        "Experiment",
        "Overall MSE",
        "Overall MAE",
        "Original MSE",
        "Original MAE",
        "Event MSE",
        "Zero MSE",
        "Rule Score",
        "Event Points",
        "Zero Points",
        "Notes",
        "Path",
    ]


def _notes(config: dict) -> str:
    notes = []
    if int(config.get("use_standard_time_features", 0)):
        notes.append("standard_time")
    if int(config.get("use_llm_rule_features", 0)):
        notes.append("llm_rule_features")
    if int(config.get("use_dataset_aware_loss", 0)):
        notes.append("dataset_aware_loss")
    if int(config.get("use_hard_intervention", 0)):
        notes.append("oracle_like_hard_intervention")
    return ", ".join(notes) if notes else "pure_dlinear"


def _markdown(rows: list[dict], filter_value: str) -> str:
    lines = [
        "# Core Innovation Results",
        "",
        f"Filter: `{filter_value}`",
        "",
        "| Experiment | Overall MSE | Overall MAE | Event MSE | Zero MSE | Rule Score | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {Experiment} | {Overall MSE:.6f} | {Overall MAE:.6f} | {Event MSE:.6f} | "
            "{Zero MSE:.6f} | {Rule Score:.6f} | {Notes} |".format(**row)
        )
    if not rows:
        lines.append("| No matching completed runs | 0 | 0 | 0 | 0 | 0 | Run scripts first |")
    lines.extend(
        [
            "",
            "## Auto-Diagnosis Draft",
            "",
            "- Naive dataset-aware loss can reduce event-window MSE while materially increasing overall MSE; do not treat that as a deployable win.",
            "- Use guarded selection, non-event preservation, and weight sweeps when optimizing long-tail regions.",
            "- Treat `hard_intervention_oracle` as an oracle-like upper bound, not a deployable method result.",
            "- If event point counts are zero, regenerate or validate rule masks before interpreting event metrics.",
            "- A deployable long-tail result should stay inside the overall-MSE guardrail while improving event-window metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize core innovation result folders.")
    parser.add_argument("--results_root", default="./results")
    parser.add_argument("--filter", default="ettm1_core")
    parser.add_argument("--output_markdown", default="docs/core_innovation_results.md")
    parser.add_argument("--output_csv", default="artifacts/core_results/ettm1_core_summary.csv")
    args = parser.parse_args()
    summarize_core_results(**vars(args))


if __name__ == "__main__":
    main()
