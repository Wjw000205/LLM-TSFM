"""Select long-tail candidates under an overall-MSE guardrail."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def select_pareto(
    baseline_metrics: str,
    sweep_csv: str,
    output_csv: str = "artifacts/core_results/ettm1_pareto_longtail.csv",
    output_markdown: str = "docs/longtail_guardrail_results.md",
    overall_tolerance: float = 0.05,
):
    """Read sweep results, mark guardrail acceptance, and emit a Pareto table."""
    baseline = _read_json(baseline_metrics)
    baseline_mse = _number_from_mapping(baseline, "mse", "overall_mse", "Overall MSE")
    baseline_event_mse = _number_from_mapping(baseline, "event_window_mse", "event_mse", "Event MSE")
    rows = [_normalize_row(row, baseline_mse, baseline_event_mse, overall_tolerance) for row in _read_csv(sweep_csv)]
    pareto_rows = _pareto_frontier(rows)
    accepted = [row for row in pareto_rows if row["accepted_by_guardrail"]]
    if accepted:
        best = min(accepted, key=lambda row: (row["event_mse"], row["overall_mse"]))
    elif pareto_rows:
        best = min(pareto_rows, key=lambda row: (row["overall_mse"], row["event_mse"]))
    else:
        best = {}

    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv_path, pareto_rows)

    output_md_path = Path(output_markdown)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(
        _markdown(
            pareto_rows=pareto_rows,
            all_rows=rows,
            best=best,
            baseline_mse=baseline_mse,
            baseline_event_mse=baseline_event_mse,
            overall_tolerance=overall_tolerance,
        ),
        encoding="utf-8",
    )
    return {"best": best, "rows": pareto_rows, "output_csv": str(output_csv_path), "output_markdown": str(output_md_path)}


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: str) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_row(row: dict, baseline_mse: float, baseline_event_mse: float, tolerance: float) -> dict:
    overall_mse = _number_from_mapping(row, "overall_mse", "Overall MSE", "mse")
    overall_mae = _number_from_mapping(row, "overall_mae", "Overall MAE", "mae", default=0.0)
    event_mse = _number_from_mapping(row, "event_mse", "Event MSE", "event_window_mse")
    zero_mse = _number_from_mapping(row, "zero_mse", "Zero MSE", "zero_event_mse", default=0.0)
    rule_score = _number_from_mapping(row, "rule_score", "Rule Score", "rule_consistency_score", default=0.0)
    accepted = overall_mse <= baseline_mse * (1.0 + tolerance) and event_mse < baseline_event_mse
    return {
        "experiment": _text_from_mapping(row, "experiment", "Experiment", "des"),
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "event_mse": event_mse,
        "zero_mse": zero_mse,
        "rule_score": rule_score,
        "overall_mse_delta": (overall_mse - baseline_mse) / baseline_mse,
        "event_mse_reduction": (baseline_event_mse - event_mse) / baseline_event_mse,
        "accepted_by_guardrail": accepted,
        "path": _text_from_mapping(row, "path", "Path", default=""),
    }


def _pareto_frontier(rows: list[dict]) -> list[dict]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = other["overall_mse"] <= row["overall_mse"] and other["event_mse"] <= row["event_mse"]
            strictly_better = other["overall_mse"] < row["overall_mse"] or other["event_mse"] < row["event_mse"]
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: (not item["accepted_by_guardrail"], item["overall_mse"], item["event_mse"]))


def _number_from_mapping(mapping: dict, *keys: str, default: float | None = None) -> float:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return float(value)
    if default is not None:
        return default
    raise KeyError(f"Missing required numeric field, expected one of: {keys}")


def _text_from_mapping(mapping: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _write_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "experiment",
        "overall_mse",
        "overall_mae",
        "event_mse",
        "zero_mse",
        "rule_score",
        "overall_mse_delta",
        "event_mse_reduction",
        "accepted_by_guardrail",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(
    pareto_rows: list[dict],
    all_rows: list[dict],
    best: dict,
    baseline_mse: float,
    baseline_event_mse: float,
    overall_tolerance: float,
) -> str:
    lines = [
        "# Long-Tail Guardrail Results",
        "",
        f"Baseline overall MSE: `{baseline_mse:.6f}`",
        f"Baseline event MSE: `{baseline_event_mse:.6f}`",
        f"Overall-MSE tolerance: `{overall_tolerance:.2%}`",
        "",
        "## Selected Candidate",
        "",
    ]
    if best:
        lines.append(
            f"- `{best['experiment']}`: overall_mse={best['overall_mse']:.6f}, "
            f"event_mse={best['event_mse']:.6f}, accepted={best['accepted_by_guardrail']}"
        )
    else:
        lines.append("- No completed sweep rows were available.")
    lines.extend(
        [
            "",
            "## All Sweep Runs",
            "",
            "| Experiment | Overall MSE | Event MSE | Overall Delta | Event Reduction | Accepted |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(all_rows, key=lambda item: (not item["accepted_by_guardrail"], item["overall_mse"])):
        lines.append(
            "| {experiment} | {overall_mse:.6f} | {event_mse:.6f} | {overall_mse_delta:.2%} | "
            "{event_mse_reduction:.2%} | {accepted_by_guardrail} |".format(**row)
        )
    if not all_rows:
        lines.append("| No rows | 0 | 0 | 0% | 0% | False |")
    lines.extend(
        [
            "",
            "## Pareto Frontier",
            "",
            "| Experiment | Overall MSE | Event MSE | Overall Delta | Event Reduction | Accepted |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pareto_rows:
        lines.append(
            "| {experiment} | {overall_mse:.6f} | {event_mse:.6f} | {overall_mse_delta:.2%} | "
            "{event_mse_reduction:.2%} | {accepted_by_guardrail} |".format(**row)
        )
    if not pareto_rows:
        lines.append("| No rows | 0 | 0 | 0% | 0% | False |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A candidate is accepted only if it stays within the overall-MSE guardrail and improves event-window MSE.",
            "- For checkpoint selection, prefer a validation-split baseline metrics file; using a test-split baseline can trigger fallback because validation MSE is not directly comparable.",
            "- Hard intervention remains an oracle upper bound, not a deployable long-tail result.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Select Pareto long-tail candidates under an MSE guardrail.")
    parser.add_argument("--baseline_metrics", required=True)
    parser.add_argument("--sweep_csv", required=True)
    parser.add_argument("--output_csv", default="artifacts/core_results/ettm1_pareto_longtail.csv")
    parser.add_argument("--output_markdown", default="docs/longtail_guardrail_results.md")
    parser.add_argument("--overall_tolerance", type=float, default=0.05)
    args = parser.parse_args()
    select_pareto(**vars(args))


if __name__ == "__main__":
    main()
