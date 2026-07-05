"""Validate mined rule JSON against train-only data and supported templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from llm_miner.common import default_output_dir, load_train_frame, select_value_columns, write_json


SUPPORTED_LOSSES = {"event_weighted_mse", "zero_consistency", "peak_shape", "diff", "frequency"}
SUPPORTED_FEATURES = {
    "event_mask",
    "zero_event_mask",
    "peak_mask",
    "days_to_event",
    "hour_distance_to_peak",
    "rule_confidence",
    "support_count",
}
SUPPORTED_CONDITIONS = {"calendar_periodic", "hourly", "weekday"}


def validate_rules(
    rule_path: str,
    data: str,
    root_path: str,
    data_path: str,
    features: str = "M",
    target: str = "OT",
    seq_len: int = 96,
    output_dir: str | None = None,
):
    """Validate a rule JSON and write a validation report."""
    payload = json.loads(Path(rule_path).read_text(encoding="utf-8"))
    train, date_col, _ = load_train_frame(root_path, data_path, data, seq_len)
    _, target_columns = select_value_columns(train, date_col, features, target)
    train_start = pd.Timestamp(train[date_col].iloc[0])
    train_end = pd.Timestamp(train[date_col].iloc[-1])
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("dataset_name") != data:
        errors.append(f"dataset_name mismatch: rule={payload.get('dataset_name')} data={data}")
    if payload.get("analysis_scope") not in {None, "train_only"}:
        errors.append(f"analysis_scope must be train_only, got {payload.get('analysis_scope')}")

    for pattern in payload.get("patterns", []):
        name = pattern.get("name", "<unnamed>")
        condition = pattern.get("condition", {})
        kind = condition.get("kind")
        if kind not in SUPPORTED_CONDITIONS:
            errors.append(f"Rule '{name}' has unsupported condition kind '{kind}'.")
        if kind == "calendar_periodic" and "anchor" not in condition:
            errors.append(f"Rule '{name}' calendar_periodic condition is missing anchor.")

        affected = pattern.get("affected_variables", "all")
        if affected != "all":
            for variable in affected:
                if variable not in target_columns:
                    warnings.append(f"Rule '{name}' references affected variable '{variable}' not present in target columns.")

        for loss_name in pattern.get("losses", {}):
            if loss_name not in SUPPORTED_LOSSES:
                warnings.append(f"Unsupported loss '{loss_name}' in rule '{name}'.")
        for feature_name in pattern.get("features", {}):
            if feature_name not in SUPPORTED_FEATURES:
                warnings.append(f"Unsupported feature '{feature_name}' in rule '{name}'.")

        support_count = pattern.get("support_count")
        if support_count is not None and int(support_count) <= 0:
            warnings.append(f"Rule '{name}' support_count must be > 0.")

        confidence = pattern.get("confidence")
        if confidence is not None and not (0.0 <= float(confidence) <= 1.0):
            errors.append(f"Rule '{name}' confidence must be in [0,1], got {confidence}.")

        for window in pattern.get("evidence_windows", []):
            if len(window) != 2:
                warnings.append(f"Rule '{name}' has invalid evidence window {window}.")
                continue
            start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
            if start < train_start or end > train_end:
                warnings.append(
                    f"Rule '{name}' evidence window {window} is outside train split "
                    f"[{train_start}, {train_end}]."
                )

    report = {
        "ok": not errors,
        "rule_path": str(rule_path),
        "dataset_name": payload.get("dataset_name"),
        "expected_dataset": data,
        "analysis_scope": payload.get("analysis_scope"),
        "train_start_time": str(train_start),
        "train_end_time": str(train_end),
        "errors": errors,
        "warnings": warnings,
    }
    output = default_output_dir(data, output_dir) / "rule_validation_report.json"
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate offline LLM rule JSON.")
    parser.add_argument("--rule_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    report = validate_rules(**vars(args))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

