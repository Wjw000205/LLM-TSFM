"""Diagnose whether event-mask timestamp locations are invariant to pred_len."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider


def diagnose_horizon_invariance(
    data: str,
    root_path: str,
    data_path: str,
    features: str,
    target: str,
    seq_len: int,
    label_len: int,
    pred_lens: list[int],
    llm_rule_path: str,
) -> dict[str, Any]:
    """Return per-horizon event-mask counts and cross-horizon consistency checks."""
    rows = [
        _collect_horizon_stats(
            data=data,
            root_path=root_path,
            data_path=data_path,
            features=features,
            target=target,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=int(pred_len),
            llm_rule_path=llm_rule_path,
        )
        for pred_len in pred_lens
    ]
    errors, warnings, differences = _check_horizon_invariance(rows)
    return {
        "data": data,
        "root_path": root_path,
        "data_path": data_path,
        "features": features,
        "target": target,
        "seq_len": int(seq_len),
        "label_len": int(label_len),
        "pred_lens": [int(value) for value in pred_lens],
        "llm_rule_path": llm_rule_path,
        "per_horizon": rows,
        "timestamp_differences": differences,
        "errors": errors,
        "warnings": warnings,
    }


def _collect_horizon_stats(
    data: str,
    root_path: str,
    data_path: str,
    features: str,
    target: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    llm_rule_path: str,
) -> dict[str, Any]:
    args = SimpleNamespace(
        root_path=root_path,
        data_path=data_path,
        data=data,
        features=features,
        target=target,
        seq_len=int(seq_len),
        label_len=int(label_len),
        pred_len=int(pred_len),
        use_zscore=1,
        timeenc=0,
        freq="t" if data.startswith("ETTm") else "h",
        use_llm_features=0,
        use_standard_time_features=0,
        use_llm_rule_features=0,
        use_oracle_features=0,
        llm_rule_path=llm_rule_path,
        batch_size=256,
        num_workers=0,
    )
    data_provider(args, "train")
    dataset, _ = data_provider(args, "test")

    unique_event_timestamps: set[str] = set()
    unique_prediction_timestamps: set[str] = set()
    repeated_event_points = 0
    pred_timestamp_start = None
    pred_timestamp_end = None

    for index in range(len(dataset)):
        info = dataset.get_prediction_timestamps(index)
        pred_timestamps = info["pred_timestamps"]
        pred_event_mask = np.asarray(info["pred_event_masks"])[:, 0, :]

        if pred_timestamp_start is None:
            pred_timestamp_start = pred_timestamps[0]
        pred_timestamp_end = pred_timestamps[-1]

        repeated_event_points += int(pred_event_mask.sum())
        for timestamp in pred_timestamps:
            unique_prediction_timestamps.add(str(timestamp))

        hit_rows = np.flatnonzero(pred_event_mask.max(axis=1) > 0)
        for hit_idx in hit_rows:
            unique_event_timestamps.add(str(pred_timestamps[int(hit_idx)]))

    sorted_unique_events = sorted(unique_event_timestamps)
    total_prediction_elements = int(len(dataset) * int(pred_len) * int(dataset.target_dim))
    unique_prediction_count = len(unique_prediction_timestamps)
    return {
        "num_samples": int(len(dataset)),
        "pred_len": int(pred_len),
        "target_channels": int(dataset.target_dim),
        "total_prediction_elements": total_prediction_elements,
        "repeated_event_points": int(repeated_event_points),
        "unique_event_timestamp_count": int(len(sorted_unique_events)),
        "unique_event_timestamps": sorted_unique_events,
        "first_20_unique_event_timestamps": sorted_unique_events[:20],
        "last_20_unique_event_timestamps": sorted_unique_events[-20:],
        "event_ratio_repeated": _safe_ratio(repeated_event_points, total_prediction_elements),
        "event_ratio_unique": _safe_ratio(len(sorted_unique_events), unique_prediction_count),
        "unique_prediction_timestamps": int(unique_prediction_count),
        "pred_timestamp_start": None if pred_timestamp_start is None else str(pred_timestamp_start),
        "pred_timestamp_end": None if pred_timestamp_end is None else str(pred_timestamp_end),
    }


def _check_horizon_invariance(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    warnings: list[str] = []
    differences: list[dict[str, Any]] = []
    if not rows:
        return errors, warnings, differences

    reference = set(rows[0]["unique_event_timestamps"])
    non_empty = [row for row in rows if row["unique_event_timestamp_count"] > 0]
    empty = [row for row in rows if row["unique_event_timestamp_count"] == 0]
    if non_empty and empty:
        empty_pred_lens = [row["pred_len"] for row in empty]
        non_empty_pred_lens = [row["pred_len"] for row in non_empty]
        errors.append(
            "empty_horizon_with_nonempty_reference: "
            f"empty_pred_lens={empty_pred_lens}, non_empty_pred_lens={non_empty_pred_lens}"
        )

    for row in rows[1:]:
        current = set(row["unique_event_timestamps"])
        if current != reference:
            missing = sorted(reference - current)
            extra = sorted(current - reference)
            differences.append(
                {
                    "reference_pred_len": rows[0]["pred_len"],
                    "pred_len": row["pred_len"],
                    "missing_from_pred_len": missing,
                    "extra_in_pred_len": extra,
                }
            )
            errors.append(
                "unique_event_timestamps_mismatch: "
                f"reference_pred_len={rows[0]['pred_len']}, pred_len={row['pred_len']}, "
                f"missing={len(missing)}, extra={len(extra)}"
            )

    sorted_rows = sorted(rows, key=lambda item: item["pred_len"])
    for previous, current in zip(sorted_rows, sorted_rows[1:]):
        if current["repeated_event_points"] < previous["repeated_event_points"]:
            warnings.append(
                "repeated_event_points_decreased_with_longer_horizon: "
                f"p{previous['pred_len']}={previous['repeated_event_points']}, "
                f"p{current['pred_len']}={current['repeated_event_points']}"
            )
    return errors, warnings, differences


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator == 0.0:
        return float("nan")
    return float(numerator) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--features", default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_lens", nargs="+", type=int, required=True)
    parser.add_argument("--llm_rule_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    report = diagnose_horizon_invariance(
        data=args.data,
        root_path=args.root_path,
        data_path=args.data_path,
        features=args.features,
        target=args.target,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_lens=args.pred_lens,
        llm_rule_path=args.llm_rule_path,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
