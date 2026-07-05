"""Build train-split-only dataset summaries for LLM prompting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ETT_HOURLY = {"ETTh1", "ETTh2"}
ETT_MINUTE = {"ETTm1", "ETTm2"}


def build_summary(root_path: str, data_path: str, data: str, target: str, seq_len: int, output_path: str):
    """Summarize only the train split; val/test rows are never inspected."""
    csv_path = Path(root_path) / data_path
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")
    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    train_start, train_end = _train_borders(data, len(frame), seq_len)
    train = frame.iloc[train_start:train_end].copy()
    value_cols = [col for col in train.columns if col != date_col]
    if target not in value_cols:
        raise ValueError(f"target '{target}' not found in {value_cols}")

    numeric = train[value_cols].astype(np.float32)
    summary = {
        "dataset": data,
        "split": "train",
        "source_csv": str(csv_path),
        "row_count": int(len(train)),
        "time_range": {
            "start": str(train[date_col].iloc[0]),
            "end": str(train[date_col].iloc[-1]),
        },
        "columns": value_cols,
        "target": target,
        "statistics": {
            col: {
                "mean": float(numeric[col].mean()),
                "std": float(numeric[col].std(ddof=0)),
                "min": float(numeric[col].min()),
                "max": float(numeric[col].max()),
                "zero_count": int((numeric[col] == 0).sum()),
            }
            for col in value_cols
        },
        "candidate_periodic_events": _candidate_periodic_events(train, date_col, target),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _candidate_periodic_events(frame: pd.DataFrame, date_col: str, target: str) -> dict:
    first_day = frame[date_col].dt.day == 1
    by_hour = frame.groupby(frame[date_col].dt.hour)[target].mean().to_dict()
    return {
        "first_day_count": int(first_day.sum()),
        "hourly_target_mean": {str(hour): float(value) for hour, value in by_hour.items()},
    }


def _train_borders(data: str, total_len: int, seq_len: int):
    if data in ETT_HOURLY:
        required = (12 + 4 + 4) * 30 * 24
        if total_len < required:
            raise ValueError(f"{data} requires fixed 12/4/4 split, but total_len={total_len} < required={required}.")
        return 0, 12 * 30 * 24
    if data in ETT_MINUTE:
        required = (12 + 4 + 4) * 30 * 24 * 4
        if total_len < required:
            raise ValueError(f"{data} requires fixed 12/4/4 split, but total_len={total_len} < required={required}.")
        return 0, 12 * 30 * 24 * 4
    return 0, int(total_len * 0.7)


def main():
    parser = argparse.ArgumentParser(description="Build train-only dataset summary for offline LLM mining.")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    build_summary(**vars(args))


if __name__ == "__main__":
    main()

