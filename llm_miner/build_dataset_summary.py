"""Build train-split-only dataset summaries for offline LLM prompting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_miner.common import (
    default_output_dir,
    infer_frequency,
    load_train_frame,
    select_value_columns,
    train_borders,
    write_json,
)


def build_summary(
    root_path: str,
    data_path: str,
    data: str,
    target: str = "OT",
    seq_len: int = 96,
    pred_len: int = 96,
    features: str = "M",
    split: str = "train",
    output_dir: str | None = None,
    output_path: str | None = None,
    near_zero_eps: float = 1e-5,
):
    """Summarize only the train split; val/test rows are never inspected."""
    if split != "train":
        raise ValueError("LLM miner only supports split='train' to avoid leakage.")
    train, date_col, csv_path = load_train_frame(root_path, data_path, data, seq_len)
    variable_names, target_columns = select_value_columns(train, date_col, features, target)
    numeric = train[variable_names].astype(np.float32)
    dates = train[date_col]

    mean = {col: float(numeric[col].mean()) for col in variable_names}
    std = {col: float(numeric[col].std(ddof=0)) for col in variable_names}
    min_values = {col: float(numeric[col].min()) for col in variable_names}
    max_values = {col: float(numeric[col].max()) for col in variable_names}
    zero_ratio = {col: float((numeric[col] == 0).mean()) for col in variable_names}
    near_zero_ratio = {col: float((numeric[col].abs() <= near_zero_eps).mean()) for col in variable_names}

    summary = {
        "dataset_name": data,
        "dataset": data,
        "analysis_scope": "train_only",
        "split": "train",
        "data_path": data_path,
        "source_csv": str(csv_path),
        "train_start_time": str(dates.iloc[0]),
        "train_end_time": str(dates.iloc[-1]),
        "time_range": {"start": str(dates.iloc[0]), "end": str(dates.iloc[-1])},
        "freq": infer_frequency(dates),
        "num_timesteps": int(len(train)),
        "row_count": int(len(train)),
        "num_variables": int(len(variable_names)),
        "variable_names": variable_names,
        "columns": variable_names,
        "target": target,
        "target_columns": target_columns,
        "mean": mean,
        "std": std,
        "min": min_values,
        "max": max_values,
        "zero_ratio_per_variable": zero_ratio,
        "near_zero_ratio_per_variable": near_zero_ratio,
        "statistics": {
            col: {
                "mean": mean[col],
                "std": std[col],
                "min": min_values[col],
                "max": max_values[col],
                "zero_count": int((numeric[col] == 0).sum()),
            }
            for col in variable_names
        },
        "hourly_mean_if_available": _grouped_mean(train, date_col, variable_names, "hour"),
        "daily_mean_if_available": _grouped_mean(train, date_col, variable_names, "weekday"),
        "monthly_mean_if_available": _grouped_mean(train, date_col, variable_names, "month"),
        "top_zero_windows": _top_zero_windows(train, date_col, variable_names, near_zero_eps),
        "top_peak_windows": _top_peak_windows(train, date_col, target_columns),
        "periodicity_candidates": _periodicity_candidates(train, date_col, target),
        "anomaly_candidates": _anomaly_candidates(train, date_col, variable_names),
        "candidate_periodic_events": _candidate_periodic_events(train, date_col, target),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "near_zero_eps": float(near_zero_eps),
    }

    output = Path(output_path) if output_path else default_output_dir(data, output_dir) / "dataset_summary.json"
    write_json(output, summary)
    return summary


def _grouped_mean(frame: pd.DataFrame, date_col: str, columns: list[str], group: str) -> dict:
    if group == "hour":
        key = frame[date_col].dt.hour
    elif group == "weekday":
        key = frame[date_col].dt.weekday
    elif group == "month":
        key = frame[date_col].dt.month
    else:
        return {}
    grouped = frame.groupby(key)[columns].mean(numeric_only=True)
    return {
        str(col): {str(idx): float(value) for idx, value in grouped[col].items()}
        for col in columns
        if col in grouped
    }


def _top_zero_windows(frame: pd.DataFrame, date_col: str, columns: list[str], eps: float, limit: int = 20):
    windows = []
    for col in columns:
        mask = frame[col].abs().to_numpy() <= eps
        for start, end in _runs(mask):
            values = frame[col].iloc[start : end + 1].abs()
            windows.append(
                {
                    "variable": col,
                    "start_time": str(frame[date_col].iloc[start]),
                    "end_time": str(frame[date_col].iloc[end]),
                    "duration": int(end - start + 1),
                    "zero_ratio": float(mask[start : end + 1].mean()),
                    "mean_abs_value": float(values.mean()),
                }
            )
    return sorted(windows, key=lambda item: (-item["duration"], item["mean_abs_value"]))[:limit]


def _top_peak_windows(frame: pd.DataFrame, date_col: str, columns: list[str], limit: int = 20):
    windows = []
    for col in columns:
        values = frame[col].astype(float)
        if values.empty:
            continue
        threshold = float(values.quantile(0.99))
        hits = np.flatnonzero(values.to_numpy() >= threshold)
        for idx in hits[:limit]:
            windows.append(
                {
                    "variable": col,
                    "start_time": str(frame[date_col].iloc[idx]),
                    "end_time": str(frame[date_col].iloc[idx]),
                    "value": float(values.iloc[idx]),
                    "threshold": threshold,
                }
            )
    return windows[:limit]


def _periodicity_candidates(frame: pd.DataFrame, date_col: str, target: str) -> dict:
    return {
        "first_day_count": int((frame[date_col].dt.day == 1).sum()),
        "hourly_target_mean": {
            str(hour): float(value) for hour, value in frame.groupby(frame[date_col].dt.hour)[target].mean().items()
        },
        "weekday_target_mean": {
            str(day): float(value) for day, value in frame.groupby(frame[date_col].dt.weekday)[target].mean().items()
        },
    }


def _candidate_periodic_events(frame: pd.DataFrame, date_col: str, target: str) -> dict:
    return _periodicity_candidates(frame, date_col, target)


def _anomaly_candidates(frame: pd.DataFrame, date_col: str, columns: list[str], limit: int = 20):
    candidates = []
    for col in columns:
        values = frame[col].astype(float)
        std = float(values.std(ddof=0))
        if std <= 1e-8:
            continue
        z = (values - float(values.mean())).abs() / std
        for idx in np.argsort(z.to_numpy())[-limit:][::-1]:
            if z.iloc[idx] < 3.0:
                continue
            candidates.append(
                {
                    "variable": col,
                    "timestamp": str(frame[date_col].iloc[idx]),
                    "value": float(values.iloc[idx]),
                    "z_score": float(z.iloc[idx]),
                }
            )
    return candidates[:limit]


def _runs(mask: np.ndarray):
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            yield start, idx - 1
            start = None
    if start is not None:
        yield start, len(mask) - 1


def _train_borders(data: str, total_len: int, seq_len: int):
    """Backward-compatible helper used by older tests/modules."""
    return train_borders(data, total_len, seq_len)


def main():
    parser = argparse.ArgumentParser(description="Build train-only dataset summary for offline LLM mining.")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--near_zero_eps", type=float, default=1e-5)
    args = parser.parse_args()
    build_summary(**vars(args))


if __name__ == "__main__":
    main()

