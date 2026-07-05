"""Shared train-only utilities for offline LLM rule mining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ETT_HOURLY = {"ETTh1", "ETTh2"}
ETT_MINUTE = {"ETTm1", "ETTm2"}


def train_borders(data: str, total_len: int, seq_len: int = 96) -> tuple[int, int]:
    """Return train split borders using the same fixed ETT policy as the loader."""
    if data in ETT_HOURLY:
        train_end = 12 * 30 * 24
        required = train_end + 4 * 30 * 24 + 4 * 30 * 24
        if total_len < required:
            raise ValueError(f"{data} requires fixed 12/4/4 split, but total_len={total_len} < required={required}.")
        return 0, train_end
    if data in ETT_MINUTE:
        train_end = 12 * 30 * 24 * 4
        required = train_end + 4 * 30 * 24 * 4 + 4 * 30 * 24 * 4
        if total_len < required:
            raise ValueError(f"{data} requires fixed 12/4/4 split, but total_len={total_len} < required={required}.")
        return 0, train_end
    return 0, int(total_len * 0.7)


def load_train_frame(root_path: str, data_path: str, data: str, seq_len: int = 96):
    """Load only the train split from a CSV file."""
    csv_path = Path(root_path) / data_path
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")
    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    start, end = train_borders(data, len(frame), seq_len)
    return frame.iloc[start:end].copy(), date_col, csv_path


def select_value_columns(frame: pd.DataFrame, date_col: str, features: str, target: str) -> tuple[list[str], list[str]]:
    """Return input variable names and forecast target columns for a feature mode."""
    value_cols = [col for col in frame.columns if col != date_col]
    if target not in value_cols:
        raise ValueError(f"target '{target}' not found in columns: {value_cols}")
    if features == "S":
        return [target], [target]
    if features == "M":
        return value_cols, value_cols
    if features == "MS":
        return value_cols, [target]
    raise ValueError("features must be one of 'M', 'S', or 'MS'.")


def infer_frequency(dates: pd.Series) -> str:
    """Infer a readable timestamp frequency."""
    inferred = pd.infer_freq(pd.DatetimeIndex(dates))
    if inferred:
        return str(inferred)
    if len(dates) < 2:
        return "unknown"
    delta = dates.iloc[1] - dates.iloc[0]
    return str(delta)


def default_output_dir(data: str, output_dir: str | None) -> Path:
    return Path(output_dir) if output_dir else Path("artifacts") / "llm_miner" / data


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return output


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value

