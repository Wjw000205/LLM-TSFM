"""Create auxiliary features from offline LLM rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_rules.mask_generator import generate_event_mask
from llm_rules.rule_parser import parse_llm_rules


def generate_standard_time_features(timestamps):
    """Generate ordinary calendar features that are not counted as LLM rules."""
    dates = _to_datetime_index(timestamps)
    arrays = [
        _column(dates.hour / 23.0),
        _column(dates.weekday / 6.0),
        _column((dates.day - 1) / 30.0),
        _column((dates.month - 1) / 11.0),
        _column(dates.day == 1),
    ]
    names = [
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "month_of_year",
        "first_day_indicator",
    ]
    return np.concatenate(arrays, axis=1).astype(np.float32), names


def generate_llm_rule_features(timestamps, rules, target_columns: list[str] | None = None):
    """Generate features sourced from the offline LLM rule JSON only."""
    dates = _to_datetime_index(timestamps)
    parsed = parse_llm_rules(rules)
    if parsed is None:
        return np.zeros((len(dates), 0), dtype=np.float32), []

    mask_dict = generate_event_mask(dates, parsed, target_columns=target_columns)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    for pattern in parsed.patterns:
        mask = mask_dict[pattern.name].astype(np.float32)
        if pattern.features.get("event_mask"):
            _append_channel_features(arrays, names, mask, f"event_mask_{pattern.name}", target_columns)
        if pattern.features.get("peak_mask"):
            _append_channel_features(arrays, names, mask, f"peak_mask_{pattern.name}", target_columns)
        if pattern.features.get("zero_event_mask") or pattern.type == "zero_event":
            _append_channel_features(arrays, names, mask, f"zero_event_mask_{pattern.name}", target_columns)
        if pattern.features.get("days_to_event"):
            arrays.append(_days_to_next_event(dates, mask.max(axis=1)).reshape(-1, 1))
            names.append(f"days_to_event_{pattern.name}")
        if pattern.features.get("hour_distance_to_peak"):
            hour = int(pattern.condition.get("hour", 0))
            arrays.append(_hour_distance(dates.hour.to_numpy(), hour).reshape(-1, 1))
            names.append(f"hour_distance_to_peak_{pattern.name}")
        if "rule_confidence" in pattern.features:
            arrays.append(np.full((len(dates), 1), float(pattern.features["rule_confidence"]), dtype=np.float32))
            names.append(f"rule_confidence_{pattern.name}")
        if "support_count" in pattern.features:
            arrays.append(np.full((len(dates), 1), float(pattern.features["support_count"]), dtype=np.float32))
            names.append(f"support_count_feature_{pattern.name}")

    if not arrays:
        return np.zeros((len(dates), 0), dtype=np.float32), []
    return np.concatenate(arrays, axis=1).astype(np.float32), names


def generate_oracle_features(timestamps, rules, target_columns: list[str] | None = None):
    """Generate oracle/manual rule features for explicit ablation settings."""
    dates = _to_datetime_index(timestamps)
    parsed = parse_llm_rules(rules)
    if parsed is None:
        return np.zeros((len(dates), 0), dtype=np.float32), []
    masks = generate_event_mask(dates, parsed, target_columns=target_columns)
    arrays: list[np.ndarray] = []
    names: list[str] = []
    _append_channel_features(arrays, names, masks["event_mask"], "oracle_event_mask", target_columns)
    _append_channel_features(arrays, names, masks["zero_mask"], "manual_zero_mask", target_columns)
    _append_channel_features(arrays, names, masks["peak_mask"], "manual_peak_mask", target_columns)
    if not arrays:
        return np.zeros((len(dates), 0), dtype=np.float32), []
    return np.concatenate(arrays, axis=1).astype(np.float32), names


def generate_llm_features(timestamps, rules):
    """Backward-compatible alias for rule-only LLM features."""
    return generate_llm_rule_features(timestamps, rules)


def _append_channel_features(arrays, names, values: np.ndarray, base_name: str, target_columns: list[str] | None):
    if values.shape[1] == 1 and not target_columns:
        arrays.append(values)
        names.append(base_name)
        return
    columns = target_columns or [str(idx) for idx in range(values.shape[1])]
    for idx, column in enumerate(columns):
        arrays.append(values[:, idx : idx + 1])
        names.append(f"{base_name}_{column}")


def _days_to_next_event(dates: pd.DatetimeIndex, mask: np.ndarray) -> np.ndarray:
    event_positions = np.flatnonzero(mask > 0)
    if len(event_positions) == 0:
        return np.zeros(len(dates), dtype=np.float32)
    result = np.zeros(len(dates), dtype=np.float32)
    for idx in range(len(dates)):
        pos = np.searchsorted(event_positions, idx, side="left")
        if pos >= len(event_positions):
            result[idx] = float(len(dates) - idx)
        else:
            delta = dates[event_positions[pos]] - dates[idx]
            result[idx] = max(0.0, delta.total_seconds() / 86400.0)
    return result.astype(np.float32)


def _hour_distance(hours: np.ndarray, target_hour: int) -> np.ndarray:
    raw = np.abs(hours.astype(np.float32) - float(target_hour))
    return np.minimum(raw, 24.0 - raw) / 12.0


def _column(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def _to_datetime_index(timestamps) -> pd.DatetimeIndex:
    if isinstance(timestamps, pd.DatetimeIndex):
        return timestamps
    return pd.DatetimeIndex(pd.to_datetime(timestamps))
