"""Create auxiliary features from offline LLM rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_rules.mask_generator import generate_event_mask
from llm_rules.rule_parser import parse_llm_rules


def generate_llm_features(timestamps, rules):
    """Generate deterministic features that can be concatenated to inputs."""
    dates = _to_datetime_index(timestamps)
    parsed = parse_llm_rules(rules)
    if parsed is None:
        return np.zeros((len(dates), 0), dtype=np.float32), []

    mask_dict = generate_event_mask(dates, parsed)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    for pattern in parsed.patterns:
        mask = mask_dict[pattern.name].astype(np.float32)
        if pattern.features.get("event_mask"):
            arrays.append(mask)
            names.append(f"event_mask_{pattern.name}")
        if pattern.features.get("peak_mask"):
            arrays.append(mask)
            names.append(f"peak_mask_{pattern.name}")
        if pattern.features.get("days_to_event"):
            arrays.append(_days_to_next_event(dates, mask[:, 0]).reshape(-1, 1))
            names.append(f"days_to_event_{pattern.name}")
        if pattern.features.get("hour_distance_to_peak"):
            hour = int(pattern.condition.get("hour", 0))
            arrays.append(_hour_distance(dates.hour.to_numpy(), hour).reshape(-1, 1))
            names.append(f"hour_distance_to_peak_{pattern.name}")

    interval = _first_calendar_interval(parsed)
    generic = [
        (_column(dates.day == 1), "first_day_indicator"),
        (_column(((dates.month - 1) % interval) / max(1, interval - 1)), "month_mod_feature"),
        (_column(dates.hour / 23.0), "hour_of_day"),
        (_column(dates.weekday / 6.0), "day_of_week"),
    ]
    for values, name in generic:
        arrays.append(values)
        names.append(name)

    if not arrays:
        return np.zeros((len(dates), 0), dtype=np.float32), []
    return np.concatenate(arrays, axis=1).astype(np.float32), names


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


def _first_calendar_interval(rules) -> int:
    for pattern in rules.patterns:
        condition = pattern.condition
        if condition.get("kind") == "calendar_periodic":
            return max(1, int(condition.get("month_interval", 1)))
    return 12


def _to_datetime_index(timestamps) -> pd.DatetimeIndex:
    if isinstance(timestamps, pd.DatetimeIndex):
        return timestamps
    return pd.DatetimeIndex(pd.to_datetime(timestamps))
