"""Generate deterministic event masks from offline LLM rules."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from llm_rules.rule_parser import parse_llm_rules
from llm_rules.rule_schema import RulePattern


def generate_event_mask(timestamps, rules, target_columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Generate per-pattern and aggregate masks for timestamps.

    The LLM is not called here. This function only consumes a pre-generated
    rules object or JSON payload and turns conditions into arrays.
    """
    dates = _to_datetime_index(timestamps)
    parsed = parse_llm_rules(rules)
    channels = max(1, len(target_columns or []))
    zeros = np.zeros((len(dates), channels), dtype=np.float32)
    if parsed is None:
        return {"event_mask": zeros.copy(), "zero_mask": zeros.copy(), "peak_mask": zeros.copy()}

    mask_dict: dict[str, np.ndarray] = {}
    event_union = zeros.copy()
    zero_union = zeros.copy()
    peak_union = zeros.copy()

    for pattern in parsed.patterns:
        base_mask = _condition_mask(dates, pattern).astype(np.float32).reshape(-1, 1)
        channel_mask = _affected_variable_mask(pattern, target_columns)
        mask = base_mask * channel_mask.reshape(1, -1)
        mask_dict[pattern.name] = mask
        if _is_event_pattern(pattern):
            event_union = np.maximum(event_union, mask)
        if pattern.type == "zero_event":
            zero_union = np.maximum(zero_union, mask)
        if pattern.type == "peak_event":
            peak_union = np.maximum(peak_union, mask)

    mask_dict["event_mask"] = event_union
    mask_dict["zero_mask"] = zero_union
    mask_dict["peak_mask"] = peak_union
    return mask_dict


def _condition_mask(dates: pd.DatetimeIndex, pattern: RulePattern) -> np.ndarray:
    condition = pattern.condition
    kind = condition.get("kind")
    if kind == "calendar_periodic":
        day = int(condition.get("day", 1))
        month_interval = max(1, int(condition.get("month_interval", 1)))
        if "anchor" not in condition:
            warnings.warn(
                f"calendar_periodic rule '{pattern.name}' has no anchor; using first timestamp as anchor.",
                UserWarning,
                stacklevel=2,
            )
            anchor = dates[0]
        else:
            anchor = pd.Timestamp(condition["anchor"])
        month_offset = (dates.year - anchor.year) * 12 + (dates.month - anchor.month)
        month_ok = (month_offset >= 0) & ((month_offset % month_interval) == 0)
        return (dates.day == day) & month_ok
    if kind == "calendar_window":
        center_day = int(condition.get("center_day", condition.get("day", 1)))
        center_hour = int(condition.get("center_hour", 0))
        month_interval = max(1, int(condition.get("month_interval", 1)))
        window = pd.Timedelta(hours=float(condition.get("window_hours", 24)))
        anchor = pd.Timestamp(condition.get("anchor", dates[0]))
        month_offset = (dates.year - anchor.year) * 12 + (dates.month - anchor.month)
        month_ok = (month_offset >= 0) & ((month_offset % month_interval) == 0)
        centers = pd.DatetimeIndex(
            [_safe_month_day(ts.year, ts.month, center_day) + pd.Timedelta(hours=center_hour) for ts in dates]
        )
        return month_ok & (np.abs(dates - centers) <= window)
    if kind == "hourly":
        return dates.hour == int(condition.get("hour", 0))
    if kind == "weekday":
        return dates.weekday == int(condition.get("weekday", 0))
    raise ValueError(f"Unsupported rule condition kind: {kind}")


def _affected_variable_mask(pattern: RulePattern, target_columns: list[str] | None) -> np.ndarray:
    channels = max(1, len(target_columns or []))
    affected = pattern.affected_variables
    if affected == "all" or target_columns is None:
        return np.ones(channels, dtype=np.float32)
    mask = np.zeros(channels, dtype=np.float32)
    for variable in affected:
        if variable in target_columns:
            mask[target_columns.index(variable)] = 1.0
        else:
            warnings.warn(
                f"Rule '{pattern.name}' references affected variable '{variable}' not present in target columns.",
                UserWarning,
                stacklevel=2,
            )
    return mask


def _is_event_pattern(pattern: RulePattern) -> bool:
    if pattern.type in {"zero_event", "peak_event", "calendar_event"}:
        return True
    if pattern.features.get("event_mask") or pattern.features.get("peak_mask"):
        return True
    return "event_weighted_mse" in pattern.losses


def _to_datetime_index(timestamps) -> pd.DatetimeIndex:
    if isinstance(timestamps, pd.DatetimeIndex):
        return timestamps
    return pd.DatetimeIndex(pd.to_datetime(timestamps))


def _safe_month_day(year: int, month: int, day: int) -> pd.Timestamp:
    last_day = pd.Period(f"{year:04d}-{month:02d}").days_in_month
    return pd.Timestamp(year=year, month=month, day=min(day, last_day))
