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
        period_mask = _periods_mask(dates, condition)
        if period_mask is not None:
            return period_mask
        month_values = _condition_values(
            condition,
            exact_keys=("months", "month"),
            high_low_keys=("high_months", "low_months"),
            fallback_keys=("shoulder_months",),
        )
        hour_values = _condition_values(
            condition,
            exact_keys=("hours", "hour"),
            high_low_keys=("high_hours", "low_hours"),
        )
        day_values = _condition_values(condition, exact_keys=("days", "day"))
        if month_values or hour_values or day_values:
            mask = np.ones(len(dates), dtype=bool)
            if month_values:
                mask &= np.isin(np.asarray(dates.month), month_values)
            if hour_values:
                mask &= np.isin(np.asarray(dates.hour), hour_values)
            if day_values:
                mask &= np.isin(np.asarray(dates.day), day_values)
            if "month_interval" in condition or "anchor" in condition:
                anchor = pd.Timestamp(condition.get("anchor", dates[0]))
                month_interval = max(1, int(condition.get("month_interval", 1)))
                month_offset = (dates.year - anchor.year) * 12 + (dates.month - anchor.month)
                mask &= (month_offset >= 0) & ((month_offset % month_interval) == 0)
            return mask
        return _legacy_calendar_periodic_mask(dates, pattern)
    if kind == "calendar_window":
        explicit = _explicit_window_mask(dates, condition)
        if explicit is not None:
            return explicit
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
        hours = _condition_values(
            condition,
            exact_keys=("hours", "hour"),
            high_low_keys=("high_hours", "low_hours"),
        )
        if not hours:
            hours = [0]
        return np.isin(np.asarray(dates.hour), hours)
    if kind == "weekday":
        weekdays = _condition_values(
            condition,
            exact_keys=("weekdays", "weekday"),
            high_low_keys=("high_weekdays", "low_weekdays"),
        )
        if not weekdays:
            weekdays = [0]
        return np.isin(np.asarray(dates.weekday), weekdays)
    raise ValueError(f"Unsupported rule condition kind: {kind}")


def _legacy_calendar_periodic_mask(dates: pd.DatetimeIndex, pattern: RulePattern) -> np.ndarray:
    condition = pattern.condition
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


def _periods_mask(dates: pd.DatetimeIndex, condition: dict) -> np.ndarray | None:
    periods = condition.get("periods")
    if not isinstance(periods, list):
        return None

    union = np.zeros(len(dates), dtype=bool)
    for period in periods:
        if not isinstance(period, dict):
            continue
        mask = np.ones(len(dates), dtype=bool)
        constrained = False
        month_values = _condition_values(
            period,
            exact_keys=("months", "month"),
            high_low_keys=("high_months", "low_months"),
            fallback_keys=("shoulder_months",),
        )
        hour_values = _condition_values(
            period,
            exact_keys=("hours", "hour"),
            high_low_keys=("high_hours", "low_hours"),
        )
        day_values = _condition_values(period, exact_keys=("days", "day"))
        if month_values:
            mask &= np.isin(np.asarray(dates.month), month_values)
            constrained = True
        if hour_values:
            mask &= np.isin(np.asarray(dates.hour), hour_values)
            constrained = True
        if day_values:
            mask &= np.isin(np.asarray(dates.day), day_values)
            constrained = True
        if constrained:
            union |= mask
    return union


def _explicit_window_mask(dates: pd.DatetimeIndex, condition: dict) -> np.ndarray | None:
    windows = condition.get("windows")
    if windows is None and ("date_start" in condition or "date_end" in condition or "start" in condition or "end" in condition):
        windows = [
            {
                "start": condition.get("date_start", condition.get("start")),
                "end": condition.get("date_end", condition.get("end")),
            }
        ]
    if not windows:
        return None

    mask = np.zeros(len(dates), dtype=bool)
    for window in windows:
        if not isinstance(window, dict):
            continue
        start = window.get("start", window.get("date_start"))
        end = window.get("end", window.get("date_end"))
        if not start or not end:
            continue
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return mask


def _condition_values(
    condition: dict,
    exact_keys: tuple[str, ...],
    high_low_keys: tuple[str, ...] = (),
    fallback_keys: tuple[str, ...] = (),
) -> list[int]:
    values: list[int] = []
    for key in exact_keys:
        values.extend(_as_int_list(condition.get(key)))
    if not values:
        for key in high_low_keys:
            values.extend(_as_int_list(condition.get(key)))
    if not values:
        for key in fallback_keys:
            values.extend(_as_int_list(condition.get(key)))
    return sorted(set(values))


def _as_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


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
