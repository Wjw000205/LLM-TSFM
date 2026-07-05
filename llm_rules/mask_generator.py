"""Generate deterministic event masks from offline LLM rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_rules.rule_parser import parse_llm_rules
from llm_rules.rule_schema import LLMRules, RulePattern


def generate_event_mask(timestamps, rules) -> dict[str, np.ndarray]:
    """Generate per-pattern and aggregate masks for timestamps.

    The LLM is not called here. This function only consumes a pre-generated
    rules object or JSON payload and turns conditions into arrays.
    """
    dates = _to_datetime_index(timestamps)
    parsed = parse_llm_rules(rules)
    zeros = np.zeros((len(dates), 1), dtype=np.float32)
    if parsed is None:
        return {"event_mask": zeros.copy(), "zero_mask": zeros.copy(), "peak_mask": zeros.copy()}

    mask_dict: dict[str, np.ndarray] = {}
    event_union = zeros.copy()
    zero_union = zeros.copy()
    peak_union = zeros.copy()

    for pattern in parsed.patterns:
        mask = _condition_mask(dates, pattern).astype(np.float32).reshape(-1, 1)
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
        month_ok = ((dates.month - 1) % month_interval) == 0
        return (dates.day == day) & month_ok
    if kind == "hourly":
        return dates.hour == int(condition.get("hour", 0))
    if kind == "weekday":
        return dates.weekday == int(condition.get("weekday", 0))
    raise ValueError(f"Unsupported rule condition kind: {kind}")


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

