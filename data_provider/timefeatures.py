"""Calendar feature extraction for timestamp marks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_features(dates, freq: str = "h") -> np.ndarray:
    """Return normalized calendar marks for a sequence of timestamps."""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    minute = index.minute.astype(np.float32) / 59.0
    hour = index.hour.astype(np.float32) / 23.0
    weekday = index.weekday.astype(np.float32) / 6.0
    day = (index.day.astype(np.float32) - 1.0) / 30.0
    month = (index.month.astype(np.float32) - 1.0) / 11.0

    if freq.lower().startswith("t") or freq.lower().startswith("min"):
        return np.stack([month, day, weekday, hour, minute], axis=1).astype(np.float32)
    return np.stack([month, day, weekday, hour], axis=1).astype(np.float32)

