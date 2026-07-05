"""Detect train-split rule candidates used as evidence for offline LLM mining."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from llm_miner.common import default_output_dir, load_train_frame, select_value_columns, write_json


def detect_candidates(
    root_path: str,
    data_path: str,
    data: str,
    features: str = "M",
    target: str = "OT",
    seq_len: int = 96,
    pred_len: int = 96,
    output_dir: str | None = None,
    near_zero_eps: float = 1e-5,
):
    """Find low-frequency temporal rule candidates from the train split only."""
    train, date_col, _ = load_train_frame(root_path, data_path, data, seq_len)
    variable_names, target_columns = select_value_columns(train, date_col, features, target)

    zero_candidates = _zero_event_candidates(train, date_col, variable_names, near_zero_eps)
    periodic_candidates = _periodic_calendar_candidates(train, date_col, target_columns, near_zero_eps)
    peak_candidates = _peak_candidates(train, date_col, target_columns)
    long_tail_candidates = _long_tail_candidates(zero_candidates, peak_candidates, len(train))

    payload = {
        "dataset_name": data,
        "analysis_scope": "train_only",
        "data_path": data_path,
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "near_zero_eps": float(near_zero_eps),
        "zero_event_candidates": zero_candidates,
        "periodic_calendar_candidates": periodic_candidates,
        "peak_event_candidates": peak_candidates,
        "long_tail_candidates": long_tail_candidates,
    }
    output = default_output_dir(data, output_dir) / "candidate_rules.json"
    write_json(output, payload)
    return payload


def _zero_event_candidates(frame, date_col: str, columns: list[str], eps: float, limit: int = 50):
    candidates = []
    for col in columns:
        values = frame[col].astype(float)
        mask = values.abs().to_numpy() <= eps
        for start, end in _runs(mask):
            duration = int(end - start + 1)
            candidates.append(
                {
                    "pattern_type": "zero_event_candidate",
                    "start_time": str(frame[date_col].iloc[start]),
                    "end_time": str(frame[date_col].iloc[end]),
                    "duration": duration,
                    "affected_variables": [col],
                    "zero_ratio": float(mask[start : end + 1].mean()),
                    "mean_abs_value": float(values.iloc[start : end + 1].abs().mean()),
                    "support_evidence": [
                        [str(frame[date_col].iloc[start]), str(frame[date_col].iloc[end])]
                    ],
                }
            )
    return sorted(candidates, key=lambda item: (-item["duration"], item["mean_abs_value"]))[:limit]


def _periodic_calendar_candidates(frame, date_col: str, columns: list[str], eps: float):
    candidates = []
    dates = frame[date_col]
    zero_any = frame[columns].abs().le(eps).any(axis=1)
    checks = [
        ("day", 1, dates.dt.day == 1),
        ("weekday", 0, dates.dt.weekday == 0),
    ]
    for hour in sorted(dates.dt.hour.unique()):
        checks.append(("hour", int(hour), dates.dt.hour == hour))
    for kind, value, mask in checks:
        support = int(mask.sum())
        if support <= 0:
            continue
        precision = float(zero_any[mask].mean()) if support else 0.0
        candidates.append(
            {
                "condition_candidate": {"kind": "calendar_periodic" if kind == "day" else kind, kind: int(value)},
                "support_count": support,
                "coverage_ratio": float(support / len(frame)),
                "precision_on_train": precision,
                "evidence_windows": _evidence_windows(frame, date_col, mask.to_numpy(), limit=5),
            }
        )
    if dates.dt.month.nunique() >= 3:
        first_month = int(dates.dt.month.iloc[0])
        month_offset = (dates.dt.year - dates.dt.year.iloc[0]) * 12 + (dates.dt.month - first_month)
        mask = (month_offset % 2 == 0) & (dates.dt.day == 1)
        candidates.append(
            {
                "condition_candidate": {
                    "kind": "calendar_periodic",
                    "anchor": str(dates.iloc[0]),
                    "month_interval": 2,
                    "day": 1,
                },
                "support_count": int(mask.sum()),
                "coverage_ratio": float(mask.mean()),
                "precision_on_train": float(zero_any[mask].mean()) if mask.sum() else 0.0,
                "evidence_windows": _evidence_windows(frame, date_col, mask.to_numpy(), limit=5),
            }
        )
    return sorted(candidates, key=lambda item: (-item["precision_on_train"], -item["support_count"]))[:30]


def _peak_candidates(frame, date_col: str, columns: list[str]):
    candidates = []
    for col in columns:
        hourly = frame.groupby(frame[date_col].dt.hour)[col].mean()
        if hourly.empty:
            continue
        peak_hour = int(hourly.idxmax())
        baseline = float(hourly.median())
        strength = float(hourly.max() - baseline)
        mask = frame[date_col].dt.hour == peak_hour
        candidates.append(
            {
                "pattern_type": "peak_event_candidate",
                "condition_candidate": {"kind": "hourly", "hour": peak_hour},
                "peak_hour": peak_hour,
                "affected_variables": [col],
                "mean_peak_strength": strength,
                "support_count": int(mask.sum()),
                "evidence_windows": _evidence_windows(frame, date_col, mask.to_numpy(), limit=5),
            }
        )
    return candidates


def _long_tail_candidates(zero_candidates: list[dict], peak_candidates: list[dict], total_len: int):
    candidates = []
    for candidate in zero_candidates[:20]:
        rarity = 1.0 - min(1.0, candidate["duration"] / max(1, total_len))
        candidates.append(
            {
                "pattern_type": "rare_zero_regime",
                "rarity_score": float(rarity),
                "importance_score": float(rarity * candidate["zero_ratio"]),
                "candidate_loss_templates": ["event_weighted_mse", "zero_consistency"],
                "source": candidate,
            }
        )
    for candidate in peak_candidates:
        rarity = 1.0 - min(1.0, candidate["support_count"] / max(1, total_len))
        candidates.append(
            {
                "pattern_type": "rare_peak_regime",
                "rarity_score": float(rarity),
                "importance_score": float(max(0.0, candidate["mean_peak_strength"]) * rarity),
                "candidate_loss_templates": ["event_weighted_mse", "peak_shape"],
                "source": candidate,
            }
        )
    return sorted(candidates, key=lambda item: -item["importance_score"])[:30]


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


def _evidence_windows(frame, date_col: str, mask: np.ndarray, limit: int = 5):
    windows = []
    for start, end in _runs(mask):
        windows.append([str(frame[date_col].iloc[start]), str(frame[date_col].iloc[end])])
        if len(windows) >= limit:
            break
    return windows


def main():
    parser = argparse.ArgumentParser(description="Detect train-only rule candidates for LLM mining.")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--near_zero_eps", type=float, default=1e-5)
    args = parser.parse_args()
    detect_candidates(**vars(args))


if __name__ == "__main__":
    main()

