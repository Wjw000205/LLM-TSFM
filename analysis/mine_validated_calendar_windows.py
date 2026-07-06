"""Mine train/val-validated calendar-window rules without using test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider
EPS = 1e-8


def mine_validated_calendar_windows(
    data: str = "ETTm1",
    root_path: str = "./data/",
    data_path: str = "ETTm1.csv",
    features: str = "M",
    target: str = "OT",
    seq_len: int = 336,
    label_len: int = 48,
    pred_len: int = 96,
    near_zero_quantile: float = 0.05,
    center_days: str = "1,2,3,28,29,30,31",
    center_hours: str = "0,3,6,9,12,15,18,21",
    window_hours_grid: str = "0,1,2,4,6,12",
    month_intervals: str = "1,2",
    min_train_support: int = 4,
    min_val_support: int = 2,
    min_train_precision: float = 0.25,
    min_val_precision: float = 0.25,
    min_val_recall: float = 0.02,
    max_candidates: int = 10,
    output_rule_path: str = "./llm_rules/validated_rules/ETTm1_rules_mined_calendar_windows.json",
    output_report_path: str = "./artifacts/core_results/ettm1_mined_calendar_windows_report.json",
):
    args = _build_args(
        data=data,
        root_path=root_path,
        data_path=data_path,
        features=features,
        target=target,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
    )
    train_data, _ = data_provider(args, "train")
    val_data, _ = data_provider(args, "val")
    thresholds = _train_near_zero_threshold(train_data, near_zero_quantile)
    train_event = _near_zero_mask(train_data.data_y, thresholds, train_data.zero_target)
    val_event = _near_zero_mask(val_data.data_y, thresholds, train_data.zero_target)
    timestamps = train_data.timestamps.append(val_data.timestamps)
    train_event_full = np.concatenate([train_event, np.zeros_like(val_event)], axis=0)
    val_event_full = np.concatenate([np.zeros_like(train_event), val_event], axis=0)
    train_scope = np.concatenate(
        [np.ones(len(train_event), dtype=bool), np.zeros(len(val_event), dtype=bool)],
        axis=0,
    )
    val_scope = np.concatenate(
        [np.zeros(len(train_event), dtype=bool), np.ones(len(val_event), dtype=bool)],
        axis=0,
    )
    anchor = str(train_data.timestamps[0].floor("D"))
    result = mine_calendar_window_candidates(
        timestamps=timestamps,
        train_event_mask=train_event_full,
        val_event_mask=val_event_full,
        train_scope_mask=train_scope,
        val_scope_mask=val_scope,
        target_columns=train_data.target_columns,
        anchor=anchor,
        center_days=_parse_ints(center_days),
        center_hours=_parse_ints(center_hours),
        window_hours_grid=_parse_floats(window_hours_grid),
        month_intervals=_parse_ints(month_intervals),
        min_train_support=min_train_support,
        min_val_support=min_val_support,
        min_train_precision=min_train_precision,
        min_val_precision=min_val_precision,
        min_val_recall=min_val_recall,
        max_candidates=max_candidates,
    )
    report = {
        "data": data,
        "analysis_scope": "train_val_validated",
        "test_usage": "not_used_for_selection",
        "near_zero_quantile": near_zero_quantile,
        "min_train_precision": min_train_precision,
        "min_val_precision": min_val_precision,
        "min_val_recall": min_val_recall,
        "threshold_by_channel": {
            channel: float(value) for channel, value in zip(train_data.target_columns, thresholds)
        },
        **result,
    }
    rule_payload = build_rule_payload(data, result["selected_candidates"])
    _write_json(output_rule_path, rule_payload)
    _write_json(output_report_path, report)
    return {"rules": rule_payload, "report": report}


def mine_calendar_window_candidates(
    timestamps,
    train_event_mask: np.ndarray,
    val_event_mask: np.ndarray,
    target_columns: list[str],
    anchor: str,
    center_days: list[int],
    center_hours: list[int],
    window_hours_grid: list[float],
    month_intervals: list[int],
    train_scope_mask: np.ndarray | None = None,
    val_scope_mask: np.ndarray | None = None,
    min_train_support: int = 1,
    min_val_support: int = 1,
    min_train_precision: float = 0.25,
    min_val_precision: float = 0.25,
    min_val_recall: float = 0.0,
    max_candidates: int = 10,
) -> dict:
    dates = pd.DatetimeIndex(pd.to_datetime(timestamps))
    train_event_mask = np.asarray(train_event_mask, dtype=np.float32)
    val_event_mask = np.asarray(val_event_mask, dtype=np.float32)
    train_scope = _scope_column(train_scope_mask, len(dates))
    val_scope = _scope_column(val_scope_mask, len(dates))
    anchor_ts = pd.Timestamp(anchor)
    month_offset = (dates.year - anchor_ts.year) * 12 + (dates.month - anchor_ts.month)
    month_ok_cache = {
        month_interval: np.asarray((month_offset >= 0) & ((month_offset % int(month_interval)) == 0), dtype=bool)
        for month_interval in month_intervals
    }
    delta_hours_cache = {
        (center_day, center_hour): _calendar_delta_hours(dates, center_day, center_hour)
        for center_day in center_days
        for center_hour in center_hours
    }
    rows = []
    for center_day in center_days:
        for center_hour in center_hours:
            for window_hours in window_hours_grid:
                for month_interval in month_intervals:
                    condition = {
                        "kind": "calendar_window",
                        "anchor": anchor,
                        "month_interval": int(month_interval),
                        "center_day": int(center_day),
                        "center_hour": int(center_hour),
                        "window_hours": float(window_hours),
                    }
                    mask = (
                        month_ok_cache[month_interval]
                        & (delta_hours_cache[(center_day, center_hour)] <= float(window_hours))
                    ).astype(np.float32).reshape(-1, 1)
                    for channel_idx, channel in enumerate(target_columns):
                        predicted = mask
                        train_metrics = _binary_metrics(
                            predicted * train_scope,
                            train_event_mask[:, channel_idx : channel_idx + 1] * train_scope,
                        )
                        val_metrics = _binary_metrics(
                            predicted * val_scope,
                            val_event_mask[:, channel_idx : channel_idx + 1] * val_scope,
                        )
                        row = {
                            "name": _candidate_name(center_day, center_hour, window_hours, month_interval, channel),
                            "condition": condition,
                            "affected_variables": [channel],
                            "channel": channel,
                            "train_support": train_metrics["tp"],
                            "train_predicted": train_metrics["predicted"],
                            "train_precision": train_metrics["precision"],
                            "train_recall": train_metrics["recall"],
                            "train_f1": train_metrics["f1"],
                            "val_support": val_metrics["tp"],
                            "val_predicted": val_metrics["predicted"],
                            "val_precision": val_metrics["precision"],
                            "val_recall": val_metrics["recall"],
                            "val_f1": val_metrics["f1"],
                            "val_false_positive_ratio": val_metrics["false_positive_ratio"],
                        }
                        row["selected"] = bool(
                            row["train_support"] >= int(min_train_support)
                            and row["val_support"] >= int(min_val_support)
                            and row["train_precision"] >= float(min_train_precision)
                            and row["val_precision"] >= float(min_val_precision)
                            and row["val_recall"] >= float(min_val_recall)
                        )
                        rows.append(row)
    selected = [row for row in rows if row["selected"]]
    selected = sorted(
        selected,
        key=lambda row: (
            -row["val_f1"],
            -row["val_precision"],
            -row["val_support"],
            row["condition"]["window_hours"],
        ),
    )[: int(max_candidates)]
    return {
        "num_candidates": len(rows),
        "num_selected": len(selected),
        "selected_candidates": selected,
        "top_candidates": sorted(rows, key=lambda row: (-row["val_f1"], -row["val_precision"]))[: int(max_candidates)],
    }


def build_rule_payload(dataset_name: str, candidates: list[dict]) -> dict:
    patterns = []
    for candidate in candidates:
        patterns.append(
            {
                "name": candidate["name"],
                "type": "zero_event",
                "description": "Train/val-validated mined calendar window.",
                "condition": candidate["condition"],
                "affected_variables": candidate["affected_variables"],
                "time_range": "calendar_window",
                "support_count": int(candidate.get("val_support", 0)),
                "confidence": float(candidate.get("val_precision", 0.0)),
                "losses": {
                    "event_weighted_mse": {"enabled": True, "weight": 1.0},
                },
                "features": {
                    "event_mask": True,
                    "days_to_event": True,
                    "rule_confidence": float(candidate.get("val_precision", 0.0)),
                },
            }
        )
    return {
        "dataset_name": dataset_name,
        "analysis_scope": "train_val_validated",
        "warnings": ["Generated without using the test split for selection."],
        "patterns": patterns,
    }


def _binary_metrics(predicted_mask: np.ndarray, actual_mask: np.ndarray) -> dict[str, float | int]:
    predicted = np.asarray(predicted_mask) > 0
    actual = np.asarray(actual_mask) > 0
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, np.logical_not(actual)).sum())
    fn = int(np.logical_and(np.logical_not(predicted), actual).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted": int(tp + fp),
        "actual": int(tp + fn),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall > EPS else 0.0,
        "false_positive_ratio": fp / (tp + fp) if tp + fp > 0 else 0.0,
    }


def _scope_column(scope_mask: np.ndarray | None, length: int) -> np.ndarray:
    if scope_mask is None:
        return np.ones((length, 1), dtype=np.float32)
    scope = np.asarray(scope_mask, dtype=bool)
    if scope.ndim == 2:
        scope = scope.any(axis=1)
    if len(scope) != length:
        raise ValueError(f"scope_mask length {len(scope)} does not match timestamps length {length}")
    return scope.reshape(-1, 1).astype(np.float32)


def _calendar_delta_hours(dates: pd.DatetimeIndex, center_day: int, center_hour: int) -> np.ndarray:
    center_days = np.minimum(int(center_day), dates.days_in_month)
    centers = pd.to_datetime(
        {
            "year": dates.year,
            "month": dates.month,
            "day": center_days,
            "hour": np.full(len(dates), int(center_hour), dtype=np.int16),
        }
    )
    return np.abs((dates - pd.DatetimeIndex(centers)) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float32)


def _train_near_zero_threshold(train_data, quantile: float) -> np.ndarray:
    zero_target = np.asarray(train_data.zero_target, dtype=np.float32).reshape(1, -1)
    distance = np.abs(train_data.data_y - zero_target)
    return np.quantile(distance, float(quantile), axis=0).astype(np.float32)


def _near_zero_mask(values: np.ndarray, thresholds: np.ndarray, zero_target: np.ndarray) -> np.ndarray:
    zero_target = np.asarray(zero_target, dtype=np.float32).reshape(1, -1)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (np.abs(values - zero_target) <= thresholds).astype(np.float32)


def _single_rule_payload(name: str, condition: dict, affected_variables) -> dict:
    return {
        "dataset_name": "candidate",
        "patterns": [
            {
                "name": name,
                "type": "zero_event",
                "condition": condition,
                "affected_variables": affected_variables,
                "features": {"event_mask": True},
            }
        ],
    }


def _candidate_name(center_day: int, center_hour: int, window_hours: float, month_interval: int, channel: str) -> str:
    return f"calendar_window_d{center_day}_h{center_hour}_w{window_hours:g}_m{month_interval}_{channel}"


def _build_args(**kwargs):
    config = dict(kwargs)
    config.update(
        {
            "batch_size": 8,
            "num_workers": 0,
            "use_zscore": 1,
            "use_llm_features": 0,
            "use_llm_rule_features": 0,
            "use_standard_time_features": 0,
            "use_oracle_features": 0,
            "freq": "t" if str(config.get("data", "")).startswith("ETTm") else "h",
            "timeenc": 0,
        }
    )
    return SimpleNamespace(**config)


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _write_json(path: str, payload: dict):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Mine train/val-validated calendar-window rules.")
    parser.add_argument("--data", default="ETTm1")
    parser.add_argument("--root_path", default="./data/")
    parser.add_argument("--data_path", default="ETTm1.csv")
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--near_zero_quantile", type=float, default=0.05)
    parser.add_argument("--center_days", default="1,2,3,28,29,30,31")
    parser.add_argument("--center_hours", default="0,3,6,9,12,15,18,21")
    parser.add_argument("--window_hours_grid", default="0,1,2,4,6,12")
    parser.add_argument("--month_intervals", default="1,2")
    parser.add_argument("--min_train_support", type=int, default=4)
    parser.add_argument("--min_val_support", type=int, default=2)
    parser.add_argument("--min_train_precision", type=float, default=0.25)
    parser.add_argument("--min_val_precision", type=float, default=0.25)
    parser.add_argument("--min_val_recall", type=float, default=0.02)
    parser.add_argument("--max_candidates", type=int, default=10)
    parser.add_argument("--output_rule_path", default="./llm_rules/validated_rules/ETTm1_rules_mined_calendar_windows.json")
    parser.add_argument("--output_report_path", default="./artifacts/core_results/ettm1_mined_calendar_windows_report.json")
    args = parser.parse_args()
    mine_validated_calendar_windows(**vars(args))


if __name__ == "__main__":
    main()
