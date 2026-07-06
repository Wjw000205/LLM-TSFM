"""Diagnose false positives in offline LLM rule features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider


EPS = 1e-8


def diagnose_llm_feature_false_positives(
    data: str = "ETTm1",
    root_path: str = "./data/",
    data_path: str = "ETTm1.csv",
    features: str = "M",
    target: str = "OT",
    seq_len: int = 336,
    label_len: int = 48,
    pred_len: int = 96,
    llm_rule_path: str = "./llm_rules/example_rules/ETTm1_rules.json",
    near_zero_quantile: float = 0.05,
    days_to_event_thresholds: list[float] | None = None,
    output_path: str = "artifacts/core_results/ettm1_llm_feature_false_positive_report.json",
    output_markdown_path: str = "docs/llm_feature_false_positive_diagnosis.md",
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
        llm_rule_path=llm_rule_path,
    )
    train_data, _ = data_provider(args, "train")
    thresholds = _train_near_zero_threshold(train_data, near_zero_quantile)
    report = {
        "data": data,
        "llm_rule_path": llm_rule_path,
        "near_zero_definition": {
            "source_split": "train",
            "quantile": near_zero_quantile,
            "threshold_by_channel_normalized_abs_to_zero_target": {
                channel: float(value) for channel, value in zip(train_data.target_columns, thresholds)
            },
        },
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        dataset, _ = data_provider(args, split)
        report["splits"][split] = _diagnose_split(
            dataset=dataset,
            thresholds=thresholds,
            zero_target=train_data.zero_target,
            days_to_event_thresholds=days_to_event_thresholds or [0.0, 0.25, 1.0, 2.0],
        )
    report["summary"] = _summary(report)
    _write_json(output_path, report)
    _write_markdown(output_markdown_path, report)
    return report


def binary_confusion_metrics(predicted, actual) -> dict[str, float | int]:
    predicted = np.asarray(predicted) > 0
    actual = np.asarray(actual) > 0
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, np.logical_not(actual)).sum())
    fn = int(np.logical_and(np.logical_not(predicted), actual).sum())
    tn = int(np.logical_and(np.logical_not(predicted), np.logical_not(actual)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / (precision + recall) if precision + recall > EPS else 0.0,
        "false_positive_rate": fpr,
        "false_positive_ratio_among_predicted": fp / (tp + fp) if tp + fp > 0 else 0.0,
        "support_predicted": int(tp + fp),
        "support_actual": int(tp + fn),
    }


def _diagnose_split(dataset, thresholds: np.ndarray, zero_target: np.ndarray, days_to_event_thresholds: list[float]):
    true_event = _near_zero_mask(dataset.data_y, thresholds, zero_target)
    event_mask = dataset.event_masks[:, 0, :]
    zero_mask = dataset.event_masks[:, 1, :]
    aggregate_event = binary_confusion_metrics(event_mask, true_event)
    aggregate_zero = binary_confusion_metrics(zero_mask, true_event)
    per_channel = {}
    for idx, channel in enumerate(dataset.target_columns):
        per_channel[channel] = {
            "event_feature": binary_confusion_metrics(event_mask[:, idx], true_event[:, idx]),
            "zero_event_feature": binary_confusion_metrics(zero_mask[:, idx], true_event[:, idx]),
        }
    days_to_event = _days_to_event_feature(dataset)
    days_metrics = {}
    if days_to_event is not None:
        actual_any = true_event.max(axis=1)
        for threshold in days_to_event_thresholds:
            predicted = days_to_event <= float(threshold)
            days_metrics[str(threshold)] = binary_confusion_metrics(predicted, actual_any)
    anchor_rows = _anchor_false_positive_rows(dataset, true_event)
    return {
        "time_range": {"start": str(dataset.timestamps[0]), "end": str(dataset.timestamps[-1])},
        "feature_names": list(dataset.llm_feature_names),
        "aggregate_event_feature": aggregate_event,
        "aggregate_zero_event_feature": aggregate_zero,
        "per_channel": per_channel,
        "days_to_event_threshold_metrics": days_metrics,
        "anchor_false_positive_rows": anchor_rows,
    }


def _train_near_zero_threshold(train_data, quantile: float) -> np.ndarray:
    zero_target = np.asarray(train_data.zero_target, dtype=np.float32).reshape(1, -1)
    distance = np.abs(train_data.data_y - zero_target)
    return np.quantile(distance, quantile, axis=0).astype(np.float32)


def _near_zero_mask(values: np.ndarray, thresholds: np.ndarray, zero_target: np.ndarray):
    zero_target = np.asarray(zero_target, dtype=np.float32).reshape(1, -1)
    return (np.abs(values - zero_target) <= thresholds.reshape(1, -1)).astype(np.float32)


def _days_to_event_feature(dataset):
    for idx, name in enumerate(dataset.llm_feature_names):
        if name.endswith("days_to_event_periodic_zero_day") or "days_to_event" in name:
            return dataset.llm_features[:, idx]
    return None


def _anchor_false_positive_rows(dataset, true_event):
    rows = []
    zero_any = dataset.event_masks[:, 1, :].max(axis=1) > 0
    dates = np.asarray(dataset.timestamps)
    for day in sorted({str(ts)[:10] for ts in dates[zero_any]}):
        day_mask = np.asarray([str(ts).startswith(day) for ts in dates])
        predicted = dataset.event_masks[day_mask, 1, :]
        actual = true_event[day_mask]
        metrics = binary_confusion_metrics(predicted, actual)
        rows.append(
            {
                "date": day,
                "predicted_points": metrics["support_predicted"],
                "true_points": metrics["support_actual"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "precision": metrics["precision"],
                "false_positive_ratio_among_predicted": metrics["false_positive_ratio_among_predicted"],
            }
        )
    return rows


def _summary(report):
    split_rows = {}
    for split, payload in report["splits"].items():
        event_metrics = payload["aggregate_event_feature"]
        zero_metrics = payload["aggregate_zero_event_feature"]
        split_rows[split] = {
            "event_precision": event_metrics["precision"],
            "event_false_positive_ratio": event_metrics["false_positive_ratio_among_predicted"],
            "event_recall": event_metrics["recall"],
            "zero_precision": zero_metrics["precision"],
            "zero_false_positive_ratio": zero_metrics["false_positive_ratio_among_predicted"],
            "zero_recall": zero_metrics["recall"],
        }
    test_fp = split_rows.get("test", {}).get("zero_false_positive_ratio", 0.0)
    return {
        "by_split": split_rows,
        "has_high_false_positive_risk": bool(test_fp >= 0.5),
        "recommendation": "disable_binary_llm_rule_features_as_deterministic_priors"
        if test_fp >= 0.5
        else "binary_llm_rule_features_have_acceptable_precision_for_diagnostic_use",
    }


def _build_args(**kwargs):
    config = dict(kwargs)
    config.update(
        {
            "batch_size": 8,
            "num_workers": 0,
            "use_zscore": 1,
            "use_llm_features": 0,
            "use_llm_rule_features": 1,
            "use_standard_time_features": 0,
            "use_oracle_features": 0,
            "freq": "h",
            "timeenc": 0,
        }
    )
    return SimpleNamespace(**config)


def _write_json(path: str, payload: dict):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown(path: str, report: dict):
    lines = [
        "# LLM Feature False Positive Diagnosis",
        "",
        f"- Dataset: `{report['data']}`.",
        f"- Rule path: `{report['llm_rule_path']}`.",
        f"- Near-zero threshold source: `{report['near_zero_definition']['source_split']}` "
        f"quantile `{report['near_zero_definition']['quantile']}`.",
        f"- High false-positive risk: `{report['summary']['has_high_false_positive_risk']}`.",
        f"- Recommendation: `{report['summary']['recommendation']}`.",
        "",
        "## Split Summary",
        "",
        "| Split | Event Precision | Event FP Ratio | Event Recall | Zero Precision | Zero FP Ratio | Zero Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test"]:
        row = report["summary"]["by_split"][split]
        lines.append(
            f"| {split} | {row['event_precision']:.4f} | {row['event_false_positive_ratio']:.4f} | "
            f"{row['event_recall']:.4f} | {row['zero_precision']:.4f} | "
            f"{row['zero_false_positive_ratio']:.4f} | {row['zero_recall']:.4f} |"
        )
    lines.extend(["", "## Per-channel Test Zero-event Feature", ""])
    lines.append("| Channel | Precision | FP Ratio | Recall | Predicted | Actual |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for channel, payload in report["splits"]["test"]["per_channel"].items():
        row = payload["zero_event_feature"]
        lines.append(
            f"| {channel} | {row['precision']:.4f} | {row['false_positive_ratio_among_predicted']:.4f} | "
            f"{row['recall']:.4f} | {row['support_predicted']} | {row['support_actual']} |"
        )
    lines.extend(["", "## Test Anchor False Positives", ""])
    lines.append("| Date | Predicted Points | True Points | TP | FP | Precision | FP Ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in report["splits"]["test"]["anchor_false_positive_rows"]:
        lines.append(
            f"| {row['date']} | {row['predicted_points']} | {row['true_points']} | {row['tp']} | "
            f"{row['fp']} | {row['precision']:.4f} | {row['false_positive_ratio_among_predicted']:.4f} |"
        )
    lines.extend(["", "## Days-to-event Feature", ""])
    lines.append("| Split | Threshold Days | Precision | FP Ratio | Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for split in ["train", "val", "test"]:
        for threshold, row in report["splits"][split]["days_to_event_threshold_metrics"].items():
            lines.append(
                f"| {split} | {threshold} | {row['precision']:.4f} | "
                f"{row['false_positive_ratio_among_predicted']:.4f} | {row['recall']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Downstream Enforcement",
            "",
            "`analysis/verify_and_calibrate_rules.py` applies a false-positive precision gate before "
            "enabling any calibrated rule prior. The actual event mask is defined only from train/val "
            "information as `abs(y - train_zero_target) <= train distance quantile`, with default "
            "`near_zero_quantile=0.05`.",
            "",
            "With `min_event_precision=0.5`, the current ETTm1 periodic zero-day rule is disabled for "
            "every channel on validation. Candidate priors are still reported for diagnosis, but "
            "`best_prior_type` remains `baseline`, `best_alpha` remains `0.0`, and the calibrated rule "
            "JSON has `enabled=false`.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Diagnose false positives in LLM rule features.")
    parser.add_argument("--data", default="ETTm1")
    parser.add_argument("--root_path", default="./data/")
    parser.add_argument("--data_path", default="ETTm1.csv")
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--llm_rule_path", default="./llm_rules/example_rules/ETTm1_rules.json")
    parser.add_argument("--near_zero_quantile", type=float, default=0.05)
    parser.add_argument("--output_path", default="artifacts/core_results/ettm1_llm_feature_false_positive_report.json")
    parser.add_argument("--output_markdown_path", default="docs/llm_feature_false_positive_diagnosis.md")
    args = parser.parse_args()
    diagnose_llm_feature_false_positives(**vars(args))


if __name__ == "__main__":
    main()
