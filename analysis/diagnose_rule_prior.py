"""Diagnose whether compiled rule priors match test-set targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.data_factory import data_provider
from llm_rules.mask_generator import generate_event_mask


def diagnose_rule_prior(
    result_dir: str,
    data: str = "ETTm1",
    rule_path: str = "./llm_rules/example_rules/ETTm1_rules.json",
    output_path: str = "artifacts/core_results/rule_prior_diagnosis.json",
):
    result_path = Path(result_dir)
    config = _read_json(result_path / "config.json")
    config["data"] = data
    config["llm_rule_path"] = rule_path
    config["use_llm_rule_features"] = 0
    config["use_standard_time_features"] = 0
    config["use_oracle_features"] = 0
    config["num_workers"] = 0
    args = SimpleNamespace(**config)
    train_data, _ = data_provider(args, "train")
    test_data, _ = data_provider(args, "test")

    pred = np.load(result_path / "pred_normalized.npy")
    true = np.load(result_path / "true_normalized.npy")
    masks, timestamps = _future_masks_and_timestamps(test_data)
    masks = masks[: pred.shape[0]]
    timestamps = timestamps[: pred.shape[0]]
    zero_mask = masks[:, :, 1, :]
    event_mask = masks[:, :, 0, :]
    zero_target = np.asarray(train_data.zero_target, dtype=np.float32).reshape(1, 1, -1)
    hard_pred = pred * (1.0 - zero_mask) + zero_mask * zero_target

    theoretical_zero_mse = _masked_mse(hard_pred, true, zero_mask)
    theoretical_event_mse = _masked_mse(hard_pred, true, event_mask)
    diagnosis = {
        "result_dir": str(result_path),
        "data": data,
        "rule_path": rule_path,
        "event_mask_sum": float(event_mask.sum()),
        "zero_mask_sum": float(zero_mask.sum()),
        "zero_timestamps_first20": _first_zero_timestamps(zero_mask, timestamps, test_data.target_columns),
        "zero_target": {col: float(value) for col, value in zip(test_data.target_columns, train_data.zero_target)},
        "mask_alignment": _mask_alignment(test_data),
        "timestamp_alignment_samples": _timestamp_alignment_samples(test_data),
        "zero_target_offset_scan": _zero_target_offset_scan(test_data, zero_mask, timestamps, zero_target),
        "true_at_zero_mask": _masked_stats(true, zero_mask, zero_target),
        "baseline_pred_at_zero_mask": _masked_stats(pred, zero_mask, zero_target),
        "theoretical_zero_event_mse": theoretical_zero_mse,
        "theoretical_event_mse": theoretical_event_mse,
        "hard_intervention_theoretical": {
            "zero_event_mse": theoretical_zero_mse,
            "event_mse": theoretical_event_mse,
        },
    }
    diagnosis["interpretation"] = _interpret(diagnosis)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    return diagnosis


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _future_masks_and_timestamps(dataset):
    masks = []
    timestamps = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        seq_y_masks = item[-1].numpy()
        masks.append(seq_y_masks[-dataset.pred_len :])
        s_end = idx + dataset.seq_len
        r_begin = s_end - dataset.label_len
        r_end = r_begin + dataset.label_len + dataset.pred_len
        timestamps.append([str(ts) for ts in dataset.timestamps[r_begin:r_end][-dataset.pred_len :]])
    return np.asarray(masks, dtype=np.float32), np.asarray(timestamps)


def _mask_alignment(dataset):
    recomputed = generate_event_mask(dataset.timestamps, dataset.llm_rule_path, target_columns=dataset.target_columns)
    stacked = np.stack([recomputed[name] for name in dataset.mask_names], axis=1).astype(np.float32)
    return {
        "dataset_mask_shape": list(dataset.event_masks.shape),
        "recomputed_mask_shape": list(stacked.shape),
        "max_abs_diff": float(np.max(np.abs(dataset.event_masks - stacked))) if len(stacked) else 0.0,
        "matches_recomputed_masks": bool(np.array_equal(dataset.event_masks, stacked)),
    }


def _timestamp_alignment_samples(dataset, indices=(672, 673, 674, 675)):
    samples = []
    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            continue
        s_end = idx + dataset.seq_len
        r_begin = s_end - dataset.label_len
        r_end = r_begin + dataset.label_len + dataset.pred_len
        pred_timestamps = dataset.timestamps[r_begin:r_end][-dataset.pred_len :]
        pred_masks = dataset.event_masks[r_begin:r_end][-dataset.pred_len :]
        hits = np.argwhere(pred_masks[:, 1, :] > 0)
        samples.append(
            {
                "window_index": int(idx),
                "pred_start": str(pred_timestamps[0]),
                "pred_end": str(pred_timestamps[-1]),
                "zero_hit_count": int(len(hits)),
                "first_zero_hits": [
                    {
                        "horizon_index": int(horizon_idx),
                        "channel_index": int(channel_idx),
                        "timestamp": str(pred_timestamps[int(horizon_idx)]),
                    }
                    for horizon_idx, channel_idx in hits[:10]
                ],
            }
        )
    return samples


def _zero_target_offset_scan(dataset, zero_mask, timestamps, zero_target):
    hit_times = sorted({str(timestamps[w, h]) for w, h, _ in np.argwhere(zero_mask > 0)})
    if not hit_times:
        return []
    timestamp_to_index = {str(ts): idx for idx, ts in enumerate(dataset.timestamps)}
    target = zero_target.reshape(-1)
    offsets = {
        "-1_day": -96,
        "-1_hour": -4,
        "-15_min": -1,
        "0": 0,
        "+15_min": 1,
        "+1_hour": 4,
        "+1_day": 96,
    }
    rows = []
    for label, offset in offsets.items():
        values = []
        used = 0
        for timestamp in hit_times:
            idx = timestamp_to_index.get(timestamp)
            if idx is None:
                continue
            shifted = idx + offset
            if shifted < 0 or shifted >= len(dataset.data_y):
                continue
            values.append(dataset.data_y[shifted])
            used += 1
        if not values:
            rows.append({"offset": label, "num_unique_timestamps": 0, "mse_to_zero_target": 0.0, "mean": 0.0})
            continue
        values = np.asarray(values, dtype=np.float32)
        rows.append(
            {
                "offset": label,
                "num_unique_timestamps": int(used),
                "mse_to_zero_target": float(np.mean((values - target.reshape(1, -1)) ** 2)),
                "mean": float(values.mean()),
            }
        )
    return rows


def _first_zero_timestamps(zero_mask, timestamps, columns, limit: int = 20):
    hits = np.argwhere(zero_mask > 0)
    rows = []
    for window_idx, horizon_idx, channel_idx in hits[:limit]:
        rows.append(
            {
                "window_index": int(window_idx),
                "horizon_index": int(horizon_idx),
                "timestamp": str(timestamps[window_idx, horizon_idx]),
                "channel_index": int(channel_idx),
                "channel": columns[int(channel_idx)] if int(channel_idx) < len(columns) else str(channel_idx),
            }
        )
    return rows


def _masked_stats(values, mask, zero_target):
    if mask.sum() <= 1e-8:
        return {"mean": 0.0, "variance": 0.0, "mse_to_zero_target": 0.0}
    selected = values[mask > 0]
    target = np.broadcast_to(zero_target, values.shape)[mask > 0]
    return {
        "mean": float(selected.mean()),
        "variance": float(selected.var()),
        "mse_to_zero_target": float(np.mean((selected - target) ** 2)),
    }


def _masked_mse(pred, true, mask):
    denom = mask.sum()
    if denom <= 1e-8:
        return 0.0
    return float((((pred - true) ** 2) * mask).sum() / denom)


def _interpret(diagnosis: dict) -> list[str]:
    notes = []
    true_mse = diagnosis["true_at_zero_mask"]["mse_to_zero_target"]
    pred_mse = diagnosis["baseline_pred_at_zero_mask"]["mse_to_zero_target"]
    offset_scan = diagnosis.get("zero_target_offset_scan", [])
    if diagnosis["zero_mask_sum"] <= 0:
        notes.append("zero_mask is empty; zero-event prior cannot affect this split.")
    elif true_mse > pred_mse:
        notes.append("true values at zero_mask positions are farther from zero_target than baseline predictions; hard zero prior is not a valid oracle on this split.")
    else:
        notes.append("true values are closer to zero_target than baseline predictions at zero_mask positions.")
    if offset_scan:
        best = min(offset_scan, key=lambda row: row.get("mse_to_zero_target", float("inf")))
        current = next((row for row in offset_scan if row.get("offset") == "0"), None)
        if current is not None and best["offset"] != "0" and best["mse_to_zero_target"] < current["mse_to_zero_target"] * 0.9:
            notes.append(
                f"offset scan suggests a possible timestamp shift: {best['offset']} has lower MSE_to_zero_target than offset 0."
            )
        else:
            notes.append("offset scan does not show a strong fixed timestamp shift; inspect rule condition, anchor, and zero_target validity first.")
    if diagnosis["event_mask_sum"] != diagnosis["zero_mask_sum"]:
        notes.append("event_mask and zero_mask differ; inspect whether event metrics and zero prior target the same timestamps.")
    return notes


def main():
    parser = argparse.ArgumentParser(description="Diagnose rule-prior masks and zero targets.")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--data", default="ETTm1")
    parser.add_argument("--rule_path", default="./llm_rules/example_rules/ETTm1_rules.json")
    parser.add_argument("--output_path", default="artifacts/core_results/rule_prior_diagnosis.json")
    args = parser.parse_args()
    diagnose_rule_prior(**vars(args))


if __name__ == "__main__":
    main()
