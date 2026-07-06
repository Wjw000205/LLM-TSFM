"""Summarize ETTm1 peak-transfer seed stability runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluate_rule_gated_ensemble import load_test_masks


CSV_FIELDS = [
    "pred_len",
    "seed",
    "baseline_overall_mse",
    "gated_overall_mse",
    "baseline_event_mse",
    "gated_event_mse",
    "event_reduction_pct",
    "non_event_delta",
    "selected_reason",
    "selected_epoch",
    "status",
    "baseline_dir",
    "gated_dir",
]


def summarize(pred_lens: list[int], seeds: list[int], root: Path = Path("results")) -> list[dict]:
    rows = []
    for pred_len in pred_lens:
        for seed in seeds:
            baseline_dir = root / (
                f"long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl{pred_len}_"
                f"ettm1_peak_transfer_seed{seed}_p{pred_len}_baseline_0"
            )
            expert_dir = root / (
                f"long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl{pred_len}_"
                f"ettm1_peak_transfer_seed{seed}_p{pred_len}_finetune_loss_0"
            )
            gated_dir = root / (
                f"long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl{pred_len}_"
                f"ettm1_peak_transfer_seed{seed}_p{pred_len}_gated_alpha_1p0_0"
            )
            if not (baseline_dir / "metrics_normalized.json").exists():
                rows.append(_missing_row(pred_len, seed, baseline_dir, gated_dir, "missing_baseline"))
                continue
            if not (expert_dir / "config.json").exists() or not (gated_dir / "metrics_normalized.json").exists():
                rows.append(_missing_row(pred_len, seed, baseline_dir, gated_dir, "missing_gated"))
                continue
            rows.append(_summarize_one(pred_len, seed, baseline_dir, expert_dir, gated_dir))
    return rows


def _summarize_one(pred_len: int, seed: int, baseline_dir: Path, expert_dir: Path, gated_dir: Path) -> dict:
    baseline_metrics = _read_json(baseline_dir / "metrics_normalized.json")
    gated_metrics = _read_json(gated_dir / "metrics_normalized.json")
    expert_config = _read_json(expert_dir / "config.json")
    true = np.load(baseline_dir / "true_normalized.npy", mmap_mode="r")
    baseline_pred = np.load(baseline_dir / "pred_normalized.npy", mmap_mode="r")
    gated_pred = np.load(gated_dir / "pred_normalized.npy", mmap_mode="r")
    masks = load_test_masks(baseline_dir / "config.json", expected_windows=true.shape[0])
    non_event_mask = ~(masks[:, :, 0, :].astype(bool))
    baseline_non_event = _masked_mse(baseline_pred, true, non_event_mask)
    gated_non_event = _masked_mse(gated_pred, true, non_event_mask)
    baseline_event = float(baseline_metrics["event_window_mse"])
    gated_event = float(gated_metrics["event_window_mse"])
    selected_reason = expert_config.get("selected_reason")
    return {
        "pred_len": pred_len,
        "seed": seed,
        "baseline_overall_mse": float(baseline_metrics["mse"]),
        "gated_overall_mse": float(gated_metrics["mse"]),
        "baseline_event_mse": baseline_event,
        "gated_event_mse": gated_event,
        "event_reduction_pct": _pct_reduction(baseline_event, gated_event),
        "non_event_delta": gated_non_event - baseline_non_event,
        "selected_reason": selected_reason,
        "selected_epoch": expert_config.get("selected_epoch"),
        "status": "guarded" if selected_reason == "guarded_event_mse" else str(selected_reason or "unknown"),
        "baseline_dir": str(baseline_dir),
        "gated_dir": str(gated_dir),
    }


def _missing_row(pred_len: int, seed: int, baseline_dir: Path, gated_dir: Path, status: str) -> dict:
    return {
        "pred_len": pred_len,
        "seed": seed,
        "baseline_overall_mse": "",
        "gated_overall_mse": "",
        "baseline_event_mse": "",
        "gated_event_mse": "",
        "event_reduction_pct": "",
        "non_event_delta": "",
        "selected_reason": "",
        "selected_epoch": "",
        "status": status,
        "baseline_dir": str(baseline_dir),
        "gated_dir": str(gated_dir),
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _masked_mse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(np.square(pred - true)[mask].mean())


def _pct_reduction(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


def write_outputs(rows: list[dict], output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ETTm1 peak-transfer seed stability runs.")
    parser.add_argument("--pred_lens", default="96,192,336")
    parser.add_argument("--seeds", default="2021,2022,2023")
    parser.add_argument("--output_csv", default="artifacts/core_results/ettm1_peak_transfer_seed_stability.csv")
    parser.add_argument("--output_json", default="artifacts/core_results/ettm1_peak_transfer_seed_stability.json")
    args = parser.parse_args()
    pred_lens = [int(value) for value in args.pred_lens.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    rows = summarize(pred_lens=pred_lens, seeds=seeds)
    write_outputs(rows, Path(args.output_csv), Path(args.output_json))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
