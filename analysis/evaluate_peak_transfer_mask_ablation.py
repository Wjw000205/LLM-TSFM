"""Evaluate whether peak-transfer gains depend on the correct event mask."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluate_rule_gated_ensemble import extract_event_mask, load_test_masks, rule_gated_prediction


DEFAULT_RUNS = [
    {
        "pred_len": 96,
        "baseline_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_myfram_gpt55_peak_only_baseline_0",
        "expert_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl96_ettm1_myfram_gpt55_peak_only_finetune_generated_loss_guarded_0",
    },
    {
        "pred_len": 192,
        "baseline_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl192_ettm1_gpt55_peak_transfer_p192_baseline_0",
        "expert_dir": "results/long_term_forecast_DLinear_ETTm1_ftM_sl336_ll48_pl192_ettm1_gpt55_peak_transfer_p192_finetune_loss_0",
    },
]

CSV_FIELDS = [
    "Experiment",
    "Pred_len",
    "Overall MSE",
    "Event MSE",
    "Non-event MSE",
    "Event Reduction",
    "Non-event Delta",
    "Mask Type",
    "Status",
    "Notes",
]


def evaluate_ablation(
    runs: list[dict[str, Any]] | None = None,
    shift_steps: int = 96,
    random_seed: int = 2024,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs or DEFAULT_RUNS:
        rows.extend(_evaluate_one(run, shift_steps=shift_steps, random_seed=random_seed))
    return rows


def _evaluate_one(run: dict[str, Any], shift_steps: int, random_seed: int) -> list[dict[str, Any]]:
    pred_len = int(run["pred_len"])
    baseline_dir = Path(run["baseline_dir"])
    expert_dir = Path(run["expert_dir"])
    true = np.load(baseline_dir / "true_normalized.npy")
    baseline = np.load(baseline_dir / "pred_normalized.npy")
    expert = np.load(expert_dir / "pred_normalized.npy")
    masks = load_test_masks(baseline_dir / "config.json", expected_windows=true.shape[0])
    correct_mask = extract_event_mask(masks, baseline.shape)
    eval_event_mask = correct_mask.astype(bool)
    eval_non_event_mask = ~eval_event_mask

    variants = {
        "baseline": baseline,
        "correct_gated_mask": rule_gated_prediction(baseline, expert, correct_mask),
        "shuffled_event_mask": rule_gated_prediction(baseline, expert, _shuffle_mask(correct_mask, random_seed + pred_len)),
        "shifted_wrong_mask_24h": rule_gated_prediction(baseline, expert, _shift_mask(correct_mask, shift_steps)),
        "random_sparse_mask_same_ratio": rule_gated_prediction(
            baseline,
            expert,
            _random_sparse_mask(correct_mask, random_seed + 10 * pred_len),
        ),
        "no_gate_global_transfer": expert,
    }
    mask_types = {
        "baseline": "none",
        "correct_gated_mask": "correct",
        "shuffled_event_mask": "shuffled",
        "shifted_wrong_mask_24h": f"shifted_by_{shift_steps}_steps",
        "random_sparse_mask_same_ratio": "random_same_ratio",
        "no_gate_global_transfer": "global_no_gate",
    }
    baseline_event = _masked_mse(baseline, true, eval_event_mask)
    baseline_non_event = _masked_mse(baseline, true, eval_non_event_mask)

    rows = []
    for name, pred in variants.items():
        event_mse = _masked_mse(pred, true, eval_event_mask)
        non_event_mse = _masked_mse(pred, true, eval_non_event_mask)
        rows.append(
            {
                "Experiment": name,
                "Pred_len": pred_len,
                "Overall MSE": float(np.mean(np.square(pred - true))),
                "Event MSE": event_mse,
                "Non-event MSE": non_event_mse,
                "Event Reduction": _pct_reduction(baseline_event, event_mse),
                "Non-event Delta": non_event_mse - baseline_non_event,
                "Mask Type": mask_types[name],
                "Status": "reference" if name == "baseline" else "evaluated",
                "Notes": _notes(name),
            }
        )
    return rows


def _shuffle_mask(mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(mask.shape[0])
    return mask[order]


def _shift_mask(mask: np.ndarray, shift_steps: int) -> np.ndarray:
    return np.roll(mask, shift=int(shift_steps), axis=0)


def _random_sparse_mask(mask: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = np.zeros(mask.size, dtype=np.float32)
    count = int(np.asarray(mask).sum())
    if count > 0:
        flat[rng.choice(mask.size, size=count, replace=False)] = 1.0
    return flat.reshape(mask.shape)


def _masked_mse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    return float(np.square(pred - true)[mask].mean())


def _pct_reduction(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


def _notes(name: str) -> str:
    return {
        "baseline": "Pure DLinear baseline.",
        "correct_gated_mask": "Uses loss expert only under the original event mask.",
        "shuffled_event_mask": "Uses the same mask density but assigns event windows to other samples.",
        "shifted_wrong_mask_24h": "Uses the mask from a 24h-shifted sample index for ETTm1.",
        "random_sparse_mask_same_ratio": "Uses random sparse positions with the same event count.",
        "no_gate_global_transfer": "Uses the loss expert everywhere without event gating.",
    }[name]


def write_outputs(rows: list[dict[str, Any]], output_csv: Path, output_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ETTm1 peak-transfer mask ablations.")
    parser.add_argument("--output_csv", default="artifacts/core_results/ettm1_peak_transfer_mask_ablation_summary.csv")
    parser.add_argument("--output_json", default="artifacts/core_results/ettm1_peak_transfer_mask_ablation_summary.json")
    parser.add_argument("--shift_steps", type=int, default=96)
    parser.add_argument("--random_seed", type=int, default=2024)
    args = parser.parse_args()
    rows = evaluate_ablation(shift_steps=args.shift_steps, random_seed=args.random_seed)
    write_outputs(rows, Path(args.output_csv), Path(args.output_json))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
