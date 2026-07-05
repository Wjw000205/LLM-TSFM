import csv
import json
from types import SimpleNamespace

import torch


def test_nonevent_preservation_loss_uses_only_non_event_region():
    from losses.dataset_aware_loss import DatasetAwareLoss

    pred = torch.tensor([[[1.0], [2.0], [3.0]]])
    true = torch.zeros_like(pred)
    masks = torch.zeros(1, 3, 3, 1)
    masks[:, 1, 0, :] = 1.0
    criterion = DatasetAwareLoss(
        {
            "use_nonevent_preservation_loss": True,
            "nonevent_weight": 2.0,
        }
    )

    loss_dict = criterion(pred, true, batch_masks=masks)

    assert torch.isclose(loss_dict["nonevent_loss"], torch.tensor(5.0))
    assert torch.isclose(loss_dict["loss"], loss_dict["base_loss"] + 2.0 * loss_dict["nonevent_loss"])


def test_baseline_distillation_loss_uses_only_non_event_region():
    from losses.dataset_aware_loss import DatasetAwareLoss

    pred = torch.tensor([[[1.0], [5.0], [3.0]]])
    baseline_pred = torch.tensor([[[0.0], [0.0], [1.0]]])
    true = torch.zeros_like(pred)
    masks = torch.zeros(1, 3, 3, 1)
    masks[:, 1, 0, :] = 1.0
    criterion = DatasetAwareLoss(
        {
            "use_baseline_distillation": True,
            "distill_weight": 0.5,
        }
    )

    loss_dict = criterion(pred, true, batch_masks=masks, baseline_pred=baseline_pred)

    assert torch.isclose(loss_dict["distill_loss"], torch.tensor(2.5))
    assert torch.isclose(loss_dict["loss"], loss_dict["base_loss"] + 0.5 * loss_dict["distill_loss"])


def test_guarded_checkpoint_selector_prefers_event_mse_within_mse_guardrail(tmp_path):
    from exp.exp_long_term_forecasting import GuardedCheckpointSelector

    model = torch.nn.ModuleDict({"model": torch.nn.Linear(1, 1)})
    selector = GuardedCheckpointSelector(
        selection_metric="guarded_event_mse",
        baseline_mse=1.0,
        overall_mse_tolerance=0.05,
    )

    selector.update(1, {"base_loss": 1.2, "event_mse": 0.1, "zero_mse": 0.1, "rule_score": 0.0, "loss": 1.2}, model, tmp_path)
    selector.update(2, {"base_loss": 1.04, "event_mse": 0.4, "zero_mse": 0.4, "rule_score": 0.0, "loss": 1.04}, model, tmp_path)
    selector.update(3, {"base_loss": 1.03, "event_mse": 0.2, "zero_mse": 0.2, "rule_score": 0.0, "loss": 1.03}, model, tmp_path)

    selected = selector.finalize()

    assert selected["selected_epoch"] == 3
    assert selected["selected_reason"] == "guarded_event_mse"
    assert (tmp_path / "checkpoint.pth").exists()


def test_guarded_checkpoint_selector_falls_back_when_guardrail_has_no_candidate(tmp_path):
    from exp.exp_long_term_forecasting import GuardedCheckpointSelector

    model = torch.nn.ModuleDict({"model": torch.nn.Linear(1, 1)})
    selector = GuardedCheckpointSelector(
        selection_metric="guarded_event_mse",
        baseline_mse=1.0,
        overall_mse_tolerance=0.01,
    )

    selector.update(1, {"base_loss": 1.2, "event_mse": 0.1, "zero_mse": 0.1, "rule_score": 0.0, "loss": 1.2}, model, tmp_path)
    selector.update(2, {"base_loss": 1.1, "event_mse": 0.2, "zero_mse": 0.2, "rule_score": 0.0, "loss": 1.1}, model, tmp_path)

    selected = selector.finalize()

    assert selected["selected_epoch"] == 2
    assert selected["selected_reason"] == "fallback_base_mse_no_guardrail_candidate"


def test_guardrail_cli_args_are_parsed_and_normalized():
    from main import normalize_args

    args = SimpleNamespace(
        use_llm_features=0,
        use_llm_rule_features=0,
        load_pretrained_checkpoint="baseline.pth",
        finetune_learning_rate=1e-5,
        finetune_epochs=4,
        finetune_patience=2,
        learning_rate=1e-4,
        train_epochs=10,
        patience=3,
    )

    normalize_args(args)

    assert args.learning_rate == 1e-5
    assert args.train_epochs == 4
    assert args.patience == 2


def test_guardrail_scripts_and_pareto_selector_exist(tmp_path):
    guarded = open("scripts/run_ettm1_guarded_longtail.sh", encoding="utf-8").read()
    sweep = open("scripts/run_ettm1_longtail_weight_sweep.sh", encoding="utf-8").read()
    assert "--selection_metric guarded_event_mse" in guarded
    assert "--use_nonevent_preservation_loss 1" in guarded
    assert "--event_weight 1.0 --zero_weight 0.1 --nonevent_weight 1.0" in sweep

    sweep_csv = tmp_path / "sweep.csv"
    with sweep_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "overall_mse",
                "overall_mae",
                "event_mse",
                "zero_mse",
                "rule_score",
            ],
        )
        writer.writeheader()
        writer.writerow({"experiment": "bad", "overall_mse": 1.2, "overall_mae": 0.0, "event_mse": 0.2, "zero_mse": 0.2, "rule_score": 0.0})
        writer.writerow({"experiment": "good", "overall_mse": 1.03, "overall_mae": 0.0, "event_mse": 0.5, "zero_mse": 0.5, "rule_score": 0.0})
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"mse": 1.0, "event_window_mse": 0.8}), encoding="utf-8")

    from analysis.select_pareto_longtail import select_pareto

    result = select_pareto(
        baseline_metrics=str(baseline_path),
        sweep_csv=str(sweep_csv),
        output_csv=str(tmp_path / "pareto.csv"),
        output_markdown=str(tmp_path / "pareto.md"),
        overall_tolerance=0.05,
    )

    assert result["best"]["experiment"] == "good"
    assert result["best"]["accepted_by_guardrail"] is True
