from types import SimpleNamespace

import torch


def test_rule_prior_fusion_is_identity_when_zero_mask_is_empty():
    from models.layers.RulePriorFusion import RulePriorFusion

    pred = torch.randn(2, 4, 3)
    masks = torch.zeros(2, 4, 3, 3)
    fusion = RulePriorFusion(alpha=0.5)

    fused = fusion(pred_base=pred, future_masks=masks, zero_target=torch.zeros(3))

    assert torch.allclose(fused, pred)


def test_rule_prior_fusion_alpha_zero_is_baseline_and_alpha_one_matches_hard_zero():
    from models.layers.RulePriorFusion import RulePriorFusion

    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    masks = torch.zeros(1, 2, 3, 2)
    masks[:, :, 1, :] = 1.0
    zero_target = torch.tensor([-1.0, 10.0])

    alpha_zero = RulePriorFusion(alpha=0.0)
    alpha_one = RulePriorFusion(alpha=1.0)

    assert torch.allclose(alpha_zero(pred, masks, zero_target), pred)
    expected = zero_target.view(1, 1, 2).expand_as(pred)
    assert torch.allclose(alpha_one(pred, masks, zero_target), expected)


def test_rule_prior_fusion_respects_channel_specific_zero_mask():
    from models.layers.RulePriorFusion import RulePriorFusion

    pred = torch.tensor([[[2.0, 4.0]]])
    masks = torch.zeros(1, 1, 3, 2)
    masks[:, :, 1, 0] = 1.0
    zero_target = torch.tensor([0.0, 100.0])

    fused = RulePriorFusion(alpha=0.25)(pred, masks, zero_target)

    assert torch.allclose(fused, torch.tensor([[[1.5, 4.0]]]))


def test_rule_prior_fusion_uses_zero_column_for_three_dimensional_masks():
    from models.layers.RulePriorFusion import RulePriorFusion

    pred = torch.tensor([[[2.0]]])
    masks = torch.zeros(1, 1, 3)
    masks[:, :, 0] = 1.0
    zero_target = torch.tensor([0.0])

    fused = RulePriorFusion(alpha=1.0)(pred, masks, zero_target)

    assert torch.allclose(fused, pred)

    masks[:, :, 1] = 1.0
    fused = RulePriorFusion(alpha=1.0)(pred, masks, zero_target)

    assert torch.allclose(fused, torch.zeros_like(pred))


def test_calibrated_rule_prior_changes_only_valid_channels(tmp_path):
    from models.layers.RulePriorFusion import RulePriorFusion

    calibrated = {
        "dataset_name": "toy",
        "patterns": [
            {
                "name": "periodic_zero_day",
                "type": "zero_event",
                "enabled": True,
                "valid_channels": ["A", "C"],
                "calibrated_alpha": {"A": 0.5, "C": 1.0},
                "channel_diagnostics": {
                    "A": {
                        "best_prior_type": "residual_mean",
                        "best_alpha": 0.5,
                        "prior_value": 2.0,
                    },
                    "C": {
                        "best_prior_type": "conditional_mean",
                        "best_alpha": 1.0,
                        "prior_value": -1.0,
                    },
                },
                "disabled_channels": {"B": "no prior improves baseline"},
            }
        ],
    }
    path = tmp_path / "calibrated.json"
    path.write_text(__import__("json").dumps(calibrated), encoding="utf-8")
    pred = torch.tensor([[[10.0, 20.0, 30.0]]])
    masks = torch.zeros(1, 1, 3, 3)
    masks[:, :, 1, :] = 1.0

    fusion = RulePriorFusion(
        mode="calibrated",
        validated_rule_path=str(path),
        channel_names=["A", "B", "C"],
    )
    fused = fusion(pred_base=pred, future_masks=masks, zero_target=torch.zeros(3))

    assert torch.allclose(fused, torch.tensor([[[11.0, 20.0, -1.0]]]))


def test_calibrated_rule_prior_identity_when_all_channels_disabled(tmp_path):
    from models.layers.RulePriorFusion import RulePriorFusion

    calibrated = {
        "dataset_name": "toy",
        "patterns": [
            {
                "name": "periodic_zero_day",
                "type": "zero_event",
                "enabled": False,
                "valid_channels": [],
                "channel_diagnostics": {},
                "disabled_channels": {"A": "no prior improves baseline"},
            }
        ],
    }
    path = tmp_path / "calibrated.json"
    path.write_text(__import__("json").dumps(calibrated), encoding="utf-8")
    pred = torch.tensor([[[10.0, 20.0]]])
    masks = torch.ones(1, 1, 3, 2)

    fusion = RulePriorFusion(
        mode="calibrated",
        validated_rule_path=str(path),
        channel_names=["A", "B"],
    )
    fused = fusion(pred_base=pred, future_masks=masks, zero_target=torch.zeros(2))

    assert torch.allclose(fused, pred)


def test_rule_prior_cli_args_are_parsed():
    from main import normalize_args

    args = SimpleNamespace(
        use_llm_features=0,
        use_llm_rule_features=0,
        use_rule_prior_fusion=1,
        rule_prior_alpha=0.25,
        rule_prior_use_confidence=0,
        rule_prior_types="zero_event",
        rule_prior_mode="calibrated",
        validated_rule_path="rules.json",
        disable_invalid_rules=1,
    )

    normalize_args(args)

    assert args.use_rule_prior_fusion == 1
    assert args.rule_prior_alpha == 0.25
    assert args.rule_prior_types == "zero_event"
    assert args.rule_prior_mode == "calibrated"
    assert args.validated_rule_path == "rules.json"
    assert args.disable_invalid_rules == 1


def test_rule_prior_scripts_and_analysis_exist():
    script = open("scripts/run_ettm1_rule_prior_core.sh", encoding="utf-8").read()
    assert "--use_rule_prior_fusion 1" in script
    assert "--rule_prior_alpha 0.25" in script
    assert "--rule_prior_alpha 1.0" in script

    summary = open("analysis/summarize_rule_prior_results.py", encoding="utf-8").read()
    assert "non_event_mse" in summary

    diagnosis = open("analysis/diagnose_rule_prior.py", encoding="utf-8").read()
    assert "theoretical_event_mse" in diagnosis


def test_calibrator_selects_residual_mean_and_disables_bad_channel():
    from analysis.verify_and_calibrate_rules import calibrate_channels

    pred = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]]).numpy()
    true = torch.tensor([[[2.0, 10.0], [3.0, 20.0], [4.0, 30.0]]]).numpy()
    mask = torch.ones(1, 3, 2).numpy()
    zero_target = torch.zeros(2).numpy()

    rows = calibrate_channels(
        pred,
        true,
        mask,
        channel_names=["good", "bad"],
        zero_target=zero_target,
        alpha_grid=[0.0, 1.0],
        min_prior_improvement=0.01,
    )

    assert rows["good"]["enabled"] is True
    assert rows["good"]["best_prior_type"] == "residual_mean"
    assert rows["good"]["best_alpha"] == 1.0
    assert rows["good"]["prior_value"] == 1.0
    assert rows["bad"]["enabled"] is False
    assert rows["bad"]["disable_reason"] == "no prior improves baseline on calibration split"


def test_calibrator_disables_false_positive_rule_mask_even_when_prior_improves():
    from analysis.verify_and_calibrate_rules import calibrate_channels

    pred = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]).numpy()
    true = torch.tensor([[[2.0], [3.0], [4.0], [5.0]]]).numpy()
    mask = torch.ones(1, 4, 1).numpy()
    actual_event_mask = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]]).numpy()

    rows = calibrate_channels(
        pred,
        true,
        mask,
        channel_names=["A"],
        zero_target=torch.zeros(1).numpy(),
        alpha_grid=[0.0, 1.0],
        min_prior_improvement=0.01,
        actual_event_mask=actual_event_mask,
        min_event_precision=0.5,
    )

    assert rows["A"]["enabled"] is False
    assert rows["A"]["event_precision"] == 0.25
    assert rows["A"]["false_positive_ratio"] == 0.75
    assert rows["A"]["disable_reason"] == "rule mask precision below min_event_precision on calibration split"


def test_calibrator_enables_rule_when_precision_and_prior_improve():
    from analysis.verify_and_calibrate_rules import calibrate_channels

    pred = torch.tensor([[[1.0], [2.0], [10.0], [20.0]]]).numpy()
    true = torch.tensor([[[2.0], [3.0], [10.0], [20.0]]]).numpy()
    mask = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]]).numpy()
    actual_event_mask = mask.copy()

    rows = calibrate_channels(
        pred,
        true,
        mask,
        channel_names=["A"],
        zero_target=torch.zeros(1).numpy(),
        alpha_grid=[0.0, 1.0],
        min_prior_improvement=0.01,
        actual_event_mask=actual_event_mask,
        min_event_precision=0.5,
    )

    assert rows["A"]["enabled"] is True
    assert rows["A"]["event_precision"] == 1.0
    assert rows["A"]["false_positive_ratio"] == 0.0
    assert rows["A"]["best_prior_type"] == "residual_mean"


def test_calibrator_rule_window_objective_ignores_near_zero_precision_gate():
    from analysis.verify_and_calibrate_rules import calibrate_channels

    pred = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]).numpy()
    true = torch.tensor([[[2.0], [3.0], [4.0], [5.0]]]).numpy()
    mask = torch.ones(1, 4, 1).numpy()
    actual_event_mask = torch.tensor([[[1.0], [0.0], [0.0], [0.0]]]).numpy()

    rows = calibrate_channels(
        pred,
        true,
        mask,
        channel_names=["A"],
        zero_target=torch.zeros(1).numpy(),
        alpha_grid=[0.0, 1.0],
        min_prior_improvement=0.01,
        actual_event_mask=actual_event_mask,
        min_event_precision=0.5,
        calibration_objective="rule_window_mse",
        allowed_prior_types=["residual_mean", "conditional_mean"],
    )

    assert rows["A"]["enabled"] is True
    assert rows["A"]["event_precision"] == 0.25
    assert rows["A"]["precision_gate_applied"] is False
    assert rows["A"]["best_prior_type"] == "residual_mean"


def test_calibrator_can_exclude_zero_target_prior():
    from analysis.verify_and_calibrate_rules import calibrate_channels

    pred = torch.tensor([[[10.0], [10.0]]]).numpy()
    true = torch.tensor([[[0.0], [0.0]]]).numpy()
    mask = torch.ones(1, 2, 1).numpy()

    rows = calibrate_channels(
        pred,
        true,
        mask,
        channel_names=["A"],
        zero_target=torch.zeros(1).numpy(),
        alpha_grid=[0.0, 1.0],
        min_prior_improvement=0.01,
        calibration_objective="rule_window_mse",
        allowed_prior_types=["residual_mean"],
    )

    assert rows["A"]["enabled"] is True
    assert rows["A"]["candidate_best_prior_type"] == "residual_mean"
    assert rows["A"]["best_prior_type"] == "residual_mean"
