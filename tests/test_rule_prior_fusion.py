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


def test_rule_prior_cli_args_are_parsed():
    from main import normalize_args

    args = SimpleNamespace(
        use_llm_features=0,
        use_llm_rule_features=0,
        use_rule_prior_fusion=1,
        rule_prior_alpha=0.25,
        rule_prior_use_confidence=0,
        rule_prior_types="zero_event",
    )

    normalize_args(args)

    assert args.use_rule_prior_fusion == 1
    assert args.rule_prior_alpha == 0.25
    assert args.rule_prior_types == "zero_event"


def test_rule_prior_scripts_and_analysis_exist():
    script = open("scripts/run_ettm1_rule_prior_core.sh", encoding="utf-8").read()
    assert "--use_rule_prior_fusion 1" in script
    assert "--rule_prior_alpha 0.25" in script
    assert "--rule_prior_alpha 1.0" in script

    summary = open("analysis/summarize_rule_prior_results.py", encoding="utf-8").read()
    assert "non_event_mse" in summary

    diagnosis = open("analysis/diagnose_rule_prior.py", encoding="utf-8").read()
    assert "theoretical_event_mse" in diagnosis
