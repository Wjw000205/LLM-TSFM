from types import SimpleNamespace

import json
import numpy as np
import pandas as pd
import torch


def test_rule_gated_intervention_is_identity_when_event_mask_is_zero():
    from models.layers.RuleGatedIntervention import RuleGatedIntervention

    layer = RuleGatedIntervention(hidden_dim=3, feature_dim=2, intervention_scale=1.0, init_zero=False)
    h = torch.randn(2, 4, 3)
    future_features = torch.randn(2, 4, 2)
    masks = torch.zeros(2, 4, 3, 3)

    out = layer(h, future_features=future_features, future_masks=masks)

    assert torch.allclose(out, h)
    stats = layer.get_intervention_stats()
    assert stats["mean_event_gate"] == 0.0
    assert stats["mean_event_delta_norm"] == 0.0
    assert stats["mean_non_event_gate"] >= 0.0


def test_rule_gated_intervention_init_zero_starts_as_identity_on_events():
    from models.layers.RuleGatedIntervention import RuleGatedIntervention

    layer = RuleGatedIntervention(hidden_dim=3, feature_dim=2, intervention_scale=1.0, init_zero=True)
    h = torch.randn(2, 4, 3)
    future_features = torch.randn(2, 4, 2)
    masks = torch.zeros(2, 4, 3, 3)
    masks[:, :, 0, :] = 1.0

    out = layer(h, future_features=future_features, future_masks=masks)

    assert torch.allclose(out, h)
    assert torch.isclose(layer.get_intervention_reg_loss(), torch.tensor(0.0))


def test_dlinear_intervention_disabled_matches_plain_forward():
    from models.DLinear import DLinear

    args = SimpleNamespace(
        seq_len=12,
        pred_len=4,
        enc_in=3,
        c_out=3,
        individual=0,
        moving_avg=3,
        use_revin=0,
        dlinear_init_avg=0,
        use_intervention_layer=0,
    )
    model = DLinear(args)
    x = torch.randn(2, 12, 3)
    future_features = torch.randn(2, 4, 5)
    future_masks = torch.ones(2, 4, 3, 3)

    base = model(x)
    with_optional_inputs = model(x, future_features=future_features, future_masks=future_masks)

    assert torch.allclose(base, with_optional_inputs)


def test_dlinear_intervention_init_zero_matches_same_weights_without_intervention():
    from models.DLinear import DLinear

    common = dict(
        seq_len=12,
        pred_len=4,
        enc_in=3,
        c_out=3,
        individual=0,
        moving_avg=3,
        use_revin=0,
        dlinear_init_avg=0,
    )
    torch.manual_seed(7)
    baseline = DLinear(SimpleNamespace(**common, use_intervention_layer=0))
    torch.manual_seed(7)
    intervention = DLinear(
        SimpleNamespace(
            **common,
            use_intervention_layer=1,
            llm_feature_dim=5,
            intervention_hidden=8,
            intervention_dropout=0.0,
            intervention_scale=1.0,
            intervention_init_zero=1,
        )
    )
    intervention.load_state_dict(baseline.state_dict(), strict=False)
    x = torch.randn(2, 12, 3)
    future_features = torch.randn(2, 4, 5)
    future_masks = torch.ones(2, 4, 3, 3)

    assert torch.allclose(baseline(x), intervention(x, future_features=future_features, future_masks=future_masks))


def test_intervention_cli_and_scripts_are_available():
    from main import get_args

    args = get_args_from_tokens(
        [
            "--use_intervention_layer",
            "1",
            "--intervention_hidden",
            "16",
            "--intervention_dropout",
            "0.1",
            "--intervention_scale",
            "0.5",
            "--intervention_init_zero",
            "1",
            "--use_intervention_reg",
            "1",
            "--intervention_reg_weight",
            "0.01",
        ],
        get_args,
    )

    assert args.use_intervention_layer == 1
    assert args.intervention_hidden == 16
    assert args.intervention_dropout == 0.1
    assert args.intervention_scale == 0.5
    assert args.intervention_init_zero == 1
    assert args.use_intervention_reg == 1
    assert args.intervention_reg_weight == 0.01

    script = open("scripts/run_ettm1_intervention_core.sh", encoding="utf-8").read()
    assert "--use_intervention_layer 1" in script
    assert "--use_rule_adapter 1" in script
    analysis = open("analysis/summarize_intervention_results.py", encoding="utf-8").read()
    assert "Mean Event Gate" in analysis


def get_args_from_tokens(tokens, get_args):
    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["main.py", *tokens]
        return get_args()
    finally:
        sys.argv = original_argv


def test_intervention_features_do_not_change_dlinear_backbone_input_dim(tmp_path):
    from data_provider.data_factory import data_provider

    rows = 96
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "HUFL": np.arange(rows, dtype=np.float32),
            "OT": np.arange(rows, dtype=np.float32) * 0.5,
        }
    ).to_csv(tmp_path / "toy.csv", index=False)
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(
        json.dumps(
            {
                "dataset_name": "Toy",
                "patterns": [
                    {
                        "name": "noon_zero",
                        "type": "zero_event",
                        "condition": {"kind": "hourly", "hour": 12},
                        "affected_variables": "all",
                        "time_range": "single_step",
                        "features": {"event_mask": True, "zero_event_mask": True, "rule_confidence": 1.0},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        root_path=str(tmp_path),
        data_path="toy.csv",
        seq_len=12,
        label_len=6,
        pred_len=6,
        features="M",
        target="OT",
        data="Toy",
        use_zscore=1,
        timeenc=0,
        freq="h",
        use_llm_features=0,
        use_standard_time_features=0,
        use_llm_rule_features=1,
        use_oracle_features=0,
        use_intervention_layer=1,
        llm_rule_path=str(rule_path),
        batch_size=4,
        num_workers=0,
    )

    dataset, _ = data_provider(args, "train")

    assert dataset.llm_feature_dim > 0
    assert args.enc_in == dataset.raw_feature_dim
