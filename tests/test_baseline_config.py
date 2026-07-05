from types import SimpleNamespace

import json
import numpy as np
import pandas as pd
import torch


def test_dlinear_init_avg_flag_controls_linear_initialization():
    from models.DLinear import DLinear

    common = dict(
        seq_len=24,
        pred_len=12,
        enc_in=3,
        c_out=3,
        individual=False,
        moving_avg=3,
        use_revin=False,
    )

    avg_model = DLinear(SimpleNamespace(**common, dlinear_init_avg=1))
    default_model = DLinear(SimpleNamespace(**common, dlinear_init_avg=0))

    expected = torch.full_like(avg_model.linear_seasonal.weight, 1.0 / 24)
    assert torch.allclose(avg_model.linear_seasonal.weight, expected)
    assert not torch.allclose(default_model.linear_seasonal.weight, expected)


def test_early_stop_metric_can_use_total_or_base_mse():
    from exp.exp_long_term_forecasting import select_early_stop_value

    losses = {"loss": 10.0, "base_loss": 2.0}

    assert select_early_stop_value(losses, "total_loss") == 10.0
    assert select_early_stop_value(losses, "base_mse") == 2.0


def test_run_config_snapshot_is_json_serializable(tmp_path):
    from exp.exp_long_term_forecasting import save_run_config

    args = SimpleNamespace(
        data="ETTh1",
        data_path="ETTh1.csv",
        seq_len=336,
        scaler=object(),
        raw_feature_dim=7,
        llm_feature_dim=0,
        target_dim=7,
        zero_target=[0.0] * 7,
    )

    save_run_config(args, tmp_path)

    assert (tmp_path / "config.json").exists()
    text = (tmp_path / "setting.txt").read_text(encoding="utf-8")
    assert "data: ETTh1" in text
    assert "raw_feature_dim: 7" in text
    assert "scaler" not in text


def test_data_provider_records_split_feature_dims_and_updates_model_dims(tmp_path):
    from data_provider.data_factory import data_provider

    rows = 96
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "HUFL": np.arange(rows, dtype=np.float32),
            "OT": np.arange(rows, dtype=np.float32) * 2.0,
        }
    ).to_csv(tmp_path / "toy.csv", index=False)
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(
        json.dumps(
            {
                "dataset_name": "Toy",
                "patterns": [
                    {
                        "name": "noon_peak",
                        "type": "peak_event",
                        "condition": {"kind": "hourly", "hour": 12},
                        "affected_variables": "all",
                        "time_range": "single_step",
                        "features": {
                            "event_mask": True,
                            "peak_mask": True,
                            "hour_distance_to_peak": True,
                            "rule_confidence": 0.8,
                            "support_count": 4,
                        },
                    },
                    {
                        "name": "midnight_zero",
                        "type": "zero_event",
                        "condition": {"kind": "hourly", "hour": 0},
                        "affected_variables": ["OT"],
                        "time_range": "single_step",
                        "features": {"zero_event_mask": True},
                    },
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
        use_standard_time_features=1,
        use_llm_rule_features=1,
        use_oracle_features=1,
        llm_rule_path=str(rule_path),
        batch_size=4,
        num_workers=0,
    )

    dataset, _ = data_provider(args, "train")

    assert dataset.raw_feature_dim == 2
    assert dataset.standard_time_feature_dim == 5
    assert dataset.llm_rule_feature_dim > 0
    assert dataset.oracle_feature_dim == 6
    assert dataset.llm_feature_dim == (
        dataset.standard_time_feature_dim + dataset.llm_rule_feature_dim + dataset.oracle_feature_dim
    )
    assert args.raw_feature_dim == dataset.raw_feature_dim
    assert args.raw_input_dim == dataset.raw_feature_dim
    assert args.standard_time_feature_dim == dataset.standard_time_feature_dim
    assert args.llm_rule_feature_dim == dataset.llm_rule_feature_dim
    assert args.oracle_feature_dim == dataset.oracle_feature_dim
    assert args.enc_in == args.raw_feature_dim + args.llm_feature_dim
    assert args.c_out == dataset.target_dim == 2


def test_prediction_output_payload_uses_original_scale_for_pred_aliases():
    from exp.exp_long_term_forecasting import build_prediction_output_payload

    pred_normalized = np.array([[[1.0]]], dtype=np.float32)
    true_normalized = np.array([[[2.0]]], dtype=np.float32)
    pred_original = np.array([[[10.0]]], dtype=np.float32)
    true_original = np.array([[[20.0]]], dtype=np.float32)

    payload = build_prediction_output_payload(pred_normalized, true_normalized, pred_original, true_original)

    assert np.array_equal(payload["pred.npy"], pred_original)
    assert np.array_equal(payload["true.npy"], true_original)
    assert np.array_equal(payload["pred_normalized.npy"], pred_normalized)
    assert np.array_equal(payload["true_normalized.npy"], true_normalized)
    assert np.array_equal(payload["pred_original.npy"], pred_original)
    assert np.array_equal(payload["true_original.npy"], true_original)


def test_legacy_llm_features_alias_normalizes_to_rule_features():
    from main import normalize_args

    args = SimpleNamespace(use_llm_features=1, use_llm_rule_features=0)

    normalize_args(args)

    assert args.use_llm_rule_features == 1


def test_official_baseline_scripts_exist_and_disable_non_baseline_modules():
    required_off_flags = [
        "--use_revin 0",
        "--use_llm_features 0",
        "--use_llm_rule_features 0",
        "--use_dataset_aware_loss 0",
        "--use_event_weighted_loss 0",
        "--use_zero_consistency_loss 0",
        "--use_peak_shape_loss 0",
        "--use_diff_loss 0",
        "--use_freq_loss 0",
        "--use_rule_adapter 0",
        "--use_hard_intervention 0",
        "--early_stop_metric base_mse",
        "--dlinear_init_avg 0",
    ]
    scripts = {
        "scripts/run_ettm1_dlinear_baseline.sh": ["--data ETTm1", "--seq_len 336", "--batch_size 8", "--learning_rate 0.0001"],
        "scripts/run_etth1_dlinear_baseline.sh": ["--data ETTh1", "--seq_len 336", "--batch_size 32", "--learning_rate 0.005"],
    }
    for path, expected_flags in scripts.items():
        text = open(path, encoding="utf-8").read()
        for flag in required_off_flags + expected_flags:
            assert flag in text


def test_dlinear_ablation_scripts_cover_required_combinations():
    scripts = [
        "scripts/run_ettm1_dlinear_ablation.sh",
        "scripts/run_etth1_dlinear_ablation.sh",
    ]
    expected = [
        "--des pure_dlinear_336",
        "--des pure_dlinear_96",
        "--des revin",
        "--des standard_time_features",
        "--des llm_rule_features",
        "--des dataset_aware_loss",
        "--des llm_rule_features_loss",
        "--des revin_llm_rule_features_loss",
        "--des rule_adapter",
        "--des hard_intervention",
        "--use_standard_time_features 1",
        "--use_llm_rule_features 1",
        "--use_dataset_aware_loss 1",
        "--use_rule_adapter 1",
        "--use_hard_intervention 1",
    ]

    for path in scripts:
        text = open(path, encoding="utf-8").read()
        for token in expected:
            assert token in text
