from types import SimpleNamespace

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
