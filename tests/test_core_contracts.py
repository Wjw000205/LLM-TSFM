import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


def test_standard_scaler_round_trip_and_zero_std():
    from utils.scaler import StandardScaler

    data = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(data)

    transformed = scaler.transform(data)
    restored = scaler.inverse_transform(transformed)

    assert np.allclose(restored, data)
    assert np.isfinite(transformed).all()
    assert scaler.std[1] > 0.0


def test_revin_norm_denorm_round_trip():
    from models.layers.RevIN import RevIN

    x = torch.randn(4, 16, 3)
    layer = RevIN(num_features=3, affine=True)

    normalized = layer(x, mode="norm")
    restored = layer(normalized, mode="denorm")

    assert normalized.shape == x.shape
    assert torch.allclose(restored, x, atol=1e-5)


def test_rule_masks_and_features_cover_supported_conditions():
    from llm_rules.feature_generator import generate_llm_rule_features, generate_standard_time_features
    from llm_rules.mask_generator import generate_event_mask
    from llm_rules.rule_schema import LLMRules, RulePattern

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=48, freq="h")
    rules = LLMRules(
        dataset_name="unit",
        patterns=[
            RulePattern(
                name="first_day",
                type="zero_event",
                condition={
                    "kind": "calendar_periodic",
                    "anchor": "2024-01-01 00:00:00",
                    "month_interval": 2,
                    "day": 1,
                },
                affected_variables=["OT"],
                time_range="whole_day",
                losses={"event_weighted_mse": {"enabled": True, "weight": 2.0}},
                features={"event_mask": True, "days_to_event": True},
            ),
            RulePattern(
                name="noon_peak",
                type="peak_event",
                condition={"kind": "hourly", "hour": 12},
                affected_variables="all",
                time_range="single_step",
                losses={"peak_shape": {"enabled": True, "weight": 0.5}},
                features={"peak_mask": True, "hour_distance_to_peak": True},
            ),
            RulePattern(
                name="monday",
                type="calendar_event",
                condition={"kind": "weekday", "weekday": 0},
                affected_variables="all",
                time_range="whole_day",
            ),
        ],
    )

    masks = generate_event_mask(timestamps, rules, target_columns=["OT", "HUFL"])
    llm_features, llm_names = generate_llm_rule_features(timestamps, rules, target_columns=["OT", "HUFL"])
    time_features, time_names = generate_standard_time_features(timestamps)

    assert masks["first_day"].sum() == 24
    assert masks["noon_peak"].sum() == 4
    assert masks["monday"].sum() == 48
    assert masks["event_mask"].shape == (48, 2)
    assert masks["first_day"][:, 0].sum() == 24
    assert masks["first_day"][:, 1].sum() == 0
    assert llm_features.shape[0] == 48
    assert time_features.shape[0] == 48
    assert "event_mask_first_day_OT" in llm_names
    assert "peak_mask_noon_peak_HUFL" in llm_names
    assert "hour_distance_to_peak_noon_peak" in llm_names
    assert "day_of_week" in time_names
    assert "day_of_week" not in llm_names


def test_dataset_splits_windows_and_llm_masks_align(tmp_path):
    from data_provider.data_loader import TimeSeriesDataset

    rows = 120
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "feature_1": np.arange(rows, dtype=np.float32),
            "OT": np.arange(rows, dtype=np.float32) * 2.0,
        }
    )
    frame.to_csv(tmp_path / "toy.csv", index=False)
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
                        "losses": {"event_weighted_mse": {"enabled": True, "weight": 3.0}},
                        "features": {"peak_mask": True, "hour_distance_to_peak": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    train = TimeSeriesDataset(
        root_path=str(tmp_path),
        data_path="toy.csv",
        flag="train",
        size=(12, 6, 6),
        features="M",
        target="OT",
        data="Toy",
        use_zscore=True,
        use_llm_rule_features=True,
        llm_rule_path=str(rule_path),
    )
    val = TimeSeriesDataset(
        root_path=str(tmp_path),
        data_path="toy.csv",
        flag="val",
        size=(12, 6, 6),
        features="M",
        target="OT",
        data="Toy",
        use_zscore=True,
        use_llm_rule_features=True,
        llm_rule_path=str(rule_path),
        scaler=train.scaler,
    )

    seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_llm, seq_y_llm, seq_y_masks = train[0]

    assert len(train) > len(val)
    assert seq_x.shape == (12, 2)
    assert seq_y.shape == (12, 2)
    assert seq_x_mark.shape[0] == 12
    assert seq_y_mark.shape[0] == 12
    assert seq_x_llm.shape[0] == 12
    assert seq_y_llm.shape[0] == 12
    assert seq_y_masks.shape == (12, 3, 2)
    assert train.target_dim == 2
    assert train.llm_feature_dim > 0


def test_dlinear_output_shape_with_extra_llm_features():
    from models.DLinear import DLinear

    args = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=5,
        c_out=2,
        individual=False,
        moving_avg=3,
        use_revin=False,
    )
    model = DLinear(args)
    x = torch.randn(8, 24, 5)

    y = model(x)

    assert y.shape == (8, 12, 2)


def test_gru_backbone_output_shape():
    from models.RNN import RecurrentForecastModel

    args = SimpleNamespace(
        seq_len=24,
        pred_len=12,
        enc_in=5,
        c_out=2,
        rnn_hidden_size=8,
        rnn_layers=1,
        rnn_type="GRU",
        dropout=0.0,
    )
    model = RecurrentForecastModel(args)
    x = torch.randn(4, 24, 5)

    y = model(x)

    assert y.shape == (4, 12, 2)


def test_dataset_aware_loss_uses_masks_and_returns_all_components():
    from losses.dataset_aware_loss import DatasetAwareLoss

    pred = torch.tensor([[[1.0], [3.0], [1.0]]])
    true = torch.zeros_like(pred)
    masks = torch.tensor([[[[0.0], [0.0], [0.0]], [[1.0], [1.0], [1.0]], [[0.0], [0.0], [0.0]]]])
    criterion = DatasetAwareLoss(
        {
            "use_event_weighted_loss": True,
            "event_weight": 2.0,
            "use_zero_consistency_loss": True,
            "zero_weight": 1.0,
            "use_peak_shape_loss": True,
            "peak_weight": 0.5,
            "use_diff_loss": True,
            "diff_weight": 0.1,
            "use_freq_loss": True,
            "freq_weight": 0.1,
        }
    )

    loss_dict = criterion(pred, true, batch_masks=masks)

    expected_keys = {
        "loss",
        "base_loss",
        "event_loss",
        "zero_loss",
        "peak_loss",
        "diff_loss",
        "freq_loss",
        "nonevent_loss",
        "distill_loss",
    }
    assert set(loss_dict) == expected_keys
    assert loss_dict["event_loss"] > 0
    assert loss_dict["zero_loss"] > 0
    assert torch.isclose(loss_dict["peak_loss"], torch.tensor(0.0))
    assert loss_dict["loss"] > loss_dict["base_loss"]


def test_zero_consistency_uses_scaled_zero_target_per_channel():
    from losses.dataset_aware_loss import DatasetAwareLoss

    pred = torch.zeros(1, 2, 2)
    true = torch.zeros_like(pred)
    masks = torch.ones(1, 2, 3, 2)
    criterion = DatasetAwareLoss(
        {
            "use_zero_consistency_loss": True,
            "zero_target": [2.0, -1.0],
        }
    )

    loss_dict = criterion(pred, true, batch_masks=masks)

    assert torch.isclose(loss_dict["zero_loss"], torch.tensor(1.5))


def test_peak_shape_loss_includes_boundary_peaks():
    from losses.dataset_aware_loss import DatasetAwareLoss

    pred = torch.tensor([[[1.0], [4.0], [2.0]]])
    true = torch.zeros_like(pred)
    masks = torch.zeros(1, 3, 3, 1)
    masks[:, 0, 2, :] = 1.0
    criterion = DatasetAwareLoss({"use_peak_shape_loss": True, "peak_window_size": 1})

    loss_dict = criterion(pred, true, batch_masks=masks)

    assert loss_dict["peak_loss"] > 0


def test_ett_short_dataset_does_not_fallback_to_ratio_split(tmp_path):
    from data_provider.data_loader import TimeSeriesDataset

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=200, freq="h"),
            "OT": np.arange(200, dtype=np.float32),
        }
    )
    frame.to_csv(tmp_path / "ETTh1.csv", index=False)

    with pytest.raises(ValueError, match="requires fixed 12/4/4 split"):
        TimeSeriesDataset(
            root_path=str(tmp_path),
            data_path="ETTh1.csv",
            flag="train",
            size=(96, 48, 96),
            features="S",
            target="OT",
            data="ETTh1",
        )


def test_rule_adapter_and_hard_intervention_use_future_rule_inputs():
    from models.rule_adapter import RuleAdapter, apply_hard_intervention

    pred_base = torch.zeros(2, 4, 3)
    future_llm = torch.ones(2, 4, 5)
    masks = torch.zeros(2, 4, 3, 3)
    masks[:, :, 0, :] = 1.0
    masks[:, :, 1, :] = 1.0
    adapter = RuleAdapter(feature_dim=5, c_out=3, hidden_dim=4)

    pred = adapter(pred_base, future_llm, masks)
    hard = apply_hard_intervention(pred_base, masks, torch.tensor([1.0, 2.0, 3.0]))

    assert pred.shape == pred_base.shape
    assert not torch.allclose(pred, pred_base)
    assert torch.allclose(hard, torch.tensor([1.0, 2.0, 3.0]).view(1, 1, 3).expand_as(hard))


def test_loss_config_reads_diff_frequency_and_warns_unknown():
    from llm_rules.rule_parser import loss_config_from_rules
    from llm_rules.rule_schema import LLMRules, RulePattern

    rules = LLMRules(
        dataset_name="unit",
        patterns=[
            RulePattern(
                name="trend",
                type="event",
                condition={"kind": "hourly", "hour": 0},
                losses={
                    "diff": {"enabled": True, "weight": 0.2},
                    "frequency": {"enabled": True, "weight": 0.3},
                    "unknown_loss": {"enabled": True, "weight": 1.0},
                },
            )
        ],
    )

    with pytest.warns(UserWarning, match="Unsupported loss"):
        config = loss_config_from_rules(rules)

    assert config["use_diff_loss"] is True
    assert config["diff_weight"] == 0.2
    assert config["use_freq_loss"] is True
    assert config["freq_weight"] == 0.3


def test_metrics_report_mae_for_zero_and_peak_events_with_channel_masks():
    from utils.metrics import metric

    pred = np.array([[[2.0, 0.0], [0.0, 4.0]]], dtype=np.float32)
    true = np.zeros_like(pred)
    masks = np.zeros((1, 2, 3, 2), dtype=np.float32)
    masks[:, 0, 1, 0] = 1.0
    masks[:, 1, 2, 1] = 1.0

    values = metric(pred, true, masks=masks)

    assert values["zero_event_mae"] == pytest.approx(2.0)
    assert values["peak_event_mae"] == pytest.approx(4.0)
