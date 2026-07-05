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
    from llm_rules.feature_generator import generate_llm_features
    from llm_rules.mask_generator import generate_event_mask
    from llm_rules.rule_schema import LLMRules, RulePattern

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=48, freq="h")
    rules = LLMRules(
        dataset_name="unit",
        patterns=[
            RulePattern(
                name="first_day",
                type="zero_event",
                condition={"kind": "calendar_periodic", "month_interval": 2, "day": 1},
                affected_variables="all",
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

    masks = generate_event_mask(timestamps, rules)
    features, names = generate_llm_features(timestamps, rules)

    assert masks["first_day"].sum() == 24
    assert masks["noon_peak"].sum() == 2
    assert masks["monday"].sum() == 24
    assert masks["event_mask"].shape == (48, 1)
    assert features.shape[0] == 48
    assert "event_mask_first_day" in names
    assert "peak_mask_noon_peak" in names
    assert "hour_distance_to_peak_noon_peak" in names
    assert "day_of_week" in names


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
        use_llm_features=True,
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
        use_llm_features=True,
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
    assert seq_y_masks.shape == (12, 3)
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


def test_dataset_aware_loss_uses_masks_and_returns_all_components():
    from losses.dataset_aware_loss import DatasetAwareLoss

    pred = torch.tensor([[[1.0], [3.0], [1.0]]])
    true = torch.zeros_like(pred)
    masks = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
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
    }
    assert set(loss_dict) == expected_keys
    assert loss_dict["event_loss"] > 0
    assert loss_dict["zero_loss"] > 0
    assert torch.isclose(loss_dict["peak_loss"], torch.tensor(0.0))
    assert loss_dict["loss"] > loss_dict["base_loss"]
