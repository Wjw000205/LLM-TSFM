import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_horizon_invariance_diagnostic_keeps_unique_event_timestamps_stable():
    from analysis.diagnose_event_mask_horizon_invariance import diagnose_horizon_invariance

    report = diagnose_horizon_invariance(
        data="ETTm1",
        root_path="./data/",
        data_path="ETTm1.csv",
        features="M",
        target="OT",
        seq_len=336,
        label_len=48,
        pred_lens=[96, 192, 336, 720],
        llm_rule_path="./llm_rules/generated_rules/ETTm1_peak_rules.json",
    )

    per_horizon = report["per_horizon"]
    unique_sets = [tuple(row["unique_event_timestamps"]) for row in per_horizon]

    assert report["errors"] == []
    assert all(unique_sets[0] == values for values in unique_sets[1:])
    assert per_horizon[0]["unique_event_timestamp_count"] > 0
    assert per_horizon[-1]["unique_event_timestamp_count"] > 0


def test_prediction_timestamp_debug_method_aligns_pred_mask_with_seq_y_tail():
    from data_provider.data_factory import data_provider

    args = SimpleNamespace(
        root_path="./data/",
        data_path="ETTm1.csv",
        data="ETTm1",
        features="M",
        target="OT",
        seq_len=336,
        label_len=48,
        pred_len=96,
        use_zscore=1,
        timeenc=0,
        freq="t",
        use_llm_features=0,
        use_standard_time_features=0,
        use_llm_rule_features=0,
        use_oracle_features=0,
        llm_rule_path="./llm_rules/generated_rules/ETTm1_peak_rules.json",
        batch_size=8,
        num_workers=0,
    )
    data_provider(args, "train")
    dataset, _ = data_provider(args, "test")

    index = 674
    item = dataset[index]
    seq_y_masks = item[-1].numpy()
    info = dataset.get_prediction_timestamps(index)

    s_end = index + dataset.seq_len
    r_begin = s_end - dataset.label_len
    r_end = r_begin + dataset.label_len + dataset.pred_len
    expected_pred_timestamps = dataset.timestamps[r_begin:r_end][-dataset.pred_len :]
    expected_pred_mask = seq_y_masks[-dataset.pred_len :]

    assert list(info["pred_timestamps"]) == list(expected_pred_timestamps)
    np.testing.assert_array_equal(info["pred_event_masks"], expected_pred_mask)
    assert info["pred_start"] == expected_pred_timestamps[0]
    assert info["pred_end"] == expected_pred_timestamps[-1]


def test_empty_event_mask_metric_returns_nan_warning_not_zero():
    from utils.metrics import metric

    pred = np.zeros((2, 4, 3), dtype=np.float32)
    true = np.ones_like(pred)
    masks = np.zeros((2, 4, 3, 3), dtype=np.float32)

    values = metric(pred, true, masks=masks)

    assert values["num_event_points"] == 0
    assert np.isnan(values["event_window_mse"])
    assert np.isnan(values["event_window_mae"])
    assert values["event_mask_warning"] == "empty_event_mask"
    assert values["summary_status"] == "not_applicable_empty_mask"


def test_global_event_mask_matches_dataset_split_slice(tmp_path):
    from data_provider.data_loader import TimeSeriesDataset
    from llm_rules.mask_generator import generate_event_mask

    rows = 240
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="D"),
            "OT": np.arange(rows, dtype=np.float32),
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
                        "name": "every_other_month_first_day",
                        "type": "peak_event",
                        "condition": {
                            "kind": "calendar_periodic",
                            "day": 1,
                            "month_interval": 2,
                        },
                        "affected_variables": "all",
                        "features": {"event_mask": True, "peak_mask": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = TimeSeriesDataset(
        root_path=str(tmp_path),
        data_path="toy.csv",
        flag="test",
        size=(12, 3, 6),
        features="S",
        target="OT",
        data="Toy",
        use_zscore=True,
        llm_rule_path=str(rule_path),
    )
    border1, border2 = dataset._split_borders(rows, "test")
    full_masks = generate_event_mask(frame["date"], str(rule_path), target_columns=["OT"])
    expected = np.stack([full_masks[name] for name in dataset.mask_names], axis=1).astype(np.float32)[border1:border2]

    np.testing.assert_array_equal(dataset.event_masks, expected)


def test_multidataset_summary_rejects_horizon_specific_event_rule_paths():
    from analysis.summarize_multidataset_peak_transfer_full_horizon import _validate_horizon_rule_config

    run = {"dataset": "ETTh1", "pred_len": 720}

    _validate_horizon_rule_config(
        {"llm_rule_path": "./llm_rules/generated_rules/ETTh1_peak_transfer_rules.json"},
        run,
        Path("results/generic"),
    )
    with pytest.raises(ValueError, match="horizon-specific event rule"):
        _validate_horizon_rule_config(
            {"llm_rule_path": "./llm_rules/generated_rules/ETTh1_p720_peak_transfer_rules.json"},
            run,
            Path("results/horizon-specific"),
        )
