from types import SimpleNamespace

import numpy as np
import pandas as pd


def test_dataset_prediction_horizon_timestamps_align_with_masks():
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
        freq="h",
        use_llm_features=0,
        use_standard_time_features=0,
        use_llm_rule_features=0,
        use_oracle_features=0,
        llm_rule_path="./llm_rules/example_rules/ETTm1_rules.json",
        batch_size=8,
        num_workers=0,
    )
    data_provider(args, "train")
    dataset, _ = data_provider(args, "test")

    index = 674
    item = dataset[index]
    seq_y = item[1].numpy()
    seq_y_masks = item[-1].numpy()
    s_end = index + dataset.seq_len
    r_begin = s_end - dataset.label_len
    r_end = r_begin + dataset.label_len + dataset.pred_len
    seq_y_timestamps = dataset.timestamps[r_begin:r_end]
    pred_timestamps = seq_y_timestamps[-dataset.pred_len :]

    assert len(seq_y) == dataset.label_len + dataset.pred_len
    assert seq_y_masks.shape[0] == seq_y.shape[0]
    assert str(pred_timestamps[0]) == "2017-10-31 00:30:00"
    assert str(pred_timestamps[-1]) == "2017-11-01 00:15:00"

    zero_hits = np.argwhere(seq_y_masks[-dataset.pred_len :, 1, :] > 0)
    assert str(pred_timestamps[int(zero_hits[0, 0])]) == "2017-11-01 00:00:00"
    assert str(pred_timestamps[int(zero_hits[-1, 0])]) == "2017-11-01 00:15:00"


def test_lag_search_recovers_known_shift():
    from analysis.diagnose_split_offset import lag_search

    timestamps = pd.date_range("2020-01-01", periods=16, freq="15min")
    rule_mask = np.zeros((16, 1), dtype=np.float32)
    true_mask = np.zeros((16, 1), dtype=np.float32)
    rule_mask[[4, 5], 0] = 1.0
    true_mask[[7, 8], 0] = 1.0

    result = lag_search(rule_mask, true_mask, timestamps, lag_min=-8, lag_max=8)

    assert result["best_lag_steps"] == 3
    assert result["best_overlap"] == 2
    assert result["best_precision"] == 1.0
    assert result["best_recall"] == 1.0


def test_offset_predictability_flags_test_only_regime_shift():
    from analysis.diagnose_split_offset import classify_offset_predictability

    result = classify_offset_predictability(
        train_val_offsets=[0, -1, 3, 6, -1, -1, 2, 0],
        test_offsets=[29, 34],
    )

    assert result["predictable_from_train_val"] is False
    assert result["post_hoc_test_regime"] is True
    assert result["test_offsets_outside_train_val_range"] is True


def test_calendar_window_rule_condition_hits_window():
    from llm_rules.mask_generator import generate_event_mask

    timestamps = pd.date_range("2020-01-01 21:00:00", periods=25, freq="15min")
    rules = {
        "dataset_name": "toy",
        "patterns": [
            {
                "name": "first_day_window",
                "type": "zero_event",
                "condition": {
                    "kind": "calendar_window",
                    "anchor": "2020-01-01 00:00:00",
                    "month_interval": 1,
                    "center_day": 2,
                    "window_hours": 1,
                },
                "affected_variables": "all",
            }
        ],
    }

    masks = generate_event_mask(timestamps, rules, target_columns=["OT"])
    hit_times = [str(timestamps[idx]) for idx in np.flatnonzero(masks["zero_mask"][:, 0] > 0)]

    assert hit_times[0] == "2020-01-01 23:00:00"
    assert "2020-01-02 00:00:00" in hit_times
    assert hit_times[-1] == "2020-01-02 01:00:00"


def test_calendar_window_rule_condition_respects_center_hour():
    from llm_rules.mask_generator import generate_event_mask

    timestamps = pd.date_range("2020-01-02 05:00:00", periods=9, freq="30min")
    rules = {
        "dataset_name": "toy",
        "patterns": [
            {
                "name": "first_day_morning_window",
                "type": "zero_event",
                "condition": {
                    "kind": "calendar_window",
                    "anchor": "2020-01-01 00:00:00",
                    "month_interval": 1,
                    "center_day": 2,
                    "center_hour": 7,
                    "window_hours": 1,
                },
                "affected_variables": "all",
            }
        ],
    }

    masks = generate_event_mask(timestamps, rules, target_columns=["OT"])
    hit_times = [str(timestamps[idx]) for idx in np.flatnonzero(masks["zero_mask"][:, 0] > 0)]

    assert hit_times == [
        "2020-01-02 06:00:00",
        "2020-01-02 06:30:00",
        "2020-01-02 07:00:00",
        "2020-01-02 07:30:00",
        "2020-01-02 08:00:00",
    ]


def test_shift_mask_array_moves_without_wraparound():
    from data_provider.data_loader import shift_mask_array

    mask = np.zeros((5, 1, 1), dtype=np.float32)
    mask[1, 0, 0] = 1.0

    shifted_forward = shift_mask_array(mask, 2)
    shifted_backward = shift_mask_array(mask, -1)

    assert np.flatnonzero(shifted_forward[:, 0, 0]).tolist() == [3]
    assert np.flatnonzero(shifted_backward[:, 0, 0]).tolist() == [0]
