import numpy as np
import pandas as pd


def test_calendar_window_candidate_miner_selects_validation_supported_rule():
    from analysis.mine_validated_calendar_windows import mine_calendar_window_candidates

    timestamps = pd.date_range("2020-01-01 00:00:00", periods=60 * 24, freq="h")
    train_event = np.zeros((len(timestamps), 1), dtype=np.float32)
    val_event = np.zeros((len(timestamps), 1), dtype=np.float32)
    train_mask = timestamps < pd.Timestamp("2020-02-01")
    val_mask = (timestamps >= pd.Timestamp("2020-02-01")) & (timestamps < pd.Timestamp("2020-03-01"))
    for idx, ts in enumerate(timestamps):
        if ts.day == 2 and ts.hour == 7:
            if train_mask[idx]:
                train_event[idx, 0] = 1.0
            if val_mask[idx]:
                val_event[idx, 0] = 1.0

    result = mine_calendar_window_candidates(
        timestamps=timestamps,
        train_event_mask=train_event,
        val_event_mask=val_event,
        train_scope_mask=np.asarray(train_mask, dtype=bool),
        val_scope_mask=np.asarray(val_mask, dtype=bool),
        target_columns=["OT"],
        anchor="2020-01-01 00:00:00",
        center_days=[1, 2, 3],
        center_hours=[0, 7, 12],
        window_hours_grid=[0.0],
        month_intervals=[1],
        min_train_support=1,
        min_val_support=1,
        min_val_precision=1.0,
        min_val_recall=1.0,
        max_candidates=3,
    )

    assert result["selected_candidates"]
    best = result["selected_candidates"][0]
    assert best["condition"]["center_day"] == 2
    assert best["condition"]["center_hour"] == 7
    assert best["val_precision"] == 1.0
    assert best["val_recall"] == 1.0


def test_candidate_miner_builds_rule_payload():
    from analysis.mine_validated_calendar_windows import build_rule_payload

    candidates = [
        {
            "name": "calendar_window_day2_hour7_w0.0_m1",
            "condition": {
                "kind": "calendar_window",
                "anchor": "2020-01-01 00:00:00",
                "month_interval": 1,
                "center_day": 2,
                "center_hour": 7,
                "window_hours": 0.0,
            },
            "affected_variables": ["OT"],
            "val_precision": 1.0,
            "val_recall": 1.0,
            "val_f1": 1.0,
        }
    ]

    payload = build_rule_payload("Toy", candidates)

    assert payload["dataset_name"] == "Toy"
    assert payload["analysis_scope"] == "train_val_validated"
    assert payload["patterns"][0]["condition"]["center_hour"] == 7
    assert payload["patterns"][0]["features"]["event_mask"] is True
    assert payload["patterns"][0]["features"]["days_to_event"] is True


def test_candidate_miner_rejects_validation_only_rule_when_train_precision_is_low():
    from analysis.mine_validated_calendar_windows import mine_calendar_window_candidates

    timestamps = pd.date_range("2020-01-01 00:00:00", periods=91 * 24, freq="h")
    train_event = np.zeros((len(timestamps), 1), dtype=np.float32)
    val_event = np.zeros((len(timestamps), 1), dtype=np.float32)
    train_mask = timestamps < pd.Timestamp("2020-03-01")
    val_mask = timestamps >= pd.Timestamp("2020-03-01")

    for idx, ts in enumerate(timestamps):
        if ts == pd.Timestamp("2020-01-02 07:00:00"):
            train_event[idx, 0] = 1.0
        if ts == pd.Timestamp("2020-03-02 07:00:00"):
            val_event[idx, 0] = 1.0

    result = mine_calendar_window_candidates(
        timestamps=timestamps,
        train_event_mask=train_event,
        val_event_mask=val_event,
        train_scope_mask=np.asarray(train_mask, dtype=bool),
        val_scope_mask=np.asarray(val_mask, dtype=bool),
        target_columns=["OT"],
        anchor="2020-01-01 00:00:00",
        center_days=[2],
        center_hours=[7],
        window_hours_grid=[0.0],
        month_intervals=[1],
        min_train_support=1,
        min_val_support=1,
        min_train_precision=0.75,
        min_val_precision=1.0,
        min_val_recall=1.0,
        max_candidates=3,
    )

    assert result["selected_candidates"] == []
    assert result["top_candidates"][0]["train_precision"] == 0.5
