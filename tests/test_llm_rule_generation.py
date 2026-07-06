import json
import subprocess
import sys

import pandas as pd


def test_build_dataset_profile_contains_dataset_specific_calendar_and_stats(tmp_path):
    from analysis.generate_dataset_llm_rules import build_dataset_profile

    path = tmp_path / "Toy.csv"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=48, freq="h"),
            "OT": list(range(48)),
            "HUFL": [1.0] * 48,
        }
    )
    df.to_csv(path, index=False)

    profile = build_dataset_profile(
        data="Toy",
        root_path=str(tmp_path),
        data_path="Toy.csv",
        features="M",
        target="OT",
    )

    assert profile["dataset_name"] == "Toy"
    assert profile["columns"] == ["OT", "HUFL"]
    assert profile["time"]["start"] == "2020-01-01 00:00:00"
    assert profile["time"]["freq_inferred"] == "h"
    assert profile["statistics"]["OT"]["mean"] == 23.5


def test_normalize_llm_rule_payload_forces_dataset_name_and_supported_features():
    from analysis.generate_dataset_llm_rules import normalize_llm_rule_payload
    from llm_rules.rule_parser import parse_llm_rules

    payload = {
        "dataset_name": "Wrong",
        "patterns": [
            {
                "name": "monthly_event",
                "type": "zero_event",
                "condition": {"kind": "calendar_periodic", "anchor": "2020-01-01 00:00:00", "day": 1},
                "affected_variables": ["OT"],
            }
        ],
    }

    normalized = normalize_llm_rule_payload(payload, dataset_name="Toy")

    assert normalized["dataset_name"] == "Toy"
    assert normalized["analysis_scope"] == "llm_dataset_specific_pretraining"
    assert normalized["patterns"][0]["features"]["event_mask"] is True
    assert normalized["patterns"][0]["features"]["days_to_event"] is True
    parse_llm_rules(normalized)


def test_normalize_llm_rule_payload_accepts_llm_list_features_and_losses():
    from analysis.generate_dataset_llm_rules import normalize_llm_rule_payload
    from llm_rules.rule_parser import parse_llm_rules

    payload = {
        "dataset_name": "Toy",
        "patterns": [
            {
                "name": "list_schema_event",
                "type": "event",
                "condition": {"kind": "hourly", "hours": [0, 1]},
                "features": ["event_mask", "days_to_event"],
                "losses": ["event_weighted_mse"],
            }
        ],
    }

    normalized = normalize_llm_rule_payload(payload, dataset_name="Toy")

    assert normalized["patterns"][0]["features"]["event_mask"] is True
    assert normalized["patterns"][0]["losses"]["event_weighted_mse"]["enabled"] is True
    parse_llm_rules(normalized)


def test_normalize_llm_rule_payload_accepts_llm_dict_loss_list():
    from analysis.generate_dataset_llm_rules import normalize_llm_rule_payload

    payload = {
        "dataset_name": "Toy",
        "patterns": [
            {
                "name": "dict_loss_event",
                "type": "calendar_event",
                "condition": {"kind": "hourly", "high_hours": [13]},
                "features": {"event_mask": True},
                "losses": [{"name": "event_weighted_mse", "weight": 3.0}],
            }
        ],
    }

    normalized = normalize_llm_rule_payload(payload, dataset_name="Toy")

    assert normalized["patterns"][0]["losses"]["event_weighted_mse"]["enabled"] is True
    assert normalized["patterns"][0]["losses"]["event_weighted_mse"]["weight"] == 3.0


def test_normalize_llm_rule_payload_filters_metadata_loss_objects():
    from analysis.generate_dataset_llm_rules import normalize_llm_rule_payload

    payload = {
        "dataset_name": "Toy",
        "patterns": [
            {
                "name": "metadata_loss_event",
                "type": "calendar_event",
                "condition": {"kind": "hourly", "hours": [13]},
                "features": {"event_mask": True},
                "losses": {
                    "recommended": {"enabled": True, "weight": 1.0},
                    "weighting": {"inside_window": 1.3},
                    "rationale": "metadata, not a local loss",
                },
            }
        ],
    }

    normalized = normalize_llm_rule_payload(payload, dataset_name="Toy")

    assert "recommended" not in normalized["patterns"][0]["losses"]
    assert normalized["patterns"][0]["losses"]["event_weighted_mse"]["enabled"] is True


def test_normalize_llm_rule_payload_drops_zero_consistency_by_default():
    from analysis.generate_dataset_llm_rules import normalize_llm_rule_payload

    payload = {
        "dataset_name": "Toy",
        "patterns": [
            {
                "name": "weak_zero_hypothesis",
                "type": "zero_event",
                "condition": {"kind": "calendar_window", "anchor": "2024-01-01 00:00:00"},
                "features": {"event_mask": True},
                "losses": {
                    "event_weighted_mse": {"enabled": True, "weight": 3.0},
                    "zero_consistency": {"enabled": True, "weight": 1.0},
                },
            }
        ],
    }

    normalized = normalize_llm_rule_payload(payload, dataset_name="Toy")

    losses = normalized["patterns"][0]["losses"]
    assert losses == {"event_weighted_mse": {"enabled": True, "weight": 3.0}}
    assert any("zero_consistency" in warning for warning in normalized["warnings"])


def test_normalize_llm_rule_payload_does_not_recur_weak_zero_windows():
    from analysis.generate_dataset_llm_rules import normalize_llm_rule_payload

    payload = {
        "dataset_name": "Toy",
        "patterns": [
            {
                "name": "one_off_zero",
                "type": "zero_event",
                "condition": {
                    "kind": "calendar_window",
                    "anchor": "2024-01-05 09:00:00",
                    "center_day": 5,
                    "center_hour": 9,
                    "month_interval": 2,
                    "window_hours": 4,
                },
                "affected_variables": ["OT"],
                "support_count": 1,
                "evidence_windows": [["2024-01-05 08:00:00", "2024-01-05 10:00:00"]],
                "features": {"event_mask": True},
                "losses": {"event_weighted_mse": {"enabled": True, "weight": 3.0}},
            }
        ],
    }

    normalized = normalize_llm_rule_payload(payload, dataset_name="Toy")

    condition = normalized["patterns"][0]["condition"]
    assert condition == {
        "kind": "calendar_window",
        "windows": [{"start": "2024-01-05 08:00:00", "end": "2024-01-05 10:00:00"}],
    }
    assert any("kept as explicit evidence windows" in warning for warning in normalized["warnings"])


def test_build_prompt_requires_legacy_rule_contract_and_sparse_events():
    from analysis.generate_dataset_llm_rules import _build_prompt

    prompt = _build_prompt(
        {
            "dataset_name": "Toy",
            "columns": ["OT"],
            "target": "OT",
            "analysis_scope": "train_only",
            "statistics": {},
        }
    )

    assert '"type": "zero_event"' in prompt
    assert '"condition": {"kind": "calendar_periodic", "anchor": "YYYY-MM-DD HH:MM:SS"' in prompt
    assert '"losses": {"event_weighted_mse": {"enabled": true, "weight": 5.0}}' in prompt
    assert '"features": {"event_mask": true, "days_to_event": true}' in prompt
    assert "Do not encode ordinary seasonality, hour-of-day, weekday, or month regimes as events." in prompt
    assert "Do not put recommended, weighting, rationale, or free-form metadata inside losses or features." in prompt
    assert "You are a hypothesis miner, not a verifier or final judge." in prompt
    assert "Do not validate, accept, reject, calibrate, or rank hypotheses inside the LLM output." in prompt
    assert "Return at most 3 patterns." in prompt
    assert "Do not output zero_consistency for GPT-generated hypotheses." in prompt
    assert "Do not invent recurring calendar rules from one-off near_zero evidence." in prompt
    assert "Do not return an empty patterns list when near_zero or peak candidate windows exist in the profile." in prompt
    assert "Generate candidate loss hypotheses even when confidence is low." in prompt
    assert "Validation/calibration decides whether to enable or reject each hypothesis." not in prompt
    assert "Prefer the previous ETTm1-style calendar_periodic zero_event contract" in prompt


def test_gpt55_condition_variants_generate_expected_event_timestamps():
    from llm_rules.mask_generator import generate_event_mask

    rules = {
        "dataset_name": "Toy",
        "patterns": [
            {
                "name": "afternoon",
                "type": "calendar_event",
                "condition": {"kind": "hourly", "high_hours": [13, 14], "low_hours": [3]},
                "affected_variables": "all",
                "features": {"event_mask": True},
            },
            {
                "name": "weekend",
                "type": "calendar_event",
                "condition": {"kind": "weekday", "high_weekdays": [5, 6]},
                "affected_variables": "all",
                "features": {"event_mask": True},
            },
            {
                "name": "explicit_window",
                "type": "calendar_event",
                "condition": {
                    "kind": "calendar_window",
                    "windows": [{"start": "2024-01-03 10:00:00", "end": "2024-01-03 11:00:00"}],
                },
                "affected_variables": "all",
                "features": {"event_mask": True},
            },
            {
                "name": "periods",
                "type": "calendar_event",
                "condition": {
                    "kind": "calendar_periodic",
                    "periods": [
                        {"label": "summer", "months": [7, 8]},
                        {"label": "mid_month", "days": [15]},
                    ],
                },
                "affected_variables": "all",
                "features": {"event_mask": True},
            },
        ],
    }
    timestamps = pd.to_datetime(
        [
            "2024-01-01 12:00:00",
            "2024-01-01 13:00:00",
            "2024-01-02 03:00:00",
            "2024-01-03 10:30:00",
            "2024-01-06 12:00:00",
            "2024-07-02 12:00:00",
            "2024-02-15 12:00:00",
        ]
    )

    masks = generate_event_mask(timestamps, rules, target_columns=["OT"])

    assert masks["event_mask"][:, 0].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_extract_json_object_from_chat_response():
    from analysis.generate_dataset_llm_rules import extract_json_object

    content = 'Here is JSON:\\n```json\\n{"dataset_name":"Toy","patterns":[]}\\n```'
    assert json.loads(extract_json_object(content))["dataset_name"] == "Toy"


def test_multidataset_script_generates_dataset_specific_rules_before_training():
    script = open("scripts/run_multidataset_llm_rulegate.sh", encoding="utf-8").read()

    assert "analysis/generate_dataset_llm_rules.py" in script
    assert "llm_rules/generated_rules/${DATA}_rules.json" in script
    assert "--llm_rule_path \"${RULE_PATH}\"" in script
    assert "example_rules/ETTm1_rules.json" not in script
    assert "# Use the LLM-generated loss config from the rule JSON." in script
    assert "--use_llm_rule_features 1" not in script
    assert "--event_weight \"${EVENT_WEIGHT}\"" not in script
    assert "analysis/evaluate_rule_gated_ensemble.py" not in script
    assert "--selection_metric guarded_event_mse" in script
    assert '--baseline_metric_path "./checkpoints/${BASELINE_SETTING}/validation_history.json"' in script


def test_multidataset_powershell_script_generates_dataset_specific_rules_before_training():
    script = open("scripts/run_multidataset_llm_rulegate.ps1", encoding="utf-8").read()

    assert "analysis/generate_dataset_llm_rules.py" in script
    assert "llm_rules/generated_rules/$Data`_rules.json" in script
    assert "--llm_rule_path" in script
    assert "$RulePath" in script
    assert "example_rules/ETTm1_rules.json" not in script
    assert "# Use the LLM-generated loss config from the rule JSON." in script
    assert "--use_llm_rule_features 1" not in script
    assert "--event_weight" not in script
    assert "analysis/evaluate_rule_gated_ensemble.py" not in script
    assert "--selection_metric guarded_event_mse" in script
    assert '--baseline_metric_path "./checkpoints/$BaselineSetting/validation_history.json"' in script


def test_generate_dataset_llm_rules_cli_runs_from_repo_root(tmp_path):
    csv_path = tmp_path / "Toy.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="h"),
            "OT": range(100),
        }
    ).to_csv(csv_path, index=False)
    rule_path = tmp_path / "Toy_rules.json"
    report_path = tmp_path / "Toy_report.json"

    subprocess.run(
        [
            sys.executable,
            "analysis/generate_dataset_llm_rules.py",
            "--data",
            "Toy",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "Toy.csv",
            "--seq_len",
            "12",
            "--dry_run_profile_only",
            "1",
            "--output_rule_path",
            str(rule_path),
            "--output_report_path",
            str(report_path),
        ],
        check=True,
    )

    assert json.loads(rule_path.read_text(encoding="utf-8"))["dataset_name"] == "Toy"
    assert json.loads(report_path.read_text(encoding="utf-8"))["split"] == "train"
