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


def test_multidataset_powershell_script_generates_dataset_specific_rules_before_training():
    script = open("scripts/run_multidataset_llm_rulegate.ps1", encoding="utf-8").read()

    assert "analysis/generate_dataset_llm_rules.py" in script
    assert "llm_rules/generated_rules/$Data`_rules.json" in script
    assert "--llm_rule_path" in script
    assert "$RulePath" in script
    assert "example_rules/ETTm1_rules.json" not in script


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
