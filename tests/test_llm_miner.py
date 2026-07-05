import json

import numpy as np
import pandas as pd
import pytest


def test_dataset_summary_uses_train_split_only(tmp_path):
    from llm_miner.build_dataset_summary import build_summary

    rows = 100
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "OT": np.arange(rows, dtype=np.float32),
        }
    )
    csv_path = tmp_path / "toy.csv"
    out_path = tmp_path / "summary.json"
    frame.to_csv(csv_path, index=False)

    summary = build_summary(
        root_path=str(tmp_path),
        data_path="toy.csv",
        data="Toy",
        target="OT",
        seq_len=12,
        output_path=str(out_path),
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["split"] == "train"
    assert payload["row_count"] == 70
    assert payload["time_range"]["end"] == "2024-01-03 21:00:00"


def test_train_only_miner_pipeline_generates_summary_candidates_prompt_and_validation(tmp_path):
    from llm_miner.run_miner import run_miner

    rows = 96
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "HUFL": np.sin(np.arange(rows) / 6.0).astype(np.float32),
            "OT": np.where(np.arange(rows) % 24 == 0, 0.0, np.arange(rows) / 100.0).astype(np.float32),
        }
    )
    frame.to_csv(tmp_path / "toy.csv", index=False)
    output_dir = tmp_path / "artifacts" / "Toy"

    outputs = run_miner(
        data="Toy",
        root_path=str(tmp_path),
        data_path="toy.csv",
        features="M",
        target="OT",
        seq_len=12,
        pred_len=6,
        output_dir=str(output_dir),
        near_zero_eps=1e-5,
    )

    summary = json.loads((output_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    candidates = json.loads((output_dir / "candidate_rules.json").read_text(encoding="utf-8"))
    prompt = (output_dir / "llm_prompt.md").read_text(encoding="utf-8")

    assert outputs["summary_path"] == output_dir / "dataset_summary.json"
    assert summary["analysis_scope"] == "train_only"
    assert summary["num_timesteps"] == 67
    assert summary["train_end_time"] == "2024-01-03 18:00:00"
    assert candidates["analysis_scope"] == "train_only"
    assert candidates["zero_event_candidates"]
    assert "Do not predict future values." in prompt
    assert "Do not use validation or test information." in prompt
    assert (output_dir / "figures" / "full_train_series_overview.png").exists()


def test_parse_and_validate_llm_response_reports_dataset_mismatch_and_train_leakage(tmp_path):
    from llm_miner.parse_llm_response import parse_response
    from llm_miner.validate_rules import validate_rules

    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="h"),
            "OT": np.arange(100, dtype=np.float32),
        }
    )
    frame.to_csv(tmp_path / "toy.csv", index=False)
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "dataset_name": "OtherDataset",
                "analysis_scope": "train_only",
                "patterns": [
                    {
                        "name": "bad_periodic",
                        "type": "zero_event",
                        "condition": {"kind": "calendar_periodic", "month_interval": 2, "day": 1},
                        "affected_variables": ["MISSING"],
                        "time_range": "whole_day",
                        "support_count": 1,
                        "confidence": 1.2,
                        "evidence_windows": [["2024-01-05 00:00:00", "2024-01-05 01:00:00"]],
                        "losses": {"unknown_loss": {"enabled": True, "weight": 1.0}},
                        "features": {"unknown_feature": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rule_path = tmp_path / "generated" / "OtherDataset_rules.json"

    parse_response(str(response_path), str(rule_path))
    report = validate_rules(
        rule_path=str(rule_path),
        data="Toy",
        root_path=str(tmp_path),
        data_path="toy.csv",
        features="M",
        target="OT",
        seq_len=12,
        output_dir=str(tmp_path / "validation"),
    )

    assert report["ok"] is False
    messages = "\n".join(report["errors"] + report["warnings"])
    assert "dataset_name" in messages
    assert "anchor" in messages
    assert "affected variable" in messages
    assert "Unsupported loss" in messages
    assert "Unsupported feature" in messages
    assert "confidence" in messages
    assert "outside train split" in messages


def test_metrics_include_event_point_counts():
    from utils.metrics import metric

    pred = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    true = np.zeros_like(pred)
    masks = np.zeros((1, 2, 3, 2), dtype=np.float32)
    masks[:, 0, 0, 0] = 1.0
    masks[:, 1, 1, :] = 1.0

    values = metric(pred, true, masks=masks)

    assert values["num_event_points"] == 1
    assert values["num_zero_event_points"] == 2


def test_loss_config_can_be_saved_for_dataset_aware_runs(tmp_path):
    from exp.exp_long_term_forecasting import save_loss_config
    from losses.dataset_aware_loss import DatasetAwareLoss

    criterion = DatasetAwareLoss(
        {
            "use_dataset_aware_loss": True,
            "use_event_weighted_loss": True,
            "event_weight": 5.0,
        }
    )

    save_loss_config(criterion, tmp_path)

    payload = json.loads((tmp_path / "loss_config.json").read_text(encoding="utf-8"))
    assert payload["use_event_weighted_loss"] is True
    assert payload["event_weight"] == 5.0


def test_core_innovation_script_and_result_summarizer_exist():
    assert "llm_rule_features_plus_loss" in open("scripts/run_ettm1_core_innovation.sh", encoding="utf-8").read()
    assert "summarize_core_results" in open("analysis/summarize_core_results.py", encoding="utf-8").read()
