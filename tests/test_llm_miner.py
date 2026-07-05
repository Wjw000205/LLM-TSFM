import json

import numpy as np
import pandas as pd


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
