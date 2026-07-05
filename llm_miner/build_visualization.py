"""Build simple train-only visual diagnostics for LLM review."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm_miner.build_dataset_summary import _train_borders


def build_visualization(root_path: str, data_path: str, data: str, target: str, seq_len: int, output_path: str):
    """Plot target values from the train split only."""
    import matplotlib.pyplot as plt

    csv_path = Path(root_path) / data_path
    frame = pd.read_csv(csv_path)
    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    start, end = _train_borders(data, len(frame), seq_len)
    train = frame.iloc[start:end]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(train[date_col], train[target])
    plt.title(f"{data} train split target: {target}")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output


def main():
    parser = argparse.ArgumentParser(description="Build train-only visualization for offline LLM mining.")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    build_visualization(**vars(args))


if __name__ == "__main__":
    main()

