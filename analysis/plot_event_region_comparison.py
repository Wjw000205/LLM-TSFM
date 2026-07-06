"""Plot event-region prediction comparisons for long-tail loss experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluate_rule_gated_ensemble import load_test_masks
from data_provider.data_factory import data_provider


PALETTE = {
    "blue_main": "#0F4D92",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "teal": "#42949E",
    "neutral": "#CFCECE",
}


def plot_event_region_comparison(
    baseline_result_dir: str,
    event_result_dir: str,
    gated_result_dir: str | None,
    output_dir: str,
    channel: str = "OT",
    top_k: int = 6,
) -> dict:
    baseline_dir = Path(baseline_result_dir)
    event_dir = Path(event_result_dir)
    gated_dir = Path(gated_result_dir) if gated_result_dir else None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    target_columns = list(config.get("target_columns", []))
    if channel not in target_columns:
        raise ValueError(f"channel '{channel}' not in target columns: {target_columns}")
    channel_idx = target_columns.index(channel)

    true = _load_array(baseline_dir, "true_original.npy", "true_normalized.npy")
    baseline = _load_array(baseline_dir, "pred_original.npy", "pred_normalized.npy")
    event = _load_array(event_dir, "pred_original.npy", "pred_normalized.npy")
    gated = _load_array(gated_dir, "pred_original.npy", "pred_normalized.npy") if gated_dir else None
    masks = load_test_masks(baseline_dir / "config.json", expected_windows=baseline.shape[0])
    event_mask = masks[:, :, 0, channel_idx]
    timestamps = _horizon_timestamps(config, baseline.shape[0], baseline.shape[1])
    windows = _select_event_windows(event_mask, top_k=top_k)

    saved = []
    for rank, idx in enumerate(windows, start=1):
        paths = _plot_single_window(
            output=output,
            rank=rank,
            window_idx=idx,
            channel=channel,
            timestamps=timestamps[idx],
            mask=event_mask[idx],
            true=true[idx, :, channel_idx],
            baseline=baseline[idx, :, channel_idx],
            event=event[idx, :, channel_idx],
            gated=gated[idx, :, channel_idx] if gated is not None else None,
        )
        saved.extend(str(path) for path in paths)

    summary_paths = _plot_summary_grid(
        output=output,
        windows=windows,
        channel=channel,
        timestamps=timestamps,
        event_mask=event_mask,
        true=true[:, :, channel_idx],
        baseline=baseline[:, :, channel_idx],
        event=event[:, :, channel_idx],
        gated=gated[:, :, channel_idx] if gated is not None else None,
    )
    saved.extend(str(path) for path in summary_paths)
    manifest = {
        "baseline_result_dir": str(baseline_dir),
        "event_result_dir": str(event_dir),
        "gated_result_dir": str(gated_dir) if gated_dir else None,
        "channel": channel,
        "selected_windows": [int(idx) for idx in windows],
        "saved_files": saved,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_array(directory: Path, preferred: str, fallback: str) -> np.ndarray:
    path = directory / preferred
    if path.exists():
        return np.load(path)
    return np.load(directory / fallback)


def _horizon_timestamps(config: dict, num_windows: int, pred_len: int) -> np.ndarray:
    args = SimpleNamespace(**config)
    args.batch_size = int(getattr(args, "batch_size", 8))
    args.num_workers = 0
    if not hasattr(args, "timeenc"):
        args.timeenc = 0
    data_provider(args, "train")
    test_data, _ = data_provider(args, "test")
    seq_len = int(config["seq_len"])
    values = []
    for idx in range(num_windows):
        values.append(test_data.timestamps[idx + seq_len : idx + seq_len + pred_len].to_numpy())
    return np.asarray(values)


def _select_event_windows(event_mask: np.ndarray, top_k: int) -> list[int]:
    scores = event_mask.sum(axis=1)
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    min_gap = max(1, event_mask.shape[1] // 2)
    for idx in order:
        if scores[idx] <= 0:
            break
        if all(abs(int(idx) - existing) >= min_gap for existing in selected):
            selected.append(int(idx))
        if len(selected) >= top_k:
            break
    return selected


def _plot_single_window(
    output: Path,
    rank: int,
    window_idx: int,
    channel: str,
    timestamps,
    mask,
    true,
    baseline,
    event,
    gated,
) -> list[Path]:
    _apply_style()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _plot_lines(ax, timestamps, true, baseline, event, gated)
    _shade_event_regions(ax, timestamps, mask)
    ax.set_title(f"GPT-5.5 Long-tail Region {rank}: window {window_idx}, channel {channel}")
    ax.set_ylabel(f"{channel} value")
    ax.set_xlabel("Forecast timestamp")
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    fig.autofmt_xdate(rotation=25)
    return _save(fig, output / f"gpt55_event_window_{rank:02d}_{window_idx}_{channel}")


def _plot_summary_grid(output: Path, windows, channel, timestamps, event_mask, true, baseline, event, gated) -> list[Path]:
    if not windows:
        return []
    _apply_style()
    ncols = 2
    nrows = int(np.ceil(len(windows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.7 * nrows), squeeze=False)
    for ax, window_idx in zip(axes.ravel(), windows):
        _plot_lines(
            ax,
            timestamps[window_idx],
            true[window_idx],
            baseline[window_idx],
            event[window_idx],
            gated[window_idx] if gated is not None else None,
        )
        _shade_event_regions(ax, timestamps[window_idx], event_mask[window_idx])
        ax.set_title(f"window {window_idx}")
        ax.tick_params(axis="x", labelrotation=25)
    for ax in axes.ravel()[len(windows) :]:
        ax.axis("off")
    axes[0, 0].legend(loc="upper left", ncol=2, fontsize=8)
    fig.suptitle(f"GPT-5.5 Long-tail Loss Comparisons, channel {channel}", y=0.995)
    fig.tight_layout()
    return _save(fig, output / f"gpt55_event_examples_top{len(windows)}_{channel}")


def _plot_lines(ax, timestamps, true, baseline, event, gated) -> None:
    ax.plot(timestamps, true, color="black", linewidth=2.0, label="True")
    ax.plot(timestamps, baseline, color=PALETTE["blue_main"], linewidth=1.6, label="Baseline")
    ax.plot(timestamps, event, color=PALETTE["red_strong"], linewidth=1.3, label="Loss model")
    if gated is not None:
        ax.plot(timestamps, gated, color=PALETTE["green_3"], linewidth=1.6, label="Gated")


def _shade_event_regions(ax, timestamps, mask) -> None:
    for start, end in _runs(mask > 0):
        ax.axvspan(timestamps[start], timestamps[end], color=PALETTE["neutral"], alpha=0.28, linewidth=0)


def _runs(mask: np.ndarray):
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            yield start, idx - 1
            start = None
    if start is not None:
        yield start, len(mask) - 1


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot event-region comparisons.")
    parser.add_argument("--baseline_result_dir", required=True)
    parser.add_argument("--event_result_dir", required=True)
    parser.add_argument("--gated_result_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--channel", default="OT")
    parser.add_argument("--top_k", type=int, default=6)
    args = parser.parse_args()
    manifest = plot_event_region_comparison(**vars(args))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
