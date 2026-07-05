"""Build train-only visual diagnostics for LLM/VLM review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from llm_miner.common import default_output_dir, load_train_frame, select_value_columns


FIGURE_NAMES = [
    "full_train_series_overview.png",
    "monthly_overview.png",
    "zero_candidate_windows.png",
    "hourly_profile.png",
    "calendar_heatmap.png",
    "candidate_event_zoom.png",
]


def build_visualizations(
    root_path: str,
    data_path: str,
    data: str,
    target: str = "OT",
    seq_len: int = 96,
    features: str = "M",
    output_dir: str | None = None,
    candidate_rules_path: str | None = None,
):
    """Generate train-only PNG figures without relying on GUI plotting backends."""
    train, date_col, _ = load_train_frame(root_path, data_path, data, seq_len)
    _, target_columns = select_value_columns(train, date_col, features, target)
    figures_dir = default_output_dir(data, output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(candidate_rules_path)
    target_col = target_columns[-1]

    paths = [figures_dir / name for name in FIGURE_NAMES]
    _draw_line_chart(paths[0], train[target_col].to_numpy(), f"full train series: {target_col}", _time_label(train, date_col))
    _draw_bar_chart(paths[1], _monthly_values(train, date_col, target_col), f"monthly overview: {target_col}")
    _draw_zero_chart(paths[2], train, date_col, target_col, candidates)
    _draw_bar_chart(paths[3], _hourly_values(train, date_col, target_col), f"hourly profile: {target_col}")
    _draw_heatmap(paths[4], _calendar_matrix(train, date_col, target_col), f"calendar heatmap: {target_col}")
    _draw_candidate_zoom(paths[5], train, date_col, target_col, candidates)
    return paths


def build_visualization(root_path: str, data_path: str, data: str, target: str, seq_len: int, output_path: str):
    """Backward-compatible single-figure helper."""
    train, date_col, _ = load_train_frame(root_path, data_path, data, seq_len)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _draw_line_chart(output, train[target].to_numpy(), f"{data} train target: {target}", _time_label(train, date_col))
    return output


def _draw_line_chart(path: Path, values, title: str, subtitle: str, markers: list[int] | None = None):
    width, height = 1000, 320
    margin = 42
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    draw.text((margin, 10), title, fill="black")
    draw.text((margin, height - 24), subtitle[:150], fill="black")
    arr = np.asarray(values, dtype=float)
    if arr.size > 1:
        y_min, y_max = float(np.nanmin(arr)), float(np.nanmax(arr))
        if abs(y_max - y_min) < 1e-12:
            y_max = y_min + 1.0
        xs = np.linspace(margin, width - margin, arr.size)
        ys = height - margin - (arr - y_min) / (y_max - y_min) * (height - 2 * margin)
        points = [(float(x), float(y)) for x, y in zip(xs, ys)]
        draw.line(points, fill="steelblue", width=2)
        for marker in markers or []:
            if 0 <= marker < len(xs):
                x = float(xs[marker])
                draw.line((x, margin, x, height - margin), fill="red", width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_bar_chart(path: Path, values: dict, title: str):
    width, height = 800, 300
    margin = 42
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 10), title, fill="black")
    items = list(values.items())
    if items:
        vals = np.asarray([float(v) for _, v in items], dtype=float)
        v_min, v_max = float(vals.min()), float(vals.max())
        if abs(v_max - v_min) < 1e-12:
            v_max = v_min + 1.0
        bar_w = max(2, int((width - 2 * margin) / max(1, len(items))))
        for idx, (label, value) in enumerate(items):
            x0 = margin + idx * bar_w
            x1 = x0 + max(1, bar_w - 2)
            y = height - margin - (float(value) - v_min) / (v_max - v_min) * (height - 2 * margin)
            draw.rectangle((x0, y, x1, height - margin), fill="slateblue")
            if idx % max(1, len(items) // 8) == 0:
                draw.text((x0, height - margin + 4), str(label), fill="black")
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_heatmap(path: Path, matrix: np.ndarray, title: str):
    width, height = 800, 300
    margin = 42
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 10), title, fill="black")
    if matrix.size:
        v_min, v_max = float(np.nanmin(matrix)), float(np.nanmax(matrix))
        if abs(v_max - v_min) < 1e-12:
            v_max = v_min + 1.0
        rows, cols = matrix.shape
        cell_w = (width - 2 * margin) / cols
        cell_h = (height - 2 * margin) / rows
        for r in range(rows):
            for c in range(cols):
                ratio = (float(matrix[r, c]) - v_min) / (v_max - v_min)
                color = (int(255 * ratio), int(80), int(255 * (1 - ratio)))
                draw.rectangle(
                    (
                        margin + c * cell_w,
                        margin + r * cell_h,
                        margin + (c + 1) * cell_w,
                        margin + (r + 1) * cell_h,
                    ),
                    fill=color,
                )
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_zero_chart(path: Path, frame, date_col: str, target: str, candidates: dict):
    markers = []
    timestamps = frame[date_col].astype(str).tolist()
    for candidate in candidates.get("zero_event_candidates", [])[:50]:
        start = str(pd.Timestamp(candidate["start_time"]))
        if start in timestamps:
            markers.append(timestamps.index(start))
    _draw_line_chart(path, frame[target].to_numpy(), f"zero candidate windows: {target}", _time_label(frame, date_col), markers)


def _draw_candidate_zoom(path: Path, frame, date_col: str, target: str, candidates: dict):
    if candidates.get("zero_event_candidates"):
        start = pd.Timestamp(candidates["zero_event_candidates"][0]["start_time"])
        subset = frame[(frame[date_col] >= start - pd.Timedelta(hours=24)) & (frame[date_col] <= start + pd.Timedelta(hours=24))]
        if subset.empty:
            subset = frame.iloc[: min(len(frame), 256)]
    else:
        subset = frame.iloc[: min(len(frame), 256)]
    _draw_line_chart(path, subset[target].to_numpy(), f"candidate event zoom: {target}", _time_label(subset, date_col))


def _monthly_values(frame, date_col: str, target: str) -> dict:
    values = frame.set_index(date_col)[target].resample("ME").mean()
    return {str(idx.date()): float(value) for idx, value in values.items()}


def _hourly_values(frame, date_col: str, target: str) -> dict:
    values = frame.groupby(frame[date_col].dt.hour)[target].mean()
    return {str(idx): float(value) for idx, value in values.items()}


def _calendar_matrix(frame, date_col: str, target: str) -> np.ndarray:
    pivot = frame.pivot_table(index=frame[date_col].dt.weekday, columns=frame[date_col].dt.hour, values=target, aggfunc="mean")
    return pivot.fillna(0).to_numpy(dtype=float)


def _time_label(frame, date_col: str) -> str:
    if frame.empty:
        return "train time index"
    return f"train time index, start={frame[date_col].iloc[0]}, end={frame[date_col].iloc[-1]}"


def _load_candidates(path: str | None) -> dict:
    if not path:
        return {}
    candidate_path = Path(path)
    if not candidate_path.exists():
        return {}
    return json.loads(candidate_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Build train-only visualizations for offline LLM mining.")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
    parser.add_argument("--target", default="OT")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--candidate_rules_path", default=None)
    args = parser.parse_args()
    if args.output_path:
        build_visualization(args.root_path, args.data_path, args.data, args.target, args.seq_len, args.output_path)
    else:
        build_visualizations(
            root_path=args.root_path,
            data_path=args.data_path,
            data=args.data,
            features=args.features,
            target=args.target,
            seq_len=args.seq_len,
            output_dir=args.output_dir,
            candidate_rules_path=args.candidate_rules_path,
        )


if __name__ == "__main__":
    main()

