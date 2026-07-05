"""Chronological CSV dataset for long-term forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_provider.timefeatures import time_features
from llm_rules.feature_generator import (
    generate_llm_rule_features,
    generate_oracle_features,
    generate_standard_time_features,
)
from llm_rules.mask_generator import generate_event_mask
from llm_rules.rule_parser import parse_llm_rules
from utils.scaler import StandardScaler


class TimeSeriesDataset(Dataset):
    """A leakage-safe sliding-window dataset for CSV time-series files."""

    flag_map = {"train": 0, "val": 1, "test": 2}
    ett_hourly = {"ETTh1", "ETTh2"}
    ett_minute = {"ETTm1", "ETTm2"}

    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = "train",
        size: Sequence[int] | None = None,
        features: str = "S",
        target: str = "OT",
        data: str = "custom",
        use_zscore: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        use_llm_features: bool = False,
        use_standard_time_features: bool = False,
        use_llm_rule_features: bool | None = None,
        use_oracle_features: bool = False,
        llm_rule_path: str | None = None,
        scaler: StandardScaler | None = None,
    ):
        if flag not in self.flag_map:
            raise ValueError(f"flag must be one of {sorted(self.flag_map)}, got {flag}")
        self.seq_len, self.label_len, self.pred_len = tuple(size or (96, 48, 96))
        self.root_path = Path(root_path)
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.data = data
        self.use_zscore = _flag(use_zscore)
        self.timeenc = timeenc
        self.freq = freq
        self.use_standard_time_features = _flag(use_standard_time_features)
        if use_llm_rule_features is None:
            self.use_llm_rule_features = _flag(use_llm_features)
        else:
            self.use_llm_rule_features = _flag(use_llm_rule_features)
        self.use_oracle_features = _flag(use_oracle_features)
        self.llm_rule_path = llm_rule_path
        self.scaler = scaler

        self.mask_names = ["event_mask", "zero_mask", "peak_mask"]
        self.llm_feature_names: list[str] = []
        self.target_indices: list[int] = []
        self.target_columns: list[str] = []
        self.input_columns: list[str] = []
        self.feature_dim = 0
        self.target_dim = 0
        self.llm_feature_dim = 0
        self.zero_target = np.zeros((0,), dtype=np.float32)

        self.__read_data__()

    def __read_data__(self):
        file_path = self.root_path / self.data_path
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df_raw = pd.read_csv(file_path)
        if df_raw.empty:
            raise ValueError(f"CSV file is empty: {file_path}")
        date_col = df_raw.columns[0]
        df_raw[date_col] = pd.to_datetime(df_raw[date_col])

        input_cols, target_indices, target_columns = self._select_columns(df_raw, date_col)
        input_values = df_raw[input_cols].to_numpy(dtype=np.float32)
        self.input_columns = input_cols
        self.target_indices = target_indices
        self.target_columns = target_columns
        self.feature_dim = len(input_cols)
        self.target_dim = len(target_indices)

        train_start, train_end = self._split_borders(len(df_raw), "train")
        if self.use_zscore:
            if self.scaler is None:
                self.scaler = StandardScaler().fit(input_values[train_start:train_end])
            scaled_values = self.scaler.transform(input_values)
        else:
            self.scaler = self.scaler or StandardScaler().fit(np.zeros((1, self.feature_dim), dtype=np.float32))
            scaled_values = input_values

        self.zero_target = self._compute_zero_target()
        border1, border2 = self._split_borders(len(df_raw), self.flag)
        dates = pd.DatetimeIndex(df_raw[date_col].iloc[border1:border2])
        data_x = scaled_values[border1:border2]
        if self.features == "MS":
            data_y = scaled_values[border1:border2][:, target_indices]
        else:
            data_y = scaled_values[border1:border2][:, target_indices]

        self.data_x = data_x.astype(np.float32)
        self.data_y = data_y.astype(np.float32)
        self.data_stamp = time_features(dates, freq=self.freq).astype(np.float32)
        self.timestamps = dates

        rules = parse_llm_rules(self.llm_rule_path) if self.llm_rule_path else None
        self.llm_features, self.llm_feature_names = self._build_aux_features(dates, rules)
        self.llm_feature_dim = self.llm_features.shape[1]
        self.event_masks = self._build_mask_array(dates, rules)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x_llm = self.llm_features[s_begin:s_end]
        seq_y_llm = self.llm_features[r_begin:r_end]
        seq_y_masks = self.event_masks[r_begin:r_end]

        return (
            torch.from_numpy(seq_x).float(),
            torch.from_numpy(seq_y).float(),
            torch.from_numpy(seq_x_mark).float(),
            torch.from_numpy(seq_y_mark).float(),
            torch.from_numpy(seq_x_llm).float(),
            torch.from_numpy(seq_y_llm).float(),
            torch.from_numpy(seq_y_masks).float(),
        )

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform_target(self, data):
        """Inverse-transform target variables predicted by the model."""
        if not self.use_zscore:
            return data
        return self.scaler.inverse_transform(data, feature_indices=self.target_indices)

    def _select_columns(self, df_raw: pd.DataFrame, date_col: str):
        value_cols = [col for col in df_raw.columns if col != date_col]
        if self.target not in value_cols:
            raise ValueError(f"target '{self.target}' not found in columns: {value_cols}")

        if self.features == "S":
            return [self.target], [0], [self.target]
        if self.features == "M":
            return value_cols, list(range(len(value_cols))), value_cols
        if self.features == "MS":
            return value_cols, [value_cols.index(self.target)], [self.target]
        raise ValueError("features must be one of 'M', 'S', or 'MS'.")

    def _split_borders(self, total_len: int, flag: str):
        flag_idx = self.flag_map[flag]
        if self.data in self.ett_hourly:
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
            if total_len < test_end:
                raise ValueError(
                    f"{self.data} requires fixed 12/4/4 split, but total_len={total_len} < required={test_end}."
                )
            starts = [0, train_end - self.seq_len, val_end - self.seq_len]
            ends = [train_end, val_end, test_end]
            return max(0, starts[flag_idx]), min(total_len, ends[flag_idx])
        if self.data in self.ett_minute:
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
            if total_len < test_end:
                raise ValueError(
                    f"{self.data} requires fixed 12/4/4 split, but total_len={total_len} < required={test_end}."
                )
            starts = [0, train_end - self.seq_len, val_end - self.seq_len]
            ends = [train_end, val_end, test_end]
            return max(0, starts[flag_idx]), min(total_len, ends[flag_idx])

        num_train = int(total_len * 0.7)
        num_test = int(total_len * 0.2)
        num_val = total_len - num_train - num_test
        starts = [0, num_train - self.seq_len, total_len - num_test - self.seq_len]
        ends = [num_train, num_train + num_val, total_len]
        return max(0, starts[flag_idx]), min(total_len, ends[flag_idx])

    def _build_mask_array(self, dates, rules):
        if rules is None:
            return np.zeros((len(dates), len(self.mask_names), self.target_dim), dtype=np.float32)
        masks = generate_event_mask(dates, rules, target_columns=self.target_columns)
        return np.stack([masks[name] for name in self.mask_names], axis=1).astype(np.float32)

    def _build_aux_features(self, dates, rules):
        arrays: list[np.ndarray] = []
        names: list[str] = []
        if self.use_standard_time_features:
            values, value_names = generate_standard_time_features(dates)
            arrays.append(values)
            names.extend([f"standard_time::{name}" for name in value_names])
        if self.use_llm_rule_features and rules is not None:
            values, value_names = generate_llm_rule_features(dates, rules, target_columns=self.target_columns)
            if values.shape[1] > 0:
                arrays.append(values)
                names.extend([f"llm_rule::{name}" for name in value_names])
        if self.use_oracle_features and rules is not None:
            values, value_names = generate_oracle_features(dates, rules, target_columns=self.target_columns)
            if values.shape[1] > 0:
                arrays.append(values)
                names.extend([f"oracle::{name}" for name in value_names])
        if not arrays:
            return np.zeros((len(dates), 0), dtype=np.float32), []
        return np.concatenate(arrays, axis=1).astype(np.float32), names

    def _compute_zero_target(self) -> np.ndarray:
        if not self.use_zscore:
            return np.zeros(self.target_dim, dtype=np.float32)
        zeros = np.zeros((1, self.feature_dim), dtype=np.float32)
        return self.scaler.transform(zeros)[0, self.target_indices].astype(np.float32)


def _flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)
