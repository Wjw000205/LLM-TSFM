"""DLinear backbone with moving-average decomposition."""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn

from models.layers.RevIN import RevIN


class MovingAverage(nn.Module):
    """Centered moving average for trend extraction."""

    def __init__(self, kernel_size: int):
        super().__init__()
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.kernel_size == 1:
            return x
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """Split a sequence into seasonal residual and trend components."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class DLinear(nn.Module):
    """Decomposition-Linear model for long-term forecasting."""

    def __init__(self, args: Namespace | dict):
        super().__init__()
        cfg = vars(args) if hasattr(args, "__dict__") else dict(args)
        self.seq_len = int(cfg["seq_len"])
        self.pred_len = int(cfg["pred_len"])
        self.enc_in = int(cfg["enc_in"])
        self.c_out = int(cfg.get("c_out", self.enc_in))
        self.individual = _flag(cfg.get("individual", False))
        self.use_revin = _flag(cfg.get("use_revin", False))
        self.revin_dim = int(cfg.get("raw_input_dim", min(self.enc_in, self.c_out)))
        self.revin_dim = max(1, min(self.revin_dim, self.enc_in))
        self.target_indices = list(cfg.get("target_indices", range(min(self.c_out, self.revin_dim))))

        self.decomposition = SeriesDecomp(int(cfg.get("moving_avg", 25)))
        if self.individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)])
            self.linear_trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.enc_in)])
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        self.channel_projection = nn.Identity()
        if self.c_out != self.enc_in:
            self.channel_projection = nn.Linear(self.enc_in, self.c_out)
        self.revin = RevIN(self.revin_dim) if self.use_revin else None
        self._reset_parameters()

    def forward(self, x):
        """Forecast future values from ``x`` shaped ``[B, seq_len, C]``."""
        if x.ndim != 3:
            raise ValueError("DLinear expects input shaped [B, seq_len, C].")
        if x.shape[1] != self.seq_len or x.shape[2] != self.enc_in:
            raise ValueError(f"Expected input [B, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}.")

        if self.revin is not None:
            raw = self.revin(x[..., : self.revin_dim], mode="norm")
            x = torch.cat([raw, x[..., self.revin_dim :]], dim=-1) if self.revin_dim < self.enc_in else raw

        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0), self.enc_in, self.pred_len, dtype=x.dtype, device=x.device
            )
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.enc_in):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        output = (seasonal_output + trend_output).permute(0, 2, 1)
        output = self.channel_projection(output)

        if self.revin is not None:
            indices = self.target_indices if len(self.target_indices) == self.c_out else None
            output = self.revin(output, mode="denorm", feature_indices=indices)
        return output

    def _reset_parameters(self):
        layers = []
        if self.individual:
            layers.extend(list(self.linear_seasonal))
            layers.extend(list(self.linear_trend))
        else:
            layers.extend([self.linear_seasonal, self.linear_trend])
        for layer in layers:
            nn.init.constant_(layer.weight, 1.0 / self.seq_len)
            nn.init.zeros_(layer.bias)


def _flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)
