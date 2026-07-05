"""Simple GRU/LSTM forecasting backbone."""

from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentForecastModel(nn.Module):
    """Forecast future values from the final recurrent hidden state."""

    def __init__(self, args):
        super().__init__()
        cfg = vars(args) if hasattr(args, "__dict__") else dict(args)
        self.seq_len = int(cfg["seq_len"])
        self.pred_len = int(cfg["pred_len"])
        self.enc_in = int(cfg["enc_in"])
        self.c_out = int(cfg.get("c_out", self.enc_in))
        hidden_size = int(cfg.get("rnn_hidden_size", 64))
        num_layers = int(cfg.get("rnn_layers", 1))
        dropout = float(cfg.get("dropout", 0.0)) if num_layers > 1 else 0.0
        rnn_type = str(cfg.get("rnn_type", cfg.get("model", "GRU"))).upper()
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.enc_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, self.pred_len * self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast from ``x`` shaped ``[B, seq_len, C]``."""
        if x.ndim != 3:
            raise ValueError("RNN backbone expects input shaped [B, seq_len, C].")
        if x.shape[1] != self.seq_len or x.shape[2] != self.enc_in:
            raise ValueError(f"Expected input [B, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}.")
        _, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        last = hidden[-1]
        return self.projection(last).view(x.shape[0], self.pred_len, self.c_out)
