"""
Sinusoidal Positional Encoding.
Owner: Hoàng Lâm Tùng (24021661)
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Mã hóa vị trí dạng sóng sin/cos cố định theo Vaswani et al. (2017)."""

    pe: torch.Tensor

    def __init__(self, d_model: int = 512, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor embedding shape (batch_size, seq_len, d_model).

        Returns:
            Tensor shape (batch_size, seq_len, d_model) sau khi cộng PE và áp dropout.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_len ({self.pe.size(1)})")
        x = x + self.pe[:, :seq_len, :].to(dtype=x.dtype)
        return self.dropout(x)
