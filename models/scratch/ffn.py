"""
Position-wise Feed-Forward Network (d_model=512, d_ff=2048).
Owner: Hoàng Lâm Tùng (24021661)
"""

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
