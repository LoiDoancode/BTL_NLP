"""
EncoderLayer & DecoderLayer (hỗ trợ Pre-LayerNorm).
Owner: Hoàng Lâm Tùng (24021661)
"""

import torch
import torch.nn as nn

from models.scratch.attention import MultiHeadAttention
from models.scratch.ffn import FeedForwardNetwork


class EncoderLayer(nn.Module):
    """Một khối Encoder gồm Multi-Head Self-Attention và Feed-Forward Network."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pre_ln: bool = True,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pre_ln:
            attn_in = self.norm1(x)
            x = x + self.dropout1(self.self_attn(attn_in, attn_in, attn_in, mask=src_mask))
            ffn_in = self.norm2(x)
            return x + self.dropout2(self.ffn(ffn_in))

        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask=src_mask)))
        return self.norm2(x + self.dropout2(self.ffn(x)))


class DecoderLayer(nn.Module):
    """Một khối Decoder gồm Masked Self-Attention, Cross-Attention và Feed-Forward Network."""

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pre_ln: bool = True,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_ln:
            self_in = self.norm1(x)
            x = x + self.dropout1(self.self_attn(self_in, self_in, self_in, mask=tgt_mask))
            cross_in = self.norm2(x)
            x = x + self.dropout2(self.cross_attn(cross_in, enc_out, enc_out, mask=src_mask))
            ffn_in = self.norm3(x)
            return x + self.dropout3(self.ffn(ffn_in))

        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask=tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, enc_out, enc_out, mask=src_mask)))
        return self.norm3(x + self.dropout3(self.ffn(x)))
