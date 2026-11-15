"""
EncoderLayer & DecoderLayer (hỗ trợ Pre-LayerNorm).
Owner: Hoàng Lâm Tùng (24021661)
"""
import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
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

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class DecoderLayer(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
