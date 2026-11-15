"""
Padding Mask & Causal (Look-Ahead) Mask.
Owner: Hoàng Lâm Tùng (24021661)
"""
import torch


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    raise NotImplementedError


def create_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    raise NotImplementedError
