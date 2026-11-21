"""
Padding Mask & Causal (Look-Ahead) Mask.
Owner: Hoàng Lâm Tùng (24021661)
"""

import torch


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Tạo Padding Mask với True tại vị trí token hợp lệ và False tại vị trí pad_idx.

    Args:
        seq: Tensor chỉ số token shape (batch_size, seq_len).
        pad_idx: ID của token padding.

    Returns:
        Boolean mask shape (batch_size, 1, 1, seq_len).
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Tạo Causal (Look-Ahead) Mask dạng tam giác dưới với True ở các vị trí được phép chú ý.

    Args:
        size: Độ dài chuỗi đích (tgt_len).
        device: Thiết bị đặt tensor mask.

    Returns:
        Boolean mask shape (1, 1, size, size).
    """
    mask = torch.tril(torch.ones((size, size), dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Kết hợp Padding Mask và Causal Mask cho Decoder self-attention.

    Args:
        tgt: Tensor chuỗi đích shape (batch_size, tgt_len).
        pad_idx: ID của token padding.

    Returns:
        Boolean mask shape (batch_size, 1, tgt_len, tgt_len).
    """
    pad_mask = create_padding_mask(tgt, pad_idx=pad_idx)
    causal_mask = create_causal_mask(tgt.size(1), device=tgt.device)
    return pad_mask & causal_mask
