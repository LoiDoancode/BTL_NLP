"""
Scaled Dot-Product Attention & Multi-Head Attention bằng PyTorch thuần.
Owner: Hoàng Lâm Tùng (24021661)
"""

import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Tính Scaled Dot-Product Attention: softmax((Q K^T) / sqrt(d_k) + M) V."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor shape (..., q_len, d_k).
            k: Key tensor shape (..., k_len, d_k).
            v: Value tensor shape (..., k_len, d_v).
            mask: Boolean mask broadcastable to (..., q_len, k_len) với True là vị trí hợp lệ.

        Returns:
            Tuple (output, attn_weights) với shape (..., q_len, d_v) và (..., q_len, k_len).
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            bool_mask = mask.bool()
            # Dùng finfo.min thay vì -inf để tránh NaN nếu cả hàng đều bị mask
            scores = scores.masked_fill(~bool_mask, torch.finfo(scores.dtype).min)
            attn_weights = torch.softmax(scores, dim=-1).masked_fill(~bool_mask, 0.0)
        else:
            attn_weights = torch.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention theo Vaswani et al. (2017) bằng PyTorch thuần."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.last_attn_weights: torch.Tensor | None = None

    @staticmethod
    def _normalize_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask.unsqueeze(0).unsqueeze(0)
        if mask.dim() == 3:
            return mask.unsqueeze(1)
        if mask.dim() == 4:
            return mask
        raise ValueError(f"Expected 2D, 3D, or 4D attention mask, got {mask.dim()}D")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor shape (batch_size, q_len, d_model).
            k: Key tensor shape (batch_size, k_len, d_model).
            v: Value tensor shape (batch_size, k_len, d_model).
            mask: Boolean mask broadcastable tới (batch_size, n_heads, q_len, k_len).

        Returns:
            Tensor shape (batch_size, q_len, d_model).
        """
        batch_size, q_len, _ = q.size()
        k_len = k.size(1)

        q_proj = self.w_q(q).view(batch_size, q_len, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, k_len, self.n_heads, self.d_k).transpose(1, 2)

        norm_mask = self._normalize_mask(mask)
        context, attn_weights = self.attention(q_proj, k_proj, v_proj, mask=norm_mask)
        if not self.training:
            self.last_attn_weights = attn_weights.detach()

        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.w_o(context)
