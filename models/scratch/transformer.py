"""
TransformerEncoder, TransformerDecoder & Seq2Seq Transformer (hỗ trợ Weight Tying).
Không sử dụng torch.nn.Transformer.
Owner: Hoàng Lâm Tùng (24021661)
"""
import torch
import torch.nn as nn


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pre_ln: bool = True,
        weight_tying: bool = True,
    ):
        super().__init__()

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
