"""
TransformerEncoder, TransformerDecoder & Seq2Seq Transformer (hỗ trợ Weight Tying).
Không sử dụng torch.nn.Transformer.
Owner: Hoàng Lâm Tùng (24021661)
"""

import math

import torch
import torch.nn as nn

from models.scratch.layers import DecoderLayer, EncoderLayer
from models.scratch.masks import create_padding_mask, create_target_mask
from models.scratch.positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    """Chồng N khối EncoderLayer thành bộ mã hóa hoàn chỉnh."""

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pre_ln: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    pre_ln=pre_ln,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model) if pre_ln else None

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Chồng N khối DecoderLayer thành bộ giải mã hoàn chỉnh."""

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pre_ln: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    pre_ln=pre_ln,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model) if pre_ln else None

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Seq2SeqTransformer(nn.Module):
    """Mô hình Encoder-Decoder Transformer cho tác vụ Text-to-Triples."""

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
        max_len: int = 512,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embed_scale = math.sqrt(d_model)
        self.weight_tying = weight_tying

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            pre_ln=pre_ln,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            pre_ln=pre_ln,
        )
        self.output_proj = nn.Linear(d_model, vocab_size, bias=not weight_tying)

        self._init_weights()

        if weight_tying:
            self.tgt_embedding.weight = self.src_embedding.weight
            self.output_proj.weight = self.tgt_embedding.weight

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.d_model**-0.5)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = create_padding_mask(src, pad_idx=self.pad_idx)
        src_emb = self.pos_encoding(self.src_embedding(src) * self.embed_scale)
        return self.encoder(src_emb, src_mask=src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_len = tgt.size(1)
        if tgt_mask is None:
            tgt_mask = create_target_mask(tgt, pad_idx=self.pad_idx)
        elif tgt_mask.dim() >= 3 and (tgt_mask.size(-2) > tgt_len or tgt_mask.size(-1) > tgt_len):
            # Hỗ trợ trường hợp dataloader tạo tgt_mask dài L nhưng decoder_input cắt tgt[:, :-1] dài L-1
            tgt_mask = tgt_mask[..., :tgt_len, :tgt_len]
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * self.embed_scale)
        return self.decoder(tgt_emb, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if src_mask is None:
            src_mask = create_padding_mask(src, pad_idx=self.pad_idx)
        if tgt_mask is None:
            tgt_mask = create_target_mask(tgt, pad_idx=self.pad_idx)

        enc_out = self.encode(src, src_mask=src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.output_proj(dec_out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
