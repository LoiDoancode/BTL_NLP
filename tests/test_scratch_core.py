"""
Unit tests cơ bản cho Transformer From-Scratch Core.
Owner: Hoàng Lâm Tùng (24021661)
"""

import pytest
import torch

from models.scratch import (
    DecoderLayer,
    EncoderLayer,
    FeedForwardNetwork,
    MultiHeadAttention,
    PositionalEncoding,
    ScaledDotProductAttention,
    Seq2SeqTransformer,
    TransformerDecoder,
    TransformerEncoder,
    create_causal_mask,
    create_padding_mask,
    create_target_mask,
)


def test_scaled_dot_product_attention_shape_and_mask() -> None:
    torch.manual_seed(42)
    attn = ScaledDotProductAttention(dropout=0.0)
    q = torch.randn(2, 4, 5, 16)
    k = torch.randn(2, 4, 6, 16)
    v = torch.randn(2, 4, 6, 16)

    mask = torch.ones(2, 1, 1, 6, dtype=torch.bool)
    mask[:, :, :, -2:] = False

    out, weights = attn(q, k, v, mask=mask)
    assert out.shape == (2, 4, 5, 16)
    assert weights.shape == (2, 4, 5, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)
    assert torch.all(weights[:, :, :, -2:] == 0.0)


def test_scaled_dot_product_attention_all_masked_no_nan() -> None:
    attn = ScaledDotProductAttention(dropout=0.0)
    q = torch.randn(1, 2, 3, 8, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, requires_grad=True)
    v = torch.randn(1, 2, 4, 8, requires_grad=True)

    mask = torch.zeros(1, 1, 3, 4, dtype=torch.bool)
    out, weights = attn(q, k, v, mask=mask)
    assert not torch.isnan(out).any()
    assert torch.all(weights == 0.0)

    out.sum().backward()
    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()


def test_multi_head_attention_self_and_cross() -> None:
    torch.manual_seed(42)
    mha = MultiHeadAttention(d_model=512, n_heads=8, dropout=0.0)

    x = torch.randn(2, 10, 512, requires_grad=True)
    self_out = mha(x, x, x)
    assert self_out.shape == (2, 10, 512)

    mem = torch.randn(2, 14, 512)
    cross_out = mha(x, mem, mem)
    assert cross_out.shape == (2, 10, 512)

    (self_out.sum() + cross_out.sum()).backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()


def test_multi_head_attention_invalid_heads() -> None:
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=512, n_heads=7)


def test_masks_and_causal_invariance() -> None:
    seq = torch.tensor([[5, 7, 9, 0, 0], [3, 4, 0, 0, 0]])
    pad_mask = create_padding_mask(seq, pad_idx=0)
    assert pad_mask.shape == (2, 1, 1, 5)
    assert pad_mask.dtype == torch.bool
    assert torch.equal(
        pad_mask.squeeze(1).squeeze(1),
        torch.tensor([[True, True, True, False, False], [True, True, False, False, False]]),
    )

    causal_mask = create_causal_mask(4)
    assert causal_mask.shape == (1, 1, 4, 4)
    assert torch.equal(
        causal_mask.squeeze(0).squeeze(0),
        torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ),
    )

    tgt_mask = create_target_mask(seq, pad_idx=0)
    assert tgt_mask.shape == (2, 1, 5, 5)

    # Thay đổi token ở tương lai không được ảnh hưởng tới output tại các vị trí trước đó
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    mha.eval()
    x1 = torch.randn(1, 5, 64)
    x2 = x1.clone()
    x2[:, 3:, :] = torch.randn(1, 2, 64)

    cmask = create_causal_mask(5)
    y1 = mha(x1, x1, x1, mask=cmask)
    y2 = mha(x2, x2, x2, mask=cmask)
    assert torch.allclose(y1[:, :3, :], y2[:, :3, :], atol=1e-6)


def test_padding_mask_blocks_pad_positions() -> None:
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    mha.eval()

    seq = torch.tensor([[10, 20, 0, 0]])
    pad_mask = create_padding_mask(seq, pad_idx=0)

    x1 = torch.randn(1, 4, 64)
    x2 = x1.clone()
    x2[:, 2:, :] = torch.randn(1, 2, 64)

    y1 = mha(x1, x1, x1, mask=pad_mask)
    y2 = mha(x2, x2, x2, mask=pad_mask)
    assert torch.allclose(y1[:, :2, :], y2[:, :2, :], atol=1e-6)
    assert mha.last_attn_weights is not None
    assert torch.all(mha.last_attn_weights[:, :, :, 2:] == 0.0)


def test_positional_encoding() -> None:
    pe = PositionalEncoding(d_model=512, max_len=128, dropout=0.0)
    x = torch.zeros(2, 16, 512)
    out = pe(x)
    assert out.shape == (2, 16, 512)
    # Tại vị trí pos=0: sin(0) = 0 ở các chiều chẵn, cos(0) = 1 ở các chiều lẻ
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(256), atol=1e-6)
    assert torch.allclose(out[0, 0, 1::2], torch.ones(256), atol=1e-6)

    with pytest.raises(ValueError):
        pe(torch.zeros(1, 129, 512))


def test_feed_forward_network() -> None:
    ffn = FeedForwardNetwork(d_model=512, d_ff=2048, dropout=0.1)
    x = torch.randn(2, 12, 512, requires_grad=True)
    out = ffn(x)
    assert out.shape == (2, 12, 512)
    out.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()


@pytest.mark.parametrize("pre_ln", [False, True])
def test_encoder_and_decoder_layers(pre_ln: bool) -> None:
    enc_layer = EncoderLayer(d_model=64, n_heads=4, d_ff=256, dropout=0.0, pre_ln=pre_ln)
    dec_layer = DecoderLayer(d_model=64, n_heads=4, d_ff=256, dropout=0.0, pre_ln=pre_ln)

    src = torch.randn(2, 8, 64, requires_grad=True)
    tgt = torch.randn(2, 6, 64, requires_grad=True)
    src_mask = torch.ones(2, 1, 1, 8, dtype=torch.bool)
    tgt_mask = create_causal_mask(6)

    enc_out = enc_layer(src, src_mask=src_mask)
    dec_out = dec_layer(tgt, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
    assert enc_out.shape == (2, 8, 64)
    assert dec_out.shape == (2, 6, 64)

    dec_out.sum().backward()
    assert src.grad is not None and not torch.isnan(src.grad).any()
    assert tgt.grad is not None and not torch.isnan(tgt.grad).any()


def test_encoder_and_decoder_stacks() -> None:
    encoder = TransformerEncoder(num_layers=2, d_model=64, n_heads=4, d_ff=256, dropout=0.0)
    decoder = TransformerDecoder(num_layers=2, d_model=64, n_heads=4, d_ff=256, dropout=0.0)

    src = torch.randn(2, 7, 64)
    tgt = torch.randn(2, 5, 64)
    mem = encoder(src)
    out = decoder(tgt, mem, tgt_mask=create_causal_mask(5))
    assert mem.shape == (2, 7, 64)
    assert out.shape == (2, 5, 64)


@pytest.mark.parametrize("pre_ln", [False, True])
def test_seq2seq_transformer_forward_and_causal_invariance(pre_ln: bool) -> None:
    torch.manual_seed(42)
    model = Seq2SeqTransformer(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        dropout=0.0,
        pre_ln=pre_ln,
        weight_tying=False,
        max_len=64,
        pad_idx=0,
    )
    model.eval()

    src = torch.tensor([[4, 9, 12, 0, 0], [8, 3, 5, 6, 0]])
    tgt1 = torch.tensor([[1, 15, 22, 33, 0], [1, 18, 25, 0, 0]])
    tgt2 = tgt1.clone()
    tgt2[:, 3:] = 77

    logits1 = model(src, tgt1)
    logits2 = model(src, tgt2)
    assert logits1.shape == (2, 5, 128)
    # Thay đổi token đích từ vị trí 3 trở đi không làm thay đổi logits tại 0..2
    assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5)

    # Kiểm tra encode/decode tách rời cho autoregressive decoding
    src_mask = create_padding_mask(src, pad_idx=0)
    memory = model.encode(src, src_mask=src_mask)
    dec_out = model.decode(tgt1, memory, src_mask=src_mask)
    assert torch.allclose(model.output_proj(dec_out), logits1, atol=1e-5)


def test_seq2seq_weight_tying_reduces_parameters_and_trains() -> None:
    untied = Seq2SeqTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        weight_tying=False,
    )
    tied = Seq2SeqTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=256,
        weight_tying=True,
    )
    assert tied.count_parameters() < untied.count_parameters()
    assert tied.src_embedding.weight is tied.tgt_embedding.weight
    assert tied.output_proj.weight is tied.tgt_embedding.weight

    src = torch.randint(1, 1000, (2, 10))
    tgt = torch.randint(1, 1000, (2, 8))
    logits = tied(src, tgt)
    loss = logits.sum()
    loss.backward()
    assert tied.src_embedding.weight.grad is not None
    assert not torch.isnan(tied.src_embedding.weight.grad).any()


def test_seq2seq_integration_with_train_scratch_teacher_forcing() -> None:
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from data.dataset import collate_fn
    from train_scratch import create_model, evaluate, train_one_epoch

    cfg = {
        "scratch": {
            "d_model": 64,
            "n_heads": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "d_ff": 256,
            "dropout": 0.0,
            "use_pre_ln": True,
            "weight_tying": True,
        }
    }
    device = torch.device("cpu")
    model, desc = create_model(cfg, vocab_size=100, device=device)
    assert isinstance(model, Seq2SeqTransformer)
    assert "Seq2SeqTransformer" in desc

    samples = [
        {
            "id": "0",
            "src_text": "a",
            "tgt_text": "b",
            "triples": [],
            "category": "test",
            "num_triples": 1,
            "src_ids": [2, 11, 12, 3],
            "tgt_ids": [2, 21, 22, 23, 3],
        },
        {
            "id": "1",
            "src_text": "c",
            "tgt_text": "d",
            "triples": [],
            "category": "test",
            "num_triples": 1,
            "src_ids": [2, 13, 3],
            "tgt_ids": [2, 24, 3],
        },
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=lambda b: collate_fn(b, pad_id=0))
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    train_loss, steps = train_one_epoch(model, loader, optimizer, criterion, None, device, pad_id=0)
    val_loss = evaluate(model, loader, criterion, device, pad_id=0)
    assert steps == 1
    assert train_loss > 0.0
    assert val_loss > 0.0
