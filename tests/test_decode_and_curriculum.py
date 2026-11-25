"""
Unit tests cho module giải mã tự hồi quy (Greedy Search & Beam Search B=4)
và chiến lược Curriculum Learning (Tuần 2 - Doãn Duy Lợi).
"""

import torch

from decode import beam_search_decode, greedy_decode
from models.scratch.transformer import Seq2SeqTransformer
from tokenizer.train_bpe import BOS_ID, EOS_ID, PAD_ID
from train_scratch import build_curriculum_schedule, get_curriculum_stage


def _build_tiny_transformer(vocab_size: int = 64) -> Seq2SeqTransformer:
    torch.manual_seed(42)
    return Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=64,
        dropout=0.0,
        pre_ln=True,
        weight_tying=True,
        pad_idx=PAD_ID,
    )


def test_greedy_decode_shape_and_bos() -> None:
    model = _build_tiny_transformer(vocab_size=50)
    src = torch.tensor([[BOS_ID, 10, 12, EOS_ID, PAD_ID], [BOS_ID, 7, 8, 9, EOS_ID]])
    out = greedy_decode(model, src, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, max_len=12)

    assert out.shape[0] == 2
    assert 2 <= out.shape[1] <= 12
    assert torch.all(out[:, 0] == BOS_ID)


def test_beam_search_decode_b4_and_b1_consistency() -> None:
    model = _build_tiny_transformer(vocab_size=50)
    src = torch.tensor([[BOS_ID, 5, 6, 7, EOS_ID], [BOS_ID, 11, 12, EOS_ID, PAD_ID]])

    greedy_out = greedy_decode(model, src, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, max_len=10)
    beam1_out = beam_search_decode(
        model, src, beam_size=1, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, max_len=10
    )
    beam4_out = beam_search_decode(
        model, src, beam_size=4, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, max_len=10
    )

    assert torch.equal(greedy_out, beam1_out)
    assert beam4_out.shape[0] == 2
    assert 2 <= beam4_out.shape[1] <= 10
    assert torch.all(beam4_out[:, 0] == BOS_ID)


def test_curriculum_schedule_3_stages() -> None:
    schedule = build_curriculum_schedule(total_epochs=12, stage1_ratio=0.25, stage2_ratio=0.25)
    assert len(schedule) == 3

    assert get_curriculum_stage(1, schedule).name == "stage_1_easy"
    assert get_curriculum_stage(3, schedule).num_triples_filter == 1

    assert get_curriculum_stage(4, schedule).name == "stage_2_medium"
    assert get_curriculum_stage(6, schedule).num_triples_filter == [1, 2]

    assert get_curriculum_stage(7, schedule).name == "stage_3_full"
    assert get_curriculum_stage(12, schedule).num_triples_filter is None


def test_curriculum_schedule_short_runs() -> None:
    s1 = build_curriculum_schedule(total_epochs=1)
    assert len(s1) == 1 and s1[0].num_triples_filter is None

    s2 = build_curriculum_schedule(total_epochs=2)
    assert len(s2) == 2
    assert get_curriculum_stage(1, s2).num_triples_filter == 1
    assert get_curriculum_stage(2, s2).num_triples_filter is None
