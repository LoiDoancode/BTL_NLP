"""
Unit tests cho Data Pipeline, Custom BPE Tokenizer & Training Loop.
Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy
"""

import json
from pathlib import Path

import pytest
import torch

from data.dataset import WebNLGDataset, get_dataloader
from data.prepare_webnlg import clean_str, format_triples_to_target, parse_raw_triple
from tokenizer.train_bpe import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    UNK_ID,
    BPETokenizerWrapper,
)
from train_scratch import NoamLRScheduler, SimpleSeq2SeqBaseline

HAS_PROCESSED_DATA = (
    Path("data/processed/train.json").exists()
    and Path("data/processed/val.json").exists()
    and Path("data/processed/test.json").exists()
)
HAS_TOKENIZER_ARTIFACT = Path("tokenizer/artifacts/tokenizer.json").exists()


def test_prepare_webnlg_helpers():
    """Kiểm tra các hàm làm sạch chuỗi và định dạng bộ ba (Subject | Predicate | Object)."""
    assert clean_str('  "Aarhus, Denmark"  ') == "Aarhus, Denmark"
    assert parse_raw_triple('Aarhus_Airport | cityServed | "Aarhus, Denmark"') == (
        "Aarhus_Airport",
        "cityServed",
        "Aarhus, Denmark",
    )
    triples = [
        ("Alan_Bean", "mission", "Apollo_12"),
        ("Apollo_12", "operator", "NASA"),
    ]
    assert (
        format_triples_to_target(triples)
        == "(Alan_Bean | mission | Apollo_12) | (Apollo_12 | operator | NASA)"
    )


@pytest.mark.skipif(not HAS_TOKENIZER_ARTIFACT, reason="Tokenizer artifacts not found.")
def test_tokenizer_special_tokens():
    """Kiểm tra tokenizer đã train và các special tokens chuẩn hóa."""
    tokenizer = BPETokenizerWrapper.load("tokenizer/artifacts")
    assert tokenizer.pad_token_id == PAD_ID == 0
    assert tokenizer.unk_token_id == UNK_ID == 1
    assert tokenizer.bos_token_id == BOS_ID == 2
    assert tokenizer.eos_token_id == EOS_ID == 3
    assert tokenizer.sep_token_id == SEP_ID == 4
    assert tokenizer.vocab_size > 1000


@pytest.mark.skipif(not HAS_TOKENIZER_ARTIFACT, reason="Tokenizer artifacts not found.")
def test_tokenizer_encode_decode():
    """Kiểm tra tính chính xác của thuật toán Byte-Level BPE khi mã hóa và giải mã."""
    tokenizer = BPETokenizerWrapper.load("tokenizer/artifacts")
    text = "The leader of Aarhus is Jacob Bundsgaard."
    target = "(Aarhus | leader | Jacob_Bundsgaard)"

    enc_text = tokenizer.encode(text, add_special_tokens=True)
    assert enc_text["input_ids"][0] == BOS_ID
    assert enc_text["input_ids"][-1] == EOS_ID
    assert tokenizer.decode(enc_text["input_ids"]) == text

    enc_tgt = tokenizer.encode(target, add_special_tokens=True)
    assert enc_tgt["input_ids"][0] == BOS_ID
    assert enc_tgt["input_ids"][-1] == EOS_ID
    assert tokenizer.decode(enc_tgt["input_ids"]) == target


@pytest.mark.skipif(not HAS_TOKENIZER_ARTIFACT, reason="Tokenizer artifacts not found.")
def test_dataset_curriculum_and_collate_synthetic(tmp_path: Path):
    """Kiểm tra WebNLGDataset, Curriculum filter và dynamic padding (chạy được cả trên CI)."""
    tokenizer = BPETokenizerWrapper.load("tokenizer/artifacts")
    synthetic_samples = [
        {
            "id": "s_1",
            "webnlg_id": "test/1",
            "category": "Airport",
            "text": "The leader of Aarhus is Jacob Bundsgaard.",
            "triples": [["Aarhus", "leader", "Jacob_Bundsgaard"]],
            "target_text": "(Aarhus | leader | Jacob_Bundsgaard)",
            "num_triples": 1,
        },
        {
            "id": "s_2",
            "webnlg_id": "test/2",
            "category": "Astronaut",
            "text": "Alan Bean flew on Apollo 12 operated by NASA.",
            "triples": [
                ["Alan_Bean", "mission", "Apollo_12"],
                ["Apollo_12", "operator", "NASA"],
            ],
            "target_text": "(Alan_Bean | mission | Apollo_12) | (Apollo_12 | operator | NASA)",
            "num_triples": 2,
        },
        {
            "id": "s_3",
            "webnlg_id": "test/3",
            "category": "SportsTeam",
            "text": "A sample with three triples.",
            "triples": [["A", "p1", "B"], ["B", "p2", "C"], ["C", "p3", "D"]],
            "target_text": "(A | p1 | B) | (B | p2 | C) | (C | p3 | D)",
            "num_triples": 3,
        },
    ]
    sample_file = tmp_path / "sample.json"
    sample_file.write_text(json.dumps(synthetic_samples), encoding="utf-8")

    ds_all = WebNLGDataset(sample_file, tokenizer=tokenizer)
    ds_1 = WebNLGDataset(sample_file, tokenizer=tokenizer, num_triples_filter=1)
    ds_2 = WebNLGDataset(sample_file, tokenizer=tokenizer, num_triples_filter=2)
    ds_3 = WebNLGDataset(sample_file, tokenizer=tokenizer, num_triples_filter=">=3")

    assert len(ds_all) == 3
    assert len(ds_1) == 1
    assert len(ds_2) == 1
    assert len(ds_3) == 1

    loader = get_dataloader(sample_file, tokenizer=tokenizer, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    src_ids = batch["src_ids"]
    tgt_ids = batch["tgt_ids"]
    src_mask = batch["src_mask"]
    tgt_mask = batch["tgt_mask"]

    assert src_ids.shape[0] == 3
    assert src_mask.shape == (3, 1, 1, src_ids.shape[1])
    assert tgt_mask.shape == (3, 1, tgt_ids.shape[1], tgt_ids.shape[1])
    assert src_mask.dtype == torch.bool
    assert tgt_mask.dtype == torch.bool


@pytest.mark.skipif(not HAS_PROCESSED_DATA, reason="Local processed dataset not generated yet.")
def test_processed_data_files_and_counts():
    """Kiểm tra số lượng mẫu chuẩn GEM/web_nlg khi chạy cục bộ."""
    expected_counts = {"train": 35426, "val": 1667, "test": 1779}
    base_dir = Path("data/processed")
    for split, expected in expected_counts.items():
        json_file = base_dir / f"{split}.json"
        csv_file = base_dir / f"{split}.csv"
        assert json_file.exists() and csv_file.exists()
        with open(json_file, encoding="utf-8") as f:
            samples = json.load(f)
        assert len(samples) == expected
        sample = samples[0]
        assert sample["target_text"].startswith("(")
        assert " | " in sample["target_text"]


def test_noam_scheduler_and_simple_baseline_forward():
    """Kiểm tra Noam Warmup Scheduler và mô hình baseline forward pass."""
    vocab_size = 1000
    model = SimpleSeq2SeqBaseline(vocab_size=vocab_size, d_model=64, n_heads=2, num_layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = NoamLRScheduler(optimizer, d_model=64, warmup_steps=100)

    scheduler.step()
    lr1 = scheduler.get_last_lr()[0]
    scheduler.step()
    lr2 = scheduler.get_last_lr()[0]
    assert lr2 > lr1 > 0

    src = torch.randint(0, vocab_size, (2, 8))
    tgt = torch.randint(0, vocab_size, (2, 10))
    logits = model(src, tgt)
    assert logits.shape == (2, 10, vocab_size)
