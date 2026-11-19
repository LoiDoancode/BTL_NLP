"""
Huấn luyện Custom BPE Tokenizer CHỈ trên tập train split của WebNLG+ 2020.
Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy

Đặc tả:
- Thuật toán: Byte-Level Byte-Pair Encoding (BPE)
- Thư viện: Hugging Face `tokenizers`
- Special Tokens:
    [PAD] = 0
    [UNK] = 1
    [BOS] = 2
    [EOS] = 3
    [SEP] = 4
- Vocab size mặc định: 16,000 (theo configs/config.yaml)
- Huấn luyện CHỈ trên tập train (tránh data leakage)
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4


class BPETokenizerWrapper:
    """Wrapper tiện dụng cho Custom BPE Tokenizer phục vụ PyTorch Dataset và Seq2Seq Model."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = (
            self.tokenizer.token_to_id(PAD_TOKEN)
            if self.tokenizer.token_to_id(PAD_TOKEN) is not None
            else PAD_ID
        )
        self.unk_token_id = (
            self.tokenizer.token_to_id(UNK_TOKEN)
            if self.tokenizer.token_to_id(UNK_TOKEN) is not None
            else UNK_ID
        )
        self.bos_token_id = (
            self.tokenizer.token_to_id(BOS_TOKEN)
            if self.tokenizer.token_to_id(BOS_TOKEN) is not None
            else BOS_ID
        )
        self.eos_token_id = (
            self.tokenizer.token_to_id(EOS_TOKEN)
            if self.tokenizer.token_to_id(EOS_TOKEN) is not None
            else EOS_ID
        )
        self.sep_token_id = (
            self.tokenizer.token_to_id(SEP_TOKEN)
            if self.tokenizer.token_to_id(SEP_TOKEN) is not None
            else SEP_ID
        )

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        return self.vocab_size

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = True,
    ) -> dict[str, Any]:
        """
        Mã hóa chuỗi thành input_ids và attention_mask.
        Nếu add_special_tokens=True: thêm [BOS] ở đầu và [EOS] ở cuối.
        """
        enc = self.tokenizer.encode(text)
        ids = enc.ids

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if truncation and max_length is not None and len(ids) > max_length:
            if add_special_tokens:
                ids = ids[: max_length - 1] + [self.eos_token_id]
            else:
                ids = ids[:max_length]

        attention_mask = [1] * len(ids)

        if padding and max_length is not None and len(ids) < max_length:
            pad_len = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Giải mã danh sách token IDs thành chuỗi văn bản."""
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()

        if skip_special_tokens:
            specials = {
                self.pad_token_id,
                self.unk_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.sep_token_id,
            }
            ids = [i for i in ids if i not in specials]

        return self.tokenizer.decode(ids)

    def save(self, save_dir: str | Path) -> None:
        """Lưu tokenizer và cấu hình vào thư mục."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer_file = save_path / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_file))

        vocab = self.tokenizer.get_vocab()
        with open(save_path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        meta = {
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad": {"token": PAD_TOKEN, "id": self.pad_token_id},
                "unk": {"token": UNK_TOKEN, "id": self.unk_token_id},
                "bos": {"token": BOS_TOKEN, "id": self.bos_token_id},
                "eos": {"token": EOS_TOKEN, "id": self.eos_token_id},
                "sep": {"token": SEP_TOKEN, "id": self.sep_token_id},
            },
        }
        with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"[SAVED] Tokenizer đã lưu tại: {save_path} (vocab_size={self.vocab_size:,})")

    @classmethod
    def load(cls, save_dir: str | Path) -> "BPETokenizerWrapper":
        """Nạp tokenizer từ thư mục lưu trữ."""
        save_path = Path(save_dir)
        tokenizer_file = save_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Không tìm thấy file tokenizer tại {tokenizer_file}")

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        return cls(tokenizer)


def text_iterator(train_file_path: str | Path) -> Iterator[str]:
    """Đọc dữ liệu train và trả về từng chuỗi văn bản (input text và target triples) để huấn luyện."""
    with open(train_file_path, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        text = item.get("text", "")
        target = item.get("target_text", "")
        if text:
            yield text
        if target:
            yield target


def train_bpe_tokenizer(
    train_file: str | Path = "data/processed/train.json",
    vocab_size: int = 16000,
    save_dir: str | Path = "tokenizer/artifacts",
    min_frequency: int = 2,
) -> BPETokenizerWrapper:
    """
    Huấn luyện Byte-Level BPE Tokenizer CHỈ trên tập train split.
    """
    train_path = Path(train_file)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file {train_path}. Hãy chạy `python data/prepare_webnlg.py` trước!"
        )

    print("=" * 70)
    print("BTL NLP: HUẤN LUYỆN CUSTOM BPE TOKENIZER (CHỈ TRÊN TRAIN SPLIT)")
    print(f"Vocab Size mục tiêu: {vocab_size:,} | Min Frequency: {min_frequency}")
    print("Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy")
    print("=" * 70)

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print("[TRAINING] Đang huấn luyện BPE Tokenizer...")
    tokenizer.train_from_iterator(text_iterator(train_path), trainer=trainer)
    print(f"[TRAINED] Huấn luyện hoàn tất! Vocab size thực tế: {tokenizer.get_vocab_size():,}")

    wrapper = BPETokenizerWrapper(tokenizer)
    wrapper.save(save_dir)

    sample_text = "The leader of Aarhus is Jacob Bundsgaard."
    sample_target = "(Aarhus | leader | Jacob_Bundsgaard)"

    enc_text = wrapper.encode(sample_text)
    dec_text = wrapper.decode(enc_text["input_ids"])
    enc_target = wrapper.encode(sample_target)
    dec_target = wrapper.decode(enc_target["input_ids"])

    print("\n--- KIỂM CHỨNG HOẠT ĐỘNG TOKENIZER ---")
    print(f"Gốc Text:    {sample_text}")
    print(f"Mã hóa Text: {enc_text['input_ids'][:12]}... (len={len(enc_text['input_ids'])})")
    print(f"Giải mã:     {dec_text}")
    assert sample_text == dec_text, "Lỗi giải mã text không khớp gốc!"

    print(f"\nGốc Target:  {sample_target}")
    print(f"Mã hóa Tgt:  {enc_target['input_ids'][:12]}... (len={len(enc_target['input_ids'])})")
    print(f"Giải mã:     {dec_target}")
    assert sample_target == dec_target, "Lỗi giải mã target không khớp gốc!"
    print("--> Kiểm chứng thành công 100%!")

    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện Custom BPE Tokenizer trên WebNLG+ 2020 Train Split."
    )
    parser.add_argument("--train_file", type=str, default="data/processed/train.json")
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--save_dir", type=str, default="tokenizer/artifacts")
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()

    train_bpe_tokenizer(
        train_file=args.train_file,
        vocab_size=args.vocab_size,
        save_dir=args.save_dir,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
