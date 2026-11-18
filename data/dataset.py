"""
PyTorch Dataset & DataLoader cho Text-to-Triples (hỗ trợ Curriculum Learning: 1 -> 2 -> >=3 triples).
Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy
"""

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer.train_bpe import PAD_ID, BPETokenizerWrapper


class WebNLGDataset(Dataset):
    """Dataset wrapper cho WebNLG+ 2020 đã chuẩn hóa (Text-to-Triples)."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: BPETokenizerWrapper | None = None,
        max_length: int = 256,
        num_triples_filter: int | list[int] | str | None = None,
    ):
        """
        Args:
            data_path: Đường dẫn file JSON đã tiền xử lý.
            tokenizer: BPETokenizerWrapper instance.
            max_length: Độ dài chuỗi tối đa sau khi tokenize.
            num_triples_filter: Lọc mẫu theo số bộ ba phục vụ Curriculum Learning
                                (ví dụ: 1, 2, '>=3', hoặc [1, 2]).
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_triples_filter = num_triples_filter

        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")

        with open(self.data_path, encoding="utf-8") as f:
            raw_samples = json.load(f)

        # Lọc mẫu theo Curriculum Learning (nếu được yêu cầu)
        self.samples = self._filter_samples(raw_samples, num_triples_filter)

    def _filter_samples(
        self, samples: list[dict[str, Any]], filter_rule: int | list[int] | str | None
    ) -> list[dict[str, Any]]:
        if filter_rule is None:
            return samples

        filtered = []
        for s in samples:
            n = s.get("num_triples", len(s.get("triples", [])))
            if isinstance(filter_rule, int) and n == filter_rule:
                filtered.append(s)
            elif isinstance(filter_rule, (list, tuple)) and n in filter_rule:
                filtered.append(s)
            elif isinstance(filter_rule, str):
                if filter_rule == ">=3" and n >= 3:
                    filtered.append(s)
                elif filter_rule.isdigit() and n == int(filter_rule):
                    filtered.append(s)
        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        src_text = item["text"]
        tgt_text = item["target_text"]

        output = {
            "id": item.get("id", str(idx)),
            "webnlg_id": item.get("webnlg_id", ""),
            "category": item.get("category", ""),
            "src_text": src_text,
            "tgt_text": tgt_text,
            "triples": item.get("triples", []),
            "num_triples": item.get("num_triples", len(item.get("triples", []))),
        }

        # Nếu có tokenizer, thực hiện encode luôn
        if self.tokenizer is not None:
            src_enc = self.tokenizer.encode(
                src_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding=False,
                truncation=True,
            )
            tgt_enc = self.tokenizer.encode(
                tgt_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding=False,
                truncation=True,
            )
            output["src_ids"] = src_enc["input_ids"]
            output["tgt_ids"] = tgt_enc["input_ids"]

        return output


def collate_fn(batch: list[dict[str, Any]], pad_id: int = PAD_ID) -> dict[str, Any]:
    """
    Hàm gom batch và đệm động (dynamic padding) cho Seq2Seq Transformer.
    Tự động tạo src_mask (Encoder padding mask) và tgt_mask (Decoder causal & padding mask).
    """
    ids_list = [item["id"] for item in batch]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]
    triples_list = [item["triples"] for item in batch]
    categories = [item["category"] for item in batch]
    num_triples = [item["num_triples"] for item in batch]

    res = {
        "ids": ids_list,
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "triples": triples_list,
        "categories": categories,
        "num_triples": num_triples,
    }

    if "src_ids" in batch[0] and "tgt_ids" in batch[0]:
        max_src_len = max(len(item["src_ids"]) for item in batch)
        max_tgt_len = max(len(item["tgt_ids"]) for item in batch)

        padded_src = []
        padded_tgt = []

        for item in batch:
            s_ids = item["src_ids"]
            t_ids = item["tgt_ids"]
            padded_src.append(s_ids + [pad_id] * (max_src_len - len(s_ids)))
            padded_tgt.append(t_ids + [pad_id] * (max_tgt_len - len(t_ids)))

        src_tensor = torch.tensor(padded_src, dtype=torch.long)
        tgt_tensor = torch.tensor(padded_tgt, dtype=torch.long)

        # src_mask: [B, 1, 1, L_src] (True cho valid tokens, False cho pad)
        src_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2)

        # Decoder padding mask: [B, 1, 1, L_tgt]
        tgt_pad_mask = (tgt_tensor != pad_id).unsqueeze(1).unsqueeze(2)

        # Causal mask: [1, 1, L_tgt, L_tgt]
        causal_mask = (
            torch.tril(torch.ones((max_tgt_len, max_tgt_len), dtype=torch.bool))
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Combined tgt_mask: [B, 1, L_tgt, L_tgt]
        tgt_mask = tgt_pad_mask & causal_mask

        res["src_ids"] = src_tensor
        res["tgt_ids"] = tgt_tensor
        res["src_mask"] = src_mask
        res["tgt_mask"] = tgt_mask

    return res


def get_dataloader(
    data_path: str | Path,
    tokenizer: BPETokenizerWrapper | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = 256,
    num_triples_filter: int | list[int] | str | None = None,
    num_workers: int = 0,
    pad_id: int = PAD_ID,
) -> DataLoader:
    """Tạo DataLoader tiện dụng cho huấn luyện và kiểm thử."""
    dataset = WebNLGDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        num_triples_filter=num_triples_filter,
    )

    def collate(b: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_fn(b, pad_id=pad_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
    )
