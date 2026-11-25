"""
Triplet Parser: strip chat template / special tokens về chuẩn danh sách (Subject, Predicate, Object).
Owner: Bùi Thị Bích Phượng (24021601)
"""

import re
from typing import TypeAlias
Triple: TypeAlias = tuple[str, str, str]

def clean_text(text: str) -> str:
    """Xóa các ký tự đặc biệt, khoảng trắng thừa, và chuẩn hóa văn bản."""
    special_tokens = [
        "[BOS]", "[EOS]",  "[PAD]",
        "<bos>", "<eos>", "<pad>",
        "<|im_start|>", "<|im_end|>"
    ]

    for token in special_tokens:
        text = text.replace(token, "")

    return re.sub(r"\s+", " ", text).strip()

def parse_triplet(text: str) -> list[Triple]:
    """Chuyển đổi văn bản thành danh sách các triplet (Subject, Predicate, Object)."""
    if not isinstance(text, str) or not text.strip():
        return[]
    text = clean_text(text)
    triples=[]
    candidates = re.findall(r"\(([^()]*)\)", text)
    for candidate in candidates:
        parts = candidate.split("|")
        if len(parts) != 3:
            continue
        subject, predicate, obj = [part.strip() for part in parts]
        if not subject or not predicate or not obj:
            continue
        triples.append((subject, predicate, obj))
    return triples

def normalize_triple(triple: Triple) -> Triple:
    """Chuẩn hóa các thành phần của triplet."""
    return tuple(
        re.sub(r"\s+", " ", part).strip().casefold()
        for part in triple
    )

def deduplicate_triples(triples: list[Triple]) -> list[Triple]:
    """Loại bỏ các triplet trùng lặp."""
    result = []
    seen = set()
    for triple in triples:
        normalized = normalize_triple(triple)
        if normalized not in seen:
            seen.add(normalized)
            result.append(triple)
    return result
