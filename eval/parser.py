"""
Triplet Parser: strip chat template / special tokens về chuẩn danh sách (Subject, Predicate, Object).
Chuẩn hóa thống nhất: đổi '_' thành khoảng trắng ở Subject & Object, giữ nguyên Predicate.
Owner: Bùi Thị Bích Phượng (24021601)
"""

import re
from typing import TypeAlias

Triple: TypeAlias = tuple[str, str, str]

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|im_end\|>|<\|im_start\|>(?:system|user|assistant)?|<\|endoftext\|>"
    r"|<\|eot_id\|>|<\|begin_of_text\|>"
    r"|<\|start_header_id\|>(?:system|user|assistant)?<\|end_header_id\|>"
    r"|<\|end\|>|<\|assistant\|>|<\|user\|>|<\|system\|>"
    r"|\[BOS\]|\[EOS\]|\[PAD\]|<bos>|<eos>|<pad>|<s>|</s>",
    flags=re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Xóa khối <think>, các special token của LLM/Scratch và chuẩn hóa khoảng trắng."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _SPECIAL_TOKEN_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_balanced_parens(text: str) -> list[str]:
    """Trích xuất các cụm ngoài cùng nằm trong cặp ngoặc đơn (...) kể cả khi có ngoặc lồng nhau."""
    results: list[str] = []
    depth = 0
    start_idx = -1
    for idx, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start_idx = idx + 1
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx != -1:
                results.append(text[start_idx:idx])
                start_idx = -1
    return results


def parse_triplet(text: str) -> list[Triple]:
    """Chuyển đổi văn bản thành danh sách các triplet (Subject, Predicate, Object)."""
    if not isinstance(text, str) or not text.strip():
        return []
    text = clean_text(text)
    triples: list[Triple] = []
    for candidate in _extract_balanced_parens(text):
        parts = candidate.split("|")
        if len(parts) != 3:
            continue
        subject, predicate, obj = (part.strip() for part in parts)
        if not subject or not predicate or not obj:
            continue
        triples.append((subject, predicate, obj))
    return triples


def normalize_triple(triple: Triple) -> Triple:
    """
    Chuẩn hóa các thành phần của triplet để so khớp công bằng giữa Scratch và LLMs:
    - Subject & Object: đổi '_' thành khoảng trắng, gộp khoảng trắng, chuyển thường (casefold).
    - Predicate: gộp khoảng trắng, chuyển thường (casefold).
    """
    subject, predicate, obj = triple
    norm_s = re.sub(r"\s+", " ", subject.replace("_", " ")).strip().casefold()
    norm_p = re.sub(r"\s+", " ", predicate).strip().casefold()
    norm_o = re.sub(r"\s+", " ", obj.replace("_", " ")).strip().casefold()
    return (norm_s, norm_p, norm_o)


def deduplicate_triples(triples: list[Triple]) -> list[Triple]:
    """Loại bỏ các triplet trùng lặp theo dạng chuẩn hóa."""
    result: list[Triple] = []
    seen: set[Triple] = set()
    for triple in triples:
        normalized = normalize_triple(triple)
        if normalized not in seen:
            seen.add(normalized)
            result.append(triple)
    return result
