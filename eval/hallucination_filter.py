"""
Entity Hallucination Filter: lọc bỏ các bộ ba bịa đặt thực thể không có cơ sở từ văn bản đầu vào.
Owner: Bùi Thị Bích Phượng (24021601)
"""

import re
from typing import Any

from .parser import Triple


def normalize_entity(text: str) -> str:
    """Chuẩn hóa thực thể (đổi '_' thành khoảng trắng, gộp khoảng trắng, casefold)."""
    return re.sub(r"\s+", " ", text.replace("_", " ").casefold()).strip()


def entity_exists(entity: str, source_text: str) -> bool:
    """
    Kiểm tra thực thể có xuất hiện trong văn bản nguồn không (dùng word boundary).
    """
    norm_entity = normalize_entity(entity)
    norm_source = normalize_entity(source_text)
    if not norm_entity:
        return False
    pattern = r"(?<!\w)" + re.escape(norm_entity) + r"(?!\w)"
    return re.search(pattern, norm_source) is not None


def filter_hallucinated_triples(
    triples: list[Triple],
    source_text: str,
) -> dict[str, list[Any]]:
    """
    Loại bỏ các bộ ba có subject hoặc object không thể đối chiếu trong văn bản đầu vào.
    """
    valid: list[Triple] = []
    rejected: list[dict[str, Any]] = []
    for triple in triples:
        subject, _predicate, obj = triple
        subject_found = entity_exists(subject, source_text)
        object_found = entity_exists(obj, source_text)
        if subject_found and object_found:
            valid.append(triple)
        else:
            rejected.append(
                {
                    "triple": triple,
                    "subject_found": subject_found,
                    "object_found": object_found,
                }
            )
    return {
        "valid": valid,
        "rejected": rejected,
    }
