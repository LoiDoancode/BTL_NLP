"""
Entity Hallucination Filter: lọc bỏ các bộ ba bịa đặt thực thể không có cơ sở từ văn bản đầu vào.
Owner: Bùi Thị Bích Phượng (24021601)
"""


import re

if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from parser import Triple
else:
    from .parser import Triple

def normalize_entity(text: str) -> str:
    """
    Normalize an entity for matching
    """
    return re.sub(
        r"\s+",
        " ",
        text.casefold()
    ).strip()

def entity_exists(entity: str, source_text: str) -> bool:
    """
    Check whether an entity occurs in source text.
    Use word boundaries to avoid partial-word matches.
    """
    entity = normalize_entity(entity)
    source = normalize_entity(source_text)
    if not entity:
        return False
    pattern = (
        r"(?<!\w)"
        + re.escape(entity)
        + r"(?!\w)"
    )
    return re.search(pattern, source) is not None

def filter_hallucinated_triples(
        triples: list[Triple],
        source_text: str
):
    """
    Remove triples whose subject or object cannot be grounded in the source text.
    """
    valid = []
    rejected = []
    for triple in triples:
        subject, predicate, obj = triple
        subject_found = entity_exists(
            subject, source_text
        )
        object_found = entity_exists(
            obj, source_text
        )
        if subject_found and object_found:
            valid.append(triple)
        else:
            rejected.append({
                "triple": triple,
                "subject_found": subject_found,
                "object_found": object_found
            })
    return {
        "valid": valid,
        "rejected": rejected
    }