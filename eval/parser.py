"""
Triplet Parser: strip chat template / special tokens về chuẩn danh sách (Subject, Predicate, Object).
Owner: Bùi Thị Bích Phượng (24021601)
"""


def parse_triples(raw_output: str) -> tuple[list[tuple[str, str, str]], bool]:
    """Trả về (danh_sách_bộ_ba_đã_deduplicate, is_unparseable)."""
    raise NotImplementedError
