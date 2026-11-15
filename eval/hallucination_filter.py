"""
Entity Hallucination Filter: lọc bỏ các bộ ba bịa đặt thực thể không có cơ sở từ văn bản đầu vào.
Owner: Bùi Thị Bích Phượng (24021601)
"""


def filter_hallucinations(text: str, triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    raise NotImplementedError
