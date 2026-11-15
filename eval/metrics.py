"""
Tính Full-Triple Micro/Macro Precision, Recall, F1, Exact-Set Accuracy & Unparseable Rate.
Owner: Bùi Thị Bích Phượng (24021601)
"""


def compute_metrics(
    predictions: list[list[tuple[str, str, str]]],
    references: list[list[tuple[str, str, str]]],
) -> dict[str, float]:
    raise NotImplementedError
