"""
Unit tests cho bộ tính toán Metric đánh giá Text-to-Triples.
Owner: Bùi Thị Bích Phượng (24021601)
"""

import pytest

from eval.metrics import calculate_counts, calculate_prf, evaluate_dataset


def test_full_triple_matching() -> None:
    pred = [("A_Entity", "relation", "B"), ("B", "relation", "C")]
    gold = [("A Entity", "relation", "B"), ("B", "relation", "D")]
    assert calculate_counts(pred, gold) == {"tp": 1, "fp": 1, "fn": 1}


def test_prf() -> None:
    result = calculate_prf(2, 1, 1)
    assert result["precision"] == 2 / 3
    assert result["recall"] == 2 / 3
    assert abs(result["f1"] - 2 / 3) < 1e-9


def test_exact_set_accuracy_and_unparseable_rate() -> None:
    predictions = [[("A", "r", "B")], []]
    references = [[("A", "r", "B")], [("C", "r", "E")]]
    result = evaluate_dataset(predictions, references)
    assert result["exact_set_accuracy"] == 0.5
    assert result["unparseable_rate"] == 0.5


def test_empty_prediction() -> None:
    assert calculate_counts([], [("A", "r", "B")]) == {"tp": 0, "fp": 0, "fn": 1}


def test_mismatched_dataset_size() -> None:
    with pytest.raises(ValueError):
        evaluate_dataset([[("A", "r", "B")]], [])
