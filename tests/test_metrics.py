import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import calculate_counts, calculate_prf, evaluate_dataset

def test_full_triple_matching():
    pred = [
        ("A", "relation", "B"),
        ("B", "relation", "C")
    ]
    gold = [
        ("A", "relation", "B"),
        ("B", "relation", "D")
    ]
    result = calculate_counts(pred, gold)
    assert result == {
        "tp": 1,
        "fp": 1,
        "fn": 1
    }

def test_prf():
    result = calculate_prf(2, 1, 1)
    assert result["precision"] == 2 / 3
    assert result["recall"] == 2 / 3
    assert abs(result["f1"] - 2 / 3) < 1e-9

def test_exact_set_accuracy():
    predictions = [
        [("A", "r", "B")],
        [("C", "r", "D")]
    ]
    references = [
        [("A", "r", "B")],
        [("C", "r", "E")]
    ]
    result = evaluate_dataset(
        predictions,
        references
    )
    assert result["exact_set_accuracy"] == 0.5

def test_empty_prediction():
    result = calculate_counts(
        [],
        [("A", "r", "B")]
    )
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 1

def test_mismatched_dataset_size():
    try:
        evaluate_dataset(
            [[("A", "r", "B")]],
            []
        )
    except ValueError:
        return
    assert False, "Expected ValueError"