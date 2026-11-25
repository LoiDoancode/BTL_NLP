"""
Tính Full-Triple Micro/Macro Precision, Recall, F1, Exact-Set Accuracy & Unparseable Rate.
Owner: Bùi Thị Bích Phượng (24021601)
"""

from typing import Any

from .parser import Triple, normalize_triple


def safe_divide(numerator: float, denominator: float) -> float:
    """Chia an toàn, trả về 0.0 nếu mẫu số bằng 0."""
    return numerator / denominator if denominator != 0 else 0.0


def calculate_counts(predicted: list[Triple], gold: list[Triple]) -> dict[str, int]:
    """Tính số lượng True Positives (TP), False Positives (FP) và False Negatives (FN)."""
    pred_set = {normalize_triple(triple) for triple in predicted}
    gold_set = {normalize_triple(triple) for triple in gold}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    return {"tp": tp, "fp": fp, "fn": fn}


def calculate_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Tính Precision, Recall và F1 Score."""
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_dataset(
    predictions: list[list[Triple]],
    references: list[list[Triple]],
    unparseable_flags: list[bool] | None = None,
) -> dict[str, Any]:
    """
    Đánh giá tập dữ liệu bằng các chỉ số:
    Micro/Macro Precision, Recall, F1, Exact-Set Accuracy và Unparseable Rate.

    Args:
        predictions: Danh sách dự đoán (mỗi phần tử là list các Triple).
        references: Danh sách tham chiếu (mỗi phần tử là list các Triple).
        unparseable_flags: Cờ đánh dấu mẫu không parse được (nếu None thì tính theo pred rỗng).
    """
    if len(predictions) != len(references):
        raise ValueError("Số lượng dự đoán và tham chiếu phải bằng nhau.")
    if not predictions:
        raise ValueError("Danh sách dự đoán không được rỗng.")
    if unparseable_flags is not None and len(unparseable_flags) != len(predictions):
        raise ValueError("Số lượng unparseable_flags phải bằng số lượng dự đoán.")

    total_tp = total_fp = total_fn = 0
    sample_scores: list[dict[str, float]] = []
    exact_matches = 0
    unparseable_count = 0

    for idx, (pred, gold) in enumerate(zip(predictions, references, strict=True)):
        pred_set = {normalize_triple(triple) for triple in pred}
        gold_set = {normalize_triple(triple) for triple in gold}

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if not pred_set and not gold_set:
            scores = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        else:
            scores = calculate_prf(tp, fp, fn)
        sample_scores.append(scores)

        if pred_set == gold_set:
            exact_matches += 1

        is_unparseable = (
            unparseable_flags[idx] if unparseable_flags is not None else len(pred_set) == 0
        )
        if is_unparseable:
            unparseable_count += 1

    n = len(predictions)
    micro = calculate_prf(total_tp, total_fp, total_fn)
    macro = {
        metric: sum(sample[metric] for sample in sample_scores) / n
        for metric in ("precision", "recall", "f1")
    }

    return {
        "micro": micro,
        "macro": macro,
        "exact_set_accuracy": exact_matches / n,
        "unparseable_rate": unparseable_count / n,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "num_samples": n,
    }
