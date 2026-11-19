"""
Tính Full-Triple Micro/Macro Precision, Recall, F1, Exact-Set Accuracy & Unparseable Rate.
Owner: Bùi Thị Bích Phượng (24021601)
"""


from .parser import Triple, normalize_triple

def safe_divide(numerator, denominator):
    """Chia an toàn, trả về 0 nếu mẫu số bằng 0."""
    return numerator / denominator if denominator != 0 else 0.0

def calculate_counts(predicted, gold):
    """Tính số lượng True Positives (TP), False Positives (FP) và False Negatives (FN)."""
    pred_set = {normalize_triple(triple) for triple in predicted}
    gold_set = {normalize_triple(triple) for triple in gold}

    tp = len(pred_set & gold_set)  # True Positives
    fp = len(pred_set - gold_set)  # False Positives
    fn = len(gold_set - pred_set)  # False Negatives

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def calculate_prf(tp, fp, fn):
    """Tính Precision, Recall và F1 Score."""
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate_dataset(
        predictions: list[list[Triple]],
        references: list[list[Triple]]
):
    """
    Đánh giá tập dữ liệu bằng cách tính các chỉ số Micro/Macro Precision, Recall, F1, Exact-Set Accuracy và Unparseable Rate.
    
    Args:
        predictions: Danh sách các dự đoán (mỗi dự đoán là một danh sách các Triple).
        references: Danh sách các tham chiếu (mỗi tham chiếu là một danh sách các Triple).
    """
    if len(predictions) != len(references):
        raise ValueError("Số lượng dự đoán và tham chiếu phải bằng nhau.")  
    if not predictions:
        raise ValueError("Danh sách dự đoán không được rỗng.")
    total_tp = total_fp = total_fn = 0
    sample_scores = []
    exact_matches = 0
    for pred, gold in zip(predictions, references):
        counts = calculate_counts(pred, gold)
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        scores = calculate_prf(tp, fp, fn)
        sample_scores.append(scores)
        pred_set = {normalize_triple(triple) for triple in pred}
        gold_set = {normalize_triple(triple) for triple in gold}
        if pred_set == gold_set:
            exact_matches += 1
    micro = calculate_prf(total_tp, total_fp, total_fn)
    macro = {
        metric: sum(
            sample[metric] for sample in sample_scores
        ) / len(sample_scores) for metric in ["precision", "recall", "f1"]
    }
    exact_accuracy = exact_matches / len(predictions)
    return {
        "micro": micro,
        "macro": macro,
        "exact_set_accuracy": exact_accuracy,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "num_samples": len(predictions)
    }