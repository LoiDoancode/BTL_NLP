"""
So sánh nhật ký huấn luyện (train_log.json) giữa Transformer Baseline và Transformer Cải tiến
(Pre-LN + Weight Tying + Curriculum Learning).
Owner: Doãn Duy Lợi (24021549)
"""

import argparse
import json
from pathlib import Path
from typing import Any


def compare_runs(baseline_log: Path, improved_log: Path, output_path: Path) -> dict[str, Any]:
    with open(baseline_log, encoding="utf-8") as f:
        base = json.load(f)
    with open(improved_log, encoding="utf-8") as f:
        imp = json.load(f)

    base_best = float(base["best_val_loss"])
    imp_best = float(imp["best_val_loss"])
    val_loss_delta = imp_best - base_best
    val_loss_rel_pct = (val_loss_delta / base_best * 100.0) if base_best > 0 else 0.0

    epoch_comparison = []
    max_epochs = max(len(base.get("history", [])), len(imp.get("history", [])))
    for idx in range(max_epochs):
        b_row = base["history"][idx] if idx < len(base.get("history", [])) else {}
        i_row = imp["history"][idx] if idx < len(imp.get("history", [])) else {}
        epoch_comparison.append(
            {
                "epoch": idx + 1,
                "baseline_train_loss": b_row.get("train_loss"),
                "baseline_val_loss": b_row.get("val_loss"),
                "improved_train_loss": i_row.get("train_loss"),
                "improved_val_loss": i_row.get("val_loss"),
                "improved_curriculum_stage": i_row.get("curriculum_stage"),
            }
        )

    report = {
        "baseline": {
            "log_path": str(baseline_log),
            "run_name": base.get("run_name", "baseline"),
            "architecture": base.get("architecture", {}),
            "best_val_loss": round(base_best, 4),
            "final_train_loss": base["history"][-1]["train_loss"] if base.get("history") else None,
            "total_time_sec": base.get("total_time_sec"),
        },
        "improved": {
            "log_path": str(improved_log),
            "run_name": imp.get("run_name", "improved"),
            "architecture": imp.get("architecture", {}),
            "best_val_loss": round(imp_best, 4),
            "final_train_loss": imp["history"][-1]["train_loss"] if imp.get("history") else None,
            "total_time_sec": imp.get("total_time_sec"),
        },
        "delta": {
            "best_val_loss_diff": round(val_loss_delta, 4),
            "best_val_loss_rel_pct": round(val_loss_rel_pct, 2),
        },
        "epoch_by_epoch": epoch_comparison,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh train_log.json Baseline vs Improved.")
    parser.add_argument(
        "--baseline", type=str, default="outputs/checkpoints/scratch_baseline/train_log.json"
    )
    parser.add_argument(
        "--improved", type=str, default="outputs/checkpoints/scratch_improved/train_log.json"
    )
    parser.add_argument("--output", type=str, default="outputs/checkpoints/scratch_comparison.json")
    args = parser.parse_args()

    report = compare_runs(Path(args.baseline), Path(args.improved), Path(args.output))
    print(f"[OK] Đã lưu báo cáo so sánh tại: {args.output}")
    print(
        f"Baseline best val_loss: {report['baseline']['best_val_loss']:.4f} | "
        f"Improved best val_loss: {report['improved']['best_val_loss']:.4f} | "
        f"Delta: {report['delta']['best_val_loss_diff']:+.4f} ({report['delta']['best_val_loss_rel_pct']:+.2f}%)"
    )


if __name__ == "__main__":
    main()
