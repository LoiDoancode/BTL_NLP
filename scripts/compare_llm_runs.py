"""
So sánh kết quả huấn luyện (training_summary.json) và đánh giá (metrics.json) giữa các lần chạy LLM < 6B.
Owner: Nguyễn Chí Phong (24020268)
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_llm_runs(checkpoints_dir: Path, predictions_dir: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    if checkpoints_dir.exists():
        for summary_path in sorted(checkpoints_dir.glob("*/training_summary.json")):
            summary = _load_json(summary_path)
            if not summary:
                continue
            run_tag = summary.get("run_tag", summary_path.parent.name)
            metrics_path = predictions_dir / f"{run_tag}_metrics.json"
            metrics = _load_json(metrics_path)
            overall = (metrics or {}).get("overall", {})
            runs.append(
                {
                    "run_tag": run_tag,
                    "model_id": summary.get("model_id"),
                    "use_cot": summary.get("use_cot", False),
                    "lora_r": summary.get("lora_r"),
                    "lora_alpha": summary.get("lora_alpha"),
                    "learning_rate": summary.get("learning_rate"),
                    "epochs": summary.get("num_train_epochs"),
                    "total_params_billions": summary.get("total_params_billions"),
                    "trainable_params": summary.get("trainable_params"),
                    "trainable_pct": summary.get("trainable_pct"),
                    "train_loss": summary.get("train_loss"),
                    "best_eval_loss": summary.get("best_eval_loss"),
                    "train_time_seconds": summary.get("train_time_seconds"),
                    "peak_vram_gb": summary.get("peak_vram_gb"),
                    "gpu_name": summary.get("gpu_name"),
                    "test_micro_f1": round(float(overall["micro_f1"]), 4)
                    if "micro_f1" in overall
                    else None,
                    "test_macro_f1": round(float(overall["macro_f1"]), 4)
                    if "macro_f1" in overall
                    else None,
                    "test_exact_set": round(float(overall["exact_set_accuracy"]), 4)
                    if "exact_set_accuracy" in overall
                    else None,
                    "test_unparseable_rate": round(float(overall["unparseable_rate"]), 4)
                    if "unparseable_rate" in overall
                    else None,
                    "inference_ms_per_sample": (metrics or {}).get("ms_per_sample"),
                }
            )

    baselines: list[dict[str, Any]] = []
    if predictions_dir.exists():
        for m_path in sorted(predictions_dir.glob("*_few_shot_metrics.json")):
            m = _load_json(m_path)
            if not m:
                continue
            ov = m.get("overall", {})
            baselines.append(
                {
                    "run_name": m.get("run_name"),
                    "model_id": m.get("model_id"),
                    "mode": m.get("mode"),
                    "test_micro_f1": round(float(ov.get("micro_f1", 0.0)), 4),
                    "test_macro_f1": round(float(ov.get("macro_f1", 0.0)), 4),
                    "test_exact_set": round(float(ov.get("exact_set_accuracy", 0.0)), 4),
                    "test_unparseable_rate": round(float(ov.get("unparseable_rate", 0.0)), 4),
                    "inference_ms_per_sample": m.get("ms_per_sample"),
                }
            )

    best_run = max(
        (r for r in runs if r["test_micro_f1"] is not None),
        key=lambda r: r["test_micro_f1"],
        default=None,
    )
    return {"few_shot_baselines": baselines, "sft_runs": runs, "best_sft_run": best_run}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả các lần chạy LLM < 6B.")
    parser.add_argument("--checkpoints", type=str, default="outputs/checkpoints/llm_under6b")
    parser.add_argument("--predictions", type=str, default="outputs/predictions")
    parser.add_argument("--output", type=str, default="outputs/checkpoints/llm_comparison.json")
    args = parser.parse_args()

    report = collect_llm_runs(Path(args.checkpoints), Path(args.predictions))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Đã lưu tổng hợp LLM tại: {out_path}")
    for r in report["sft_runs"]:
        print(
            f"  {r['run_tag']}: val_loss={r['best_eval_loss']} | "
            f"Micro F1={r['test_micro_f1']} | Exact-Set={r['test_exact_set']} | "
            f"VRAM={r['peak_vram_gb']} GB | time={r['train_time_seconds']}s"
        )


if __name__ == "__main__":
    main()
