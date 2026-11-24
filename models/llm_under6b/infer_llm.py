"""
Sinh dự đoán trên tập test cho 02 LLMs (< 6B), đo inference latency & xuất predictions chuẩn (s | p | o).
Owner: Nguyễn Chí Phong (24020268)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm

from models.llm_under6b.model_survey import load_base_model, load_tokenizer
from models.llm_under6b.prompts import (
    DEFAULT_FEW_SHOT_EXAMPLES,
    build_messages,
    load_webnlg_records,
    normalize_and_sort_triples,
    parse_llm_triples,
    render_chat_prompt,
    resolve_split_path,
)
from scripts.sft_qlora import slugify_model_id


def _to_canonical_set(triples: list[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    return {
        (s.strip().lower(), p.strip().lower(), o.strip().lower())
        for s, p, o in triples
        if s.strip() and p.strip() and o.strip()
    }


def _compute_full_triple_metrics(
    predictions: list[list[tuple[str, str, str]]],
    references: list[list[tuple[str, str, str]]],
    unparseable_flags: list[bool] | None = None,
) -> dict[str, float]:
    """Tính nhanh Full-Triple Micro/Macro P, R, F1, Exact-Set và Unparseable Rate độc lập."""
    n = len(predictions)
    if n == 0:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "exact_set_accuracy": 0.0,
            "unparseable_rate": 0.0,
            "num_samples": 0,
        }

    total_tp = total_pred = total_ref = exact_matches = unparseable_count = 0
    macro_p_sum = macro_r_sum = macro_f1_sum = 0.0

    for idx, (pred_list, ref_list) in enumerate(zip(predictions, references, strict=True)):
        pred_set = _to_canonical_set(pred_list)
        ref_set = _to_canonical_set(ref_list)
        tp = len(pred_set & ref_set)

        total_tp += tp
        total_pred += len(pred_set)
        total_ref += len(ref_set)
        if pred_set == ref_set:
            exact_matches += 1
        is_unparseable = (
            unparseable_flags[idx] if unparseable_flags is not None else len(pred_set) == 0
        )
        if is_unparseable:
            unparseable_count += 1

        p_i = tp / len(pred_set) if pred_set else (1.0 if not ref_set else 0.0)
        r_i = tp / len(ref_set) if ref_set else (1.0 if not pred_set else 0.0)
        f1_i = (2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) > 0 else 0.0
        macro_p_sum += p_i
        macro_r_sum += r_i
        macro_f1_sum += f1_i

    micro_p = total_tp / total_pred if total_pred > 0 else 0.0
    micro_r = total_tp / total_ref if total_ref > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": macro_p_sum / n,
        "macro_recall": macro_r_sum / n,
        "macro_f1": macro_f1_sum / n,
        "exact_set_accuracy": exact_matches / n,
        "unparseable_rate": unparseable_count / n,
        "num_samples": n,
        "total_predicted_triples": total_pred,
        "total_reference_triples": total_ref,
        "total_true_positives": total_tp,
    }


def _compute_stratified(
    predictions: list[list[tuple[str, str, str]]],
    references: list[list[tuple[str, str, str]]],
    categories: list[str],
    unparseable_flags: list[bool],
) -> dict[str, dict[str, dict[str, float]]]:
    buckets: dict[str, list[int]] = {"1": [], "2": [], ">=3": []}
    cat_indices: dict[str, list[int]] = {}
    for i, (ref, cat) in enumerate(zip(references, categories, strict=True)):
        count = len(_to_canonical_set(ref))
        buckets["1" if count <= 1 else ("2" if count == 2 else ">=3")].append(i)
        cat_indices.setdefault(cat, []).append(i)

    def _subset(idxs: list[int]) -> dict[str, float]:
        return _compute_full_triple_metrics(
            [predictions[i] for i in idxs],
            [references[i] for i in idxs],
            [unparseable_flags[i] for i in idxs],
        )

    return {
        "by_triple_count": {k: _subset(v) for k, v in buckets.items() if v},
        "by_category": {cat: _subset(idxs) for cat, idxs in sorted(cat_indices.items())},
    }


def run_inference(args: argparse.Namespace) -> dict:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("eval", {})
    llm_cfg = cfg.get("llm_under6b", {})

    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    test_path = (
        Path(args.test_file) if args.test_file else resolve_split_path(processed_dir, "test")
    )
    pred_dir = Path(args.output_dir or eval_cfg.get("predictions_dir", "outputs/predictions"))
    pred_dir.mkdir(parents=True, exist_ok=True)

    samples = load_webnlg_records(test_path, max_samples=args.max_samples)
    if not samples:
        raise ValueError(f"No samples found in {test_path}")

    load_in_4bit = not args.no_4bit and bool(llm_cfg.get("load_in_4bit", True))
    model_source = args.adapter_dir if args.adapter_dir else args.model_id
    tokenizer = load_tokenizer(model_source if Path(str(model_source)).exists() else args.model_id)
    tokenizer.padding_side = "left"

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    base_model = load_base_model(args.model_id, load_in_4bit=load_in_4bit)
    if args.mode == "sft":
        if not args.adapter_dir:
            raise ValueError("--adapter_dir is required when --mode sft")
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    else:
        model = base_model
    model.eval()

    few_shot = DEFAULT_FEW_SHOT_EXAMPLES if args.mode == "few_shot" else None
    prompts = []
    for rec in samples:
        messages = build_messages(
            text=rec["text"],
            triples=None,
            use_cot=args.use_cot,
            few_shot_examples=few_shot,
        )
        prompts.append(render_chat_prompt(tokenizer, messages, add_generation_prompt=True))

    sorted_indices = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    sorted_outputs: list[str] = [""] * len(prompts)
    t_start = time.perf_counter()

    for start in tqdm(range(0, len(sorted_indices), args.batch_size), desc=f"Infer [{args.mode}]"):
        batch_indices = sorted_indices[start : start + args.batch_size]
        batch_prompts = [prompts[i] for i in batch_indices]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        )
        if torch.cuda.is_available():
            enc = {k: v.to("cuda:0") for k, v in enc.items()}

        input_len = enc["input_ids"].shape[1]
        eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
        with torch.inference_mode():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )

        for idx, seq in zip(batch_indices, gen_ids, strict=True):
            completion_ids = seq[input_len:]
            decoded = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            sorted_outputs[idx] = decoded

    raw_outputs = sorted_outputs

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time_s = time.perf_counter() - t_start
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    parsed_predictions: list[list[tuple[str, str, str]]] = []
    references: list[list[tuple[str, str, str]]] = []
    categories: list[str] = []
    unparseable_flags: list[bool] = []

    run_name = args.run_name or (
        f"{slugify_model_id(args.model_id)}_{args.mode}{'_cot' if args.use_cot else ''}"
    )
    pred_jsonl_path = pred_dir / f"{run_name}_predictions.jsonl"
    pred_txt_path = pred_dir / f"{run_name}_predictions.txt"

    with (
        open(pred_jsonl_path, "w", encoding="utf-8") as f_jsonl,
        open(pred_txt_path, "w", encoding="utf-8") as f_txt,
    ):
        for rec, raw_out in zip(samples, raw_outputs, strict=True):
            triples, is_unparseable = parse_llm_triples(raw_out)
            ref = normalize_and_sort_triples(rec["text"], rec["triples"], sort_by_subject=False)

            parsed_predictions.append(triples)
            references.append(ref)
            categories.append(rec.get("category", "Unknown"))
            unparseable_flags.append(is_unparseable)

            record_out = {
                "id": rec["id"],
                "category": rec.get("category", "Unknown"),
                "text": rec["text"],
                "raw_output": raw_out,
                "predicted_triples": [list(t) for t in triples],
                "reference_triples": [list(t) for t in ref],
                "is_unparseable": is_unparseable,
            }
            f_jsonl.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            formatted_line = " | ".join(f"({s} | {p} | {o})" for s, p, o in triples)
            f_txt.write(formatted_line + "\n")

    overall_raw = _compute_full_triple_metrics(parsed_predictions, references, unparseable_flags)
    stratified = _compute_stratified(
        parsed_predictions,
        references,
        categories=categories,
        unparseable_flags=unparseable_flags,
    )

    metrics_report = {
        "run_name": run_name,
        "model_id": args.model_id,
        "mode": args.mode,
        "adapter_dir": args.adapter_dir,
        "use_cot": args.use_cot,
        "num_samples": len(samples),
        "inference_time_seconds": round(total_time_s, 2),
        "ms_per_sample": round(1000.0 * total_time_s / len(samples), 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "overall": overall_raw,
        "stratified": stratified,
    }

    metrics_path = pred_dir / f"{run_name}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f_m:
        json.dump(metrics_report, f_m, indent=2, ensure_ascii=False)

    print(
        f"[{run_name}] Micro F1={overall_raw['micro_f1']:.4f} "
        f"(P={overall_raw['micro_precision']:.4f}, R={overall_raw['micro_recall']:.4f}) | "
        f"Macro F1={overall_raw['macro_f1']:.4f} | "
        f"Exact-Set={overall_raw['exact_set_accuracy']:.4f} | "
        f"Unparseable={overall_raw['unparseable_rate']:.4f} | "
        f"Latency={metrics_report['ms_per_sample']} ms/sample"
    )
    return metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched LLM inference & WebNLG+ evaluation.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_id", required=True, help="Base Hugging Face model ID")
    parser.add_argument(
        "--mode",
        choices=["sft", "few_shot", "zero_shot"],
        default="sft",
        help="Evaluation mode",
    )
    parser.add_argument("--adapter_dir", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--run_name", default=None, help="Output file prefix")
    parser.add_argument("--test_file", default=None, help="Override test JSONL path")
    parser.add_argument("--output_dir", default=None, help="Override predictions directory")
    parser.add_argument("--use_cot", action="store_true", help="Enable Entity-First CoT prompt")
    parser.add_argument("--no_4bit", action="store_true", help="Load base model in bf16")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max_input_length", type=int, default=768, help="Max prompt tokens")
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Max generated tokens")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
