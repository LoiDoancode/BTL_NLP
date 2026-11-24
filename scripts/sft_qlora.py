"""
Pipeline SFT & QLoRA 4-bit với Target-only Loss Masking (label = -100 cho phần prompt).
Owner: Nguyễn Chí Phong (24020268)
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, set_seed

from models.llm_under6b.model_survey import (
    LORA_TARGET_MODULES,
    get_parameter_stats,
    load_base_model,
    load_tokenizer,
)
from models.llm_under6b.prompts import (
    build_messages,
    load_webnlg_records,
    normalize_and_sort_triples,
    render_chat_prompt,
    resolve_split_path,
)


def slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


@dataclass
class SFTDataCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def build_sft_dataset(
    data_path: str | Path,
    tokenizer,
    max_seq_length: int = 512,
    use_cot: bool = False,
    sort_triples: bool = True,
    max_samples: int | None = None,
) -> Dataset:
    raw_samples = load_webnlg_records(data_path, max_samples=max_samples)
    records = []
    eos_token = tokenizer.eos_token or ""

    for rec in raw_samples:
        triple_key = "triples" if sort_triples else "triples_unsorted"
        raw_triples = rec.get(triple_key, rec["triples"])
        triples = normalize_and_sort_triples(rec["text"], raw_triples, sort_by_subject=sort_triples)
        if not triples:
            continue
        messages = build_messages(
            text=rec["text"],
            triples=triples,
            entities=None,
            use_cot=use_cot,
        )
        prompt_text = render_chat_prompt(tokenizer, messages[:-1], add_generation_prompt=True)
        target_text = messages[-1]["content"] + eos_token

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)

        input_ids = (prompt_ids + target_ids)[:max_seq_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_seq_length]
        if all(lbl == -100 for lbl in labels):
            continue

        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "length": len(input_ids),
            }
        )

    return Dataset.from_list(records)


def train_model(args: argparse.Namespace) -> dict:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    llm_cfg = cfg.get("llm_under6b", {})
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    max_seq_length = args.max_seq_length or int(llm_cfg.get("max_seq_length", 512))
    load_in_4bit = not args.no_4bit and bool(llm_cfg.get("load_in_4bit", True))
    lora_r = args.lora_r or int(llm_cfg.get("lora_r", 64))
    lora_alpha = args.lora_alpha or int(llm_cfg.get("lora_alpha", 128))
    lora_dropout = float(llm_cfg.get("lora_dropout", 0.05))
    lr = args.lr or float(llm_cfg.get("learning_rate", 2.0e-4))
    batch_size = args.batch_size or int(llm_cfg.get("per_device_train_batch_size", 16))
    grad_accum = args.grad_accum or int(llm_cfg.get("gradient_accumulation_steps", 2))
    epochs = args.epochs or float(llm_cfg.get("num_train_epochs", 2))

    run_tag = args.run_name or (
        f"{slugify_model_id(args.model_id)}_{'cot' if args.use_cot else 'std'}"
    )
    out_dir = (
        Path(args.output_dir or llm_cfg.get("output_dir", "outputs/checkpoints/llm_under6b"))
        / run_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SFT Training: {args.model_id} -> {out_dir} ===")
    tokenizer = load_tokenizer(args.model_id)
    tokenizer.padding_side = "right"

    train_dataset = build_sft_dataset(
        resolve_split_path(processed_dir, "train"),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        use_cot=args.use_cot,
        sort_triples=not args.no_sort_triples,
        max_samples=args.max_train_samples,
    )
    val_dataset = build_sft_dataset(
        resolve_split_path(processed_dir, "val"),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        use_cot=args.use_cot,
        sort_triples=not args.no_sort_triples,
        max_samples=args.max_val_samples,
    )
    print(f"Tokenized samples: train={len(train_dataset)}, val={len(val_dataset)}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    base_model = load_base_model(args.model_id, load_in_4bit=load_in_4bit)
    if load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=args.grad_ckpt
        )
    elif args.grad_ckpt:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False

    param_stats = get_parameter_stats(model)
    assert param_stats["total_params"] < 6_000_000_000, (
        f"Model exceeds 6B limit: {param_stats['total_params']:,}"
    )
    print(
        f"Params: total={param_stats['total_params_billions']}B, "
        f"trainable={param_stats['trainable_params']:,} ({param_stats['trainable_pct']}%)"
    )

    steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum))
    eval_steps = args.eval_steps or max(50, min(250, steps_per_epoch // 4))

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer_runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=min(32, batch_size * 2),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        optim="paged_adamw_8bit" if load_in_4bit else "adamw_torch",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=4,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=SFTDataCollator(pad_token_id=tokenizer.pad_token_id),
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - t0
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    )

    adapter_dir = out_dir / "best_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f_hist:
        json.dump(trainer.state.log_history, f_hist, indent=2)

    eval_metrics = trainer.evaluate()
    summary = {
        "model_id": args.model_id,
        "run_tag": run_tag,
        "adapter_dir": str(adapter_dir),
        "use_cot": args.use_cot,
        "sort_triples": not args.no_sort_triples,
        "load_in_4bit": load_in_4bit,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "max_seq_length": max_seq_length,
        "seed": seed,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        **param_stats,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_loss": round(float(train_result.training_loss), 5),
        "best_eval_loss": round(float(eval_metrics["eval_loss"]), 5),
        "train_time_seconds": round(train_seconds, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
    }

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2, ensure_ascii=False)

    print(
        f"Done {run_tag} in {summary['train_time_seconds']}s | "
        f"train_loss={summary['train_loss']} | best_eval_loss={summary['best_eval_loss']} | "
        f"peak_VRAM={summary['peak_vram_gb']} GB"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT training with Target-Only Loss Masking."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--model_id", required=True, help="Hugging Face model ID (< 6B)")
    parser.add_argument("--run_name", default=None, help="Custom run directory name")
    parser.add_argument("--output_dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--use_cot", action="store_true", help="Enable Entity-First CoT output")
    parser.add_argument(
        "--no_sort_triples",
        action="store_true",
        help="Disable subject-order canonical triplet sorting",
    )
    parser.add_argument(
        "--no_4bit", action="store_true", help="Use full bf16 LoRA instead of 4-bit"
    )
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--epochs", type=float, default=None, help="Override num_train_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override train batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="Override grad accumulation")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="Override LoRA alpha")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Override max_seq_length")
    parser.add_argument("--eval_steps", type=int, default=None, help="Override eval/save interval")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit val samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
