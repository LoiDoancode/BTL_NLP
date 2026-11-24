"""
Khảo sát, kiểm tra số tham số (< 6B) & đo VRAM khi load 4-bit NF4 trên Colab T4 / Kaggle.
Owner: Nguyễn Chí Phong (24020268)
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAX_PARAMS_LIMIT = 6_000_000_000
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_bnb_config(load_in_4bit: bool = True) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(model_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(model_id: str, load_in_4bit: bool = True) -> AutoModelForCausalLM:
    bnb_config = build_bnb_config(load_in_4bit=load_in_4bit)
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    )


def get_parameter_stats(model: torch.nn.Module) -> dict[str, int | float]:
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        qstate = getattr(param, "quant_state", None)
        orig_numel = math.prod(qstate.shape) if qstate is not None else param.numel()
        total_params += orig_numel
        if param.requires_grad:
            trainable_params += param.numel()
    return {
        "total_params": int(total_params),
        "total_params_billions": round(total_params / 1e9, 4),
        "trainable_params": int(trainable_params),
        "trainable_pct": round(100.0 * trainable_params / max(total_params, 1), 4),
    }


def check_model_vram(
    model_id: str,
    lora_r: int = 64,
    lora_alpha: int = 128,
    batch_size: int = 2,
    seq_len: int = 512,
    load_in_4bit: bool = True,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for VRAM profiling.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = load_tokenizer(model_id)
    base_model = load_base_model(model_id, load_in_4bit=load_in_4bit)
    load_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    if load_in_4bit:
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    else:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.config.use_cache = False

    param_stats = get_parameter_stats(model)
    assert param_stats["total_params"] < MAX_PARAMS_LIMIT, (
        f"{model_id} has {param_stats['total_params']:,} params (>= 6B limit)!"
    )

    vocab_size = getattr(model.config, "vocab_size", len(tokenizer))
    dummy_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda:0")
    dummy_mask = torch.ones_like(dummy_ids)

    model.train()
    outputs = model(input_ids=dummy_ids, attention_mask=dummy_mask, labels=dummy_ids)
    loss = outputs.loss
    loss.backward()
    torch.cuda.synchronize()
    train_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    result = {
        "model_id": model_id,
        "gpu_name": torch.cuda.get_device_name(0),
        "load_in_4bit": load_in_4bit,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "probe_batch_size": batch_size,
        "probe_seq_len": seq_len,
        **param_stats,
        "under_6b_valid": param_stats["total_params"] < MAX_PARAMS_LIMIT,
        "load_vram_gb": round(load_peak_gb, 3),
        "train_peak_vram_gb": round(train_peak_gb, 3),
    }

    del dummy_ids, dummy_mask, outputs, loss, model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey < 6B LLMs and profile 4-bit VRAM.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="Override model list")
    parser.add_argument("--batch-size", type=int, default=2, help="Probe batch size")
    parser.add_argument(
        "--out",
        default="outputs/predictions/model_survey.json",
        help="Path to save survey results JSON",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm_cfg = cfg.get("llm_under6b", {})

    model_ids = args.models if args.models else llm_cfg.get("models", [])
    if not model_ids:
        raise ValueError("No models configured in llm_under6b.models or --models.")

    results = []
    for model_id in model_ids:
        print(f"Profiling {model_id} ...")
        info = check_model_vram(
            model_id=model_id,
            lora_r=int(llm_cfg.get("lora_r", 64)),
            lora_alpha=int(llm_cfg.get("lora_alpha", 128)),
            batch_size=args.batch_size,
            seq_len=int(llm_cfg.get("max_seq_length", 512)),
            load_in_4bit=bool(llm_cfg.get("load_in_4bit", True)),
        )
        results.append(info)
        print(
            f"  -> {info['total_params_billions']}B params "
            f"(trainable: {info['trainable_params']:,}, {info['trainable_pct']}%) | "
            f"load VRAM: {info['load_vram_gb']} GB | "
            f"train peak (bs={args.batch_size}): {info['train_peak_vram_gb']} GB"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)
    print(f"Saved survey report to {out_path}")


if __name__ == "__main__":
    main()
