"""
Giải mã tự hồi quy cho Transformer From-Scratch: Greedy Search & Beam Search (B=4).
Xuất file predictions chuẩn hóa (Subject | Predicate | Object).
Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data.dataset import get_dataloader
from eval.parser import parse_triplet
from models.scratch.masks import create_padding_mask, create_target_mask
from tokenizer.train_bpe import BOS_ID, EOS_ID, PAD_ID, BPETokenizerWrapper
from train_scratch import create_model, set_seed


def _length_penalty(length: int, alpha: float = 0.6) -> float:
    """Google NMT length penalty: ((5 + L) / 6)^alpha."""
    if alpha <= 0.0:
        return 1.0
    return ((5.0 + float(length)) / 6.0) ** alpha


def _step_logits(
    model: nn.Module,
    src_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    enc_out: torch.Tensor | None = None,
    src_mask: torch.Tensor | None = None,
    pad_id: int = PAD_ID,
) -> torch.Tensor:
    """
    Tính logits cho token cuối cùng ở bước giải mã hiện tại.
    Tối ưu hóa bằng cách tái sử dụng enc_out nếu mô hình hỗ trợ encode()/decode()/output_proj().
    """
    if enc_out is not None and hasattr(model, "decode") and hasattr(model, "output_proj"):
        tgt_mask = create_target_mask(tgt_ids, pad_idx=pad_id)
        dec_out = model.decode(tgt_ids, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        logits = model.output_proj(dec_out[:, -1, :])
        return logits

    logits = model(src_ids, tgt_ids, src_mask=src_mask)
    return logits[:, -1, :]


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor | None = None,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
    max_len: int = 128,
) -> torch.Tensor:
    """
    Giải mã tự hồi quy theo chiến lược Greedy Search (chọn argmax tại mỗi bước).

    Args:
        model: Mô hình Seq2Seq Transformer.
        src_ids: Tensor nguồn [B, S].
        src_mask: Mask nguồn [B, 1, 1, S] (tùy chọn).
        bos_id: Token bắt đầu chuỗi [BOS].
        eos_id: Token kết thúc chuỗi [EOS].
        pad_id: Token đệm [PAD].
        max_len: Độ dài tối đa chuỗi sinh ra (bao gồm BOS/EOS).

    Returns:
        Tensor [B, L] chứa chuỗi token IDs đã giải mã.
    """
    model.eval()
    device = src_ids.device
    batch_size = src_ids.size(0)

    if src_mask is None:
        src_mask = create_padding_mask(src_ids, pad_idx=pad_id)

    enc_out = model.encode(src_ids, src_mask=src_mask) if hasattr(model, "encode") else None

    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logits = _step_logits(
            model=model,
            src_ids=src_ids,
            tgt_ids=generated,
            enc_out=enc_out,
            src_mask=src_mask,
            pad_id=pad_id,
        )
        next_tokens = torch.argmax(logits, dim=-1)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

        finished = finished | (next_tokens == eos_id)
        if torch.all(finished):
            break

    return generated


@torch.no_grad()
def _beam_search_single(
    model: nn.Module,
    src_id: torch.Tensor,
    src_mask: torch.Tensor,
    beam_size: int = 4,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
    max_len: int = 128,
    length_penalty: float = 0.6,
) -> list[int]:
    """Beam Search tự hồi quy cho một mẫu đơn lẻ [1, S]."""
    device = src_id.device
    enc_out = model.encode(src_id, src_mask=src_mask) if hasattr(model, "encode") else None

    # Mỗi giả thuyết: (sequence_tensor [1, L], log_prob_sum, is_finished)
    beams: list[tuple[torch.Tensor, float, bool]] = [
        (torch.tensor([[bos_id]], dtype=torch.long, device=device), 0.0, False)
    ]

    for _ in range(max_len - 1):
        all_finished = all(fin for _, _, fin in beams)
        if all_finished:
            break

        candidates: list[tuple[torch.Tensor, float, bool]] = []
        active_indices: list[int] = []
        active_seqs: list[torch.Tensor] = []

        for idx, (seq, score, fin) in enumerate(beams):
            if fin:
                candidates.append((seq, score, True))
            else:
                active_indices.append(idx)
                active_seqs.append(seq)

        if active_seqs:
            # Tất cả active_seqs ở cùng bước đều có cùng độ dài
            batched_tgt = torch.cat(active_seqs, dim=0)
            num_active = batched_tgt.size(0)
            batched_src = src_id.expand(num_active, -1)
            batched_src_mask = src_mask.expand(num_active, -1, -1, -1)
            batched_enc = enc_out.expand(num_active, -1, -1) if enc_out is not None else None

            logits = _step_logits(
                model=model,
                src_ids=batched_src,
                tgt_ids=batched_tgt,
                enc_out=batched_enc,
                src_mask=batched_src_mask,
                pad_id=pad_id,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            vocab_size = log_probs.size(-1)
            k = min(beam_size, vocab_size)
            topk_log_probs, topk_ids = torch.topk(log_probs, k=k, dim=-1)

            for i, beam_idx in enumerate(active_indices):
                seq, score, _ = beams[beam_idx]
                for j in range(k):
                    tok_id = int(topk_ids[i, j].item())
                    tok_lp = float(topk_log_probs[i, j].item())
                    new_seq = torch.cat(
                        [seq, torch.tensor([[tok_id]], dtype=torch.long, device=device)], dim=1
                    )
                    new_score = score + tok_lp
                    new_fin = tok_id == eos_id
                    candidates.append((new_seq, new_score, new_fin))

        # Xếp hạng theo điểm có chuẩn hóa độ dài (length penalty)
        candidates.sort(
            key=lambda item: item[1] / _length_penalty(item[0].size(1), alpha=length_penalty),
            reverse=True,
        )
        beams = candidates[:beam_size]

    # Ưu tiên giả thuyết đã kết thúc bằng EOS; nếu không có thì lấy giả thuyết điểm cao nhất
    finished_beams = [b for b in beams if b[2]]
    best_pool = finished_beams if finished_beams else beams
    best_seq, _, _ = max(
        best_pool,
        key=lambda item: item[1] / _length_penalty(item[0].size(1), alpha=length_penalty),
    )
    return best_seq.squeeze(0).detach().cpu().tolist()


@torch.no_grad()
def beam_search_decode(
    model: nn.Module,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor | None = None,
    beam_size: int = 4,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID,
    max_len: int = 128,
    length_penalty: float = 0.6,
) -> torch.Tensor:
    """
    Giải mã tự hồi quy theo chiến lược Beam Search (mặc định B=4).

    Args:
        model: Mô hình Seq2Seq Transformer.
        src_ids: Tensor nguồn [B, S].
        src_mask: Mask nguồn [B, 1, 1, S] (tùy chọn).
        beam_size: Kích thước beam B (mặc định 4).
        bos_id: Token [BOS].
        eos_id: Token [EOS].
        pad_id: Token [PAD].
        max_len: Độ dài tối đa chuỗi sinh ra.
        length_penalty: Hệ số phạt độ dài alpha (Google NMT).

    Returns:
        Tensor [B, L_max] chứa các chuỗi token IDs đã giải mã (đệm bằng pad_id).
    """
    if beam_size <= 1:
        return greedy_decode(
            model=model,
            src_ids=src_ids,
            src_mask=src_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_len=max_len,
        )

    model.eval()
    device = src_ids.device
    if src_mask is None:
        src_mask = create_padding_mask(src_ids, pad_idx=pad_id)

    batch_results: list[list[int]] = []
    for idx in range(src_ids.size(0)):
        seq_ids = _beam_search_single(
            model=model,
            src_id=src_ids[idx : idx + 1],
            src_mask=src_mask[idx : idx + 1],
            beam_size=beam_size,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_len=max_len,
            length_penalty=length_penalty,
        )
        batch_results.append(seq_ids)

    max_out_len = max(len(seq) for seq in batch_results)
    padded = [seq + [pad_id] * (max_out_len - len(seq)) for seq in batch_results]
    return torch.tensor(padded, dtype=torch.long, device=device)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    vocab_size: int,
    device: torch.device,
    fallback_config: dict[str, Any] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Nạp mô hình Transformer Scratch từ file checkpoint (.pt)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config") or fallback_config or {}
    model, _ = create_model(config, vocab_size, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Giải mã tự hồi quy (Greedy / Beam Search) cho Transformer From-Scratch."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/checkpoints/scratch/best_model.pt"
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--method", type=str, default="beam", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scratch_cfg = config.get("scratch", {})
    beam_size = args.beam_size if args.beam_size is not None else scratch_cfg.get("beam_size", 4)
    max_length = scratch_cfg.get("max_length", 256)

    tok_dir = config.get("tokenizer", {}).get("save_dir", "tokenizer/artifacts")
    tokenizer = BPETokenizerWrapper.load(tok_dir)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    model, ckpt_meta = load_model_from_checkpoint(ckpt_path, tokenizer.vocab_size, device, config)

    data_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    data_file = data_dir / f"{args.split}.json"
    dataloader = get_dataloader(
        data_file,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        max_length=max_length,
    )

    out_dir = Path(config.get("eval", {}).get("predictions_dir", "outputs/predictions"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.output) if args.output else out_dir / f"scratch_{args.method}_{args.split}.json"
    )

    print("=" * 70)
    print(f"GIẢI MÃ TỰ HỒI QUY ({args.method.upper()}) — Transformer From-Scratch")
    print(f"Checkpoint: {ckpt_path} (Epoch {ckpt_meta.get('epoch', '?')})")
    print(
        f"Split: {args.split} | Beam size: {beam_size} | Max len: {args.max_len} | Device: {device}"
    )
    print("=" * 70)

    predictions: list[dict[str, Any]] = []
    start_time = time.time()

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        src_mask = batch["src_mask"].to(device)

        if args.method == "greedy":
            out_ids = greedy_decode(
                model=model,
                src_ids=src_ids,
                src_mask=src_mask,
                bos_id=tokenizer.bos_token_id,
                eos_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                max_len=args.max_len,
            )
        else:
            out_ids = beam_search_decode(
                model=model,
                src_ids=src_ids,
                src_mask=src_mask,
                beam_size=beam_size,
                bos_id=tokenizer.bos_token_id,
                eos_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                max_len=args.max_len,
                length_penalty=args.length_penalty,
            )

        for i in range(out_ids.size(0)):
            pred_text = tokenizer.decode(out_ids[i], skip_special_tokens=True)
            pred_triples = [list(t) for t in parse_triplet(pred_text)]
            predictions.append(
                {
                    "id": batch["ids"][i],
                    "category": batch["categories"][i],
                    "num_triples": batch["num_triples"][i],
                    "text": batch["src_texts"][i],
                    "gold_text": batch["tgt_texts"][i],
                    "gold_triples": batch["triples"][i],
                    "pred_text": pred_text,
                    "pred_triples": pred_triples,
                }
            )
            if args.max_samples is not None and len(predictions) >= args.max_samples:
                break

        if args.max_samples is not None and len(predictions) >= args.max_samples:
            break

    elapsed = time.time() - start_time
    payload = {
        "method": args.method,
        "beam_size": beam_size if args.method == "beam" else 1,
        "length_penalty": args.length_penalty if args.method == "beam" else 0.0,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_samples": len(predictions),
        "elapsed_sec": round(elapsed, 2),
        "predictions": predictions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Đã giải mã {len(predictions)} mẫu trong {elapsed:.2f}s -> {out_path}")


if __name__ == "__main__":
    main()
