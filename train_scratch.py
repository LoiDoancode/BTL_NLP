"""
Vòng lặp huấn luyện Transformer From-Scratch:
Teacher Forcing, Label Smoothing, AdamW + Warmup Scheduler, Curriculum Learning, lưu best validation checkpoint.
Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW

from data.dataset import get_dataloader
from tokenizer.train_bpe import PAD_ID, BPETokenizerWrapper


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho toàn bộ hệ thống để đảm bảo tính tái lập (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class NoamLRScheduler:
    """
    Học máy học theo công thức Noam trong Vaswani et al. (2017):
    lrate = (d_model ^ -0.5) * min(step_num ^ -0.5, step_num * warmup_steps ^ -1.5)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 512,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self) -> None:
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate

    def rate(self, step: int | None = None) -> float:
        if step is None:
            step = self._step
        step = max(1, step)
        return self.factor * (
            self.d_model ** (-0.5) * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        )

    def get_last_lr(self) -> list[float]:
        return [p["lr"] for p in self.optimizer.param_groups]


class SimpleSeq2SeqBaseline(nn.Module):
    """
    Mô hình Seq2Seq Baseline dự phòng (dùng khi kiểm thử pipeline huấn luyện trước khi Tùng hoàn thiện Core Transformer).
    Khi Tùng hoàn thành `Seq2SeqTransformer`, hệ thống sẽ tự động ưu tiên nạp mô hình của Tùng.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 2,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b_src, l_src = src.shape
        b_tgt, l_tgt = tgt.shape

        src_emb = self.embedding(src) + self.pos_encoder[:, :l_src, :]
        tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :l_tgt, :]

        src_key_padding_mask = src == self.pad_id
        tgt_key_padding_mask = tgt == self.pad_id
        causal_mask = nn.Transformer.generate_square_subsequent_mask(l_tgt, device=tgt.device)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    config: dict[str, Any], vocab_size: int, device: torch.device
) -> tuple[nn.Module, str]:
    """Khởi tạo mô hình Transformer Scratch từ config; fallback mượt mà nếu module core đang phát triển."""
    scratch_cfg = config.get("scratch", {})

    try:
        from models.scratch.transformer import Seq2SeqTransformer

        model = Seq2SeqTransformer(
            vocab_size=vocab_size,
            d_model=scratch_cfg.get("d_model", 512),
            n_heads=scratch_cfg.get("n_heads", 8),
            num_encoder_layers=scratch_cfg.get("num_encoder_layers", 6),
            num_decoder_layers=scratch_cfg.get("num_decoder_layers", 6),
            d_ff=scratch_cfg.get("d_ff", 2048),
            dropout=scratch_cfg.get("dropout", 0.1),
            pre_ln=scratch_cfg.get("use_pre_ln", True),
            weight_tying=scratch_cfg.get("weight_tying", True),
        )
        dummy_src = torch.zeros((1, 2), dtype=torch.long)
        dummy_tgt = torch.zeros((1, 2), dtype=torch.long)
        model(dummy_src, dummy_tgt)
        model_name = "Custom PyTorch Seq2SeqTransformer (Tùng & Lợi)"
    except (NotImplementedError, Exception):
        model = SimpleSeq2SeqBaseline(
            vocab_size=vocab_size,
            d_model=min(256, scratch_cfg.get("d_model", 256)),
            n_heads=4,
            num_layers=2,
            pad_id=PAD_ID,
        )
        model_name = "Transformer Baseline Fallback (for Training Loop verification)"

    model.to(device)
    return model, model_name


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Any | None,
    device: torch.device,
    pad_id: int = PAD_ID,
    clip_grad_norm: float = 1.0,
    max_steps: int | None = None,
) -> tuple[float, int]:
    """Thực hiện huấn luyện một epoch với Teacher Forcing."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_steps is not None and batch_idx >= max_steps:
            break

        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        src_mask = batch["src_mask"].to(device) if "src_mask" in batch else None
        tgt_mask = batch["tgt_mask"].to(device) if "tgt_mask" in batch else None

        decoder_input = tgt_ids[:, :-1]
        target_labels = tgt_ids[:, 1:]

        optimizer.zero_grad()

        logits = model(src_ids, decoder_input, src_mask=src_mask, tgt_mask=tgt_mask)

        vocab_size = logits.size(-1)
        loss = criterion(logits.reshape(-1, vocab_size), target_labels.reshape(-1))

        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        num_valid_tokens = (target_labels != pad_id).sum().item()
        total_loss += loss.item() * num_valid_tokens
        total_tokens += max(1, num_valid_tokens)
        step_count += 1

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, step_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_id: int = PAD_ID,
    max_steps: int | None = None,
) -> float:
    """Đánh giá loss trên tập validation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_steps is not None and batch_idx >= max_steps:
            break

        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        src_mask = batch["src_mask"].to(device) if "src_mask" in batch else None
        tgt_mask = batch["tgt_mask"].to(device) if "tgt_mask" in batch else None

        decoder_input = tgt_ids[:, :-1]
        target_labels = tgt_ids[:, 1:]

        logits = model(src_ids, decoder_input, src_mask=src_mask, tgt_mask=tgt_mask)
        vocab_size = logits.size(-1)
        loss = criterion(logits.reshape(-1, vocab_size), target_labels.reshape(-1))

        num_valid_tokens = (target_labels != pad_id).sum().item()
        total_loss += loss.item() * num_valid_tokens
        total_tokens += max(1, num_valid_tokens)

    return total_loss / max(1, total_tokens)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    train_loss: float,
    config: dict[str, Any],
) -> None:
    """Lưu checkpoint và metadata huấn luyện."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(state, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huấn luyện Transformer From-Scratch cho WebNLG+ 2020."
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--dry_run", action="store_true", help="Chạy thử nghiệm 2 batch để kiểm chứng pipeline"
    )
    parser.add_argument(
        "--num_triples_filter", type=str, default=None, help="Lọc curriculum: 1, 2, '>=3'"
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    scratch_cfg = config.get("scratch", {})
    epochs = args.epochs or (1 if args.dry_run else scratch_cfg.get("epochs", 30))
    batch_size = args.batch_size or (4 if args.dry_run else scratch_cfg.get("batch_size", 32))
    lr = args.lr or scratch_cfg.get("learning_rate", 5.0e-4)
    label_smoothing = scratch_cfg.get("label_smoothing", 0.1)
    max_length = scratch_cfg.get("max_length", 256)
    checkpoint_dir = Path(scratch_cfg.get("checkpoint_dir", "outputs/checkpoints/scratch"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("BTL NLP: VÒNG LẶP HUẤN LUYỆN TRANSFORMER FROM-SCRATCH")
    print("Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy")
    print(f"Thiết bị: {device} | Seed: {seed} | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"Label Smoothing: {label_smoothing} | Max Length: {max_length}")
    print("=" * 70)

    tok_dir = config.get("tokenizer", {}).get("save_dir", "tokenizer/artifacts")
    print(f"[*] Đang nạp Tokenizer từ: {tok_dir} ...")
    tokenizer = BPETokenizerWrapper.load(tok_dir)
    print(f"[OK] Tokenizer đã sẵn sàng! Vocab size = {tokenizer.vocab_size:,}")

    data_dir = Path(config.get("data", {}).get("processed_dir", "data/processed"))
    train_file = data_dir / "train.json"
    val_file = data_dir / "val.json"

    print(f"[*] Đang nạp tập Train: {train_file} ...")
    train_loader = get_dataloader(
        train_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
        num_triples_filter=args.num_triples_filter,
    )

    print(f"[*] Đang nạp tập Validation: {val_file} ...")
    val_loader = get_dataloader(
        val_file,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
        max_length=max_length,
        num_triples_filter=args.num_triples_filter,
    )

    print(
        f"[OK] Đã nạp thành công: {len(train_loader)} train batches, {len(val_loader)} val batches."
    )

    model, model_desc = create_model(config, tokenizer.vocab_size, device)
    print(f"[*] Mô hình: {model_desc}")
    print(f"[*] Tổng số tham số có thể huấn luyện: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    warmup_steps = scratch_cfg.get("warmup_steps", 4000)
    scheduler = NoamLRScheduler(
        optimizer, d_model=scratch_cfg.get("d_model", 512), warmup_steps=warmup_steps
    )

    best_val_loss = float("inf")
    train_history = []
    max_steps_per_epoch = 2 if args.dry_run else None

    start_time_all = time.time()
    print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, steps = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            pad_id=PAD_ID,
            max_steps=max_steps_per_epoch,
        )

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            pad_id=PAD_ID,
            max_steps=max_steps_per_epoch,
        )

        epoch_time = time.time() - epoch_start
        curr_lr = scheduler.get_last_lr()[0] if scheduler else lr

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_ckpt = checkpoint_dir / "best_model.pt"
            save_checkpoint(best_ckpt, model, optimizer, epoch, val_loss, train_loss, config)

        latest_ckpt = checkpoint_dir / "latest_model.pt"
        save_checkpoint(latest_ckpt, model, optimizer, epoch, val_loss, train_loss, config)

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "best_val_loss": round(best_val_loss, 4),
            "lr": f"{curr_lr:.2e}",
            "is_best": is_best,
            "time_sec": round(epoch_time, 2),
        }
        train_history.append(log_entry)

        status_flag = " ⭐ [BEST CHECKPOINT]" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {curr_lr:.2e} | "
            f"Time: {epoch_time:.1f}s{status_flag}"
        )

    total_duration = time.time() - start_time_all
    summary = {
        "hardware": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "seed": seed,
        "max_length": max_length,
        "vocab_size": tokenizer.vocab_size,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "total_time_sec": round(total_duration, 2),
        "history": train_history,
    }

    log_file = checkpoint_dir / "train_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("HUẤN LUYỆN HOÀN TẤT!")
    print(f"- Checkpoint tốt nhất: {checkpoint_dir / 'best_model.pt'}")
    print(f"- Checkpoint mới nhất:  {checkpoint_dir / 'latest_model.pt'}")
    print(f"- File nhật ký huấn luyện: {log_file}")
    print(f"- Tổng thời gian: {total_duration:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
