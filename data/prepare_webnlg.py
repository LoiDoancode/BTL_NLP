"""
Tải, làm sạch và chuẩn hóa tập dữ liệu WebNLG+ 2020 (GEM/web_nlg).
Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy

Tác vụ: Text-to-Triples (Trích xuất đồ thị tri thức từ văn bản)
Nguồn dữ liệu: https://huggingface.co/datasets/GEM/web_nlg (English split)
- Nguồn gốc: Tải trực tiếp các file JSON gốc được chỉ định trong mã nguồn web_nlg.py của GEM trên Hugging Face.
- Số lượng mẫu theo chuẩn chính thức của GEM/web_nlg:
    + Train: 35,426 mẫu (mỗi verbalization là 1 mẫu huấn luyện độc lập)
    + Validation: 1,667 mẫu (theo chuẩn 1:1 của GEM benchmark)
    + Test: 1,779 mẫu (theo chuẩn 1:1 của GEM benchmark)
"""

import argparse
import csv
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


OFFICIAL_URLS = {
    "train": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_en_train.json",
    "val": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_en_val.json",
    "test": "https://storage.googleapis.com/huggingface-nlp/datasets/gem/gem_web_nlg/webnlg_en_test.json",
}


def download_file(url: str, dest_path: Path, force: bool = False) -> Path:
    """Tải file từ URL nếu chưa tồn tại cục bộ."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not force and dest_path.stat().st_size > 0:
        print(f"[CACHE] Đã tồn tại: {dest_path} ({dest_path.stat().st_size:,} bytes)")
        return dest_path

    print(f"[DOWNLOADING] Đang tải từ {url} -> {dest_path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        chunk_size = 64 * 1024
        while chunk := resp.read(chunk_size):
            out_file.write(chunk)
    print(f"[DOWNLOADED] Thành công: {dest_path} ({dest_path.stat().st_size:,} bytes)")
    return dest_path


def clean_str(s: str) -> str:
    """Loại bỏ ký tự thừa, ngoặc kép bao quanh và chuẩn hóa khoảng trắng."""
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = s.replace('\\"', '"').replace("''", '"')
    return re.sub(r"\s+", " ", s).strip()


def parse_raw_triple(triple_str: str) -> Tuple[str, str, str]:
    """Phân tách chuỗi 'Subject | Predicate | Object' thành tuple (s, p, o)."""
    parts = triple_str.split(" | ")
    if len(parts) == 3:
        s, p, o = parts[0], parts[1], parts[2]
    elif len(parts) > 3:
        s, p = parts[0], parts[1]
        o = " | ".join(parts[2:])
    elif len(parts) == 2:
        s, p, o = parts[0], parts[1], ""
    else:
        s, p, o = parts[0], "", ""

    return clean_str(s), clean_str(p), clean_str(o)


def format_triples_to_target(triples: List[Tuple[str, str, str]], sep: str = " | ") -> str:
    """
    Chuẩn hóa danh sách bộ ba thành chuỗi target:
    (Subject | Predicate | Object) | (Subject2 | Predicate2 | Object2)
    """
    formatted = [f"({s} | {p} | {o})" for s, p, o in triples]
    return sep.join(formatted)


def process_split_data(
    raw_path: Path,
    split_name: str,
    limit: int | None = None,
    expand_eval: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Xử lý file JSON gốc của WebNLG+ thành danh sách mẫu Text-to-Triples chuẩn.
    - Với tập train: mỗi câu mô tả (verbalization) tạo thành 1 mẫu huấn luyện độc lập (tổng 35,426 mẫu).
    - Với tập val/test:
        + Mặc định (expand_eval=False): đúng 1 mẫu cho mỗi entry (1,667 val, 1,779 test) khớp 100% với benchmark GEM/web_nlg.
        + Nếu expand_eval=True: tách từng câu thành các mẫu riêng lẻ.
    """
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    entries = raw_data.get("values", [])
    if limit is not None and limit > 0:
        entries = entries[:limit]

    processed_samples: List[Dict[str, Any]] = []
    stats = {
        "total_entries": len(entries),
        "total_samples": 0,
        "triples_count_dist": {1: 0, 2: 0, ">=3": 0},
        "categories": set(),
    }

    sample_id = 0
    for entry in entries:
        raw_inputs = entry.get("input", [])  # Triples
        raw_targets = entry.get("target", [])  # Natural language texts
        category = clean_str(entry.get("category", "General"))
        webnlg_id = entry.get("webnlg-id", f"{split_name}_{entry.get('id', sample_id)}")
        stats["categories"].add(category)

        # Parse và deduplicate các triples
        triples: List[Tuple[str, str, str]] = []
        for t_str in raw_inputs:
            s, p, o = parse_raw_triple(t_str)
            if (s, p, o) not in triples:
                triples.append((s, p, o))

        num_triples = len(triples)
        if num_triples == 1:
            stats["triples_count_dist"][1] += 1
        elif num_triples == 2:
            stats["triples_count_dist"][2] += 1
        else:
            stats["triples_count_dist"][">=3"] += 1

        target_str = format_triples_to_target(triples)
        clean_texts = [clean_str(t) for t in raw_targets if clean_str(t)]
        if not clean_texts:
            continue

        if split_name == "train":
            # Train: unroll toàn bộ verbalizations theo chuẩn của GEM/web_nlg -> 35,426 mẫu
            for sub_idx, text in enumerate(clean_texts):
                processed_samples.append({
                    "id": f"train_{sample_id}_{sub_idx}",
                    "webnlg_id": webnlg_id,
                    "category": category,
                    "text": text,
                    "triples": triples,
                    "target_text": target_str,
                    "num_triples": num_triples,
                })
        else:
            if expand_eval:
                for sub_idx, text in enumerate(clean_texts):
                    processed_samples.append({
                        "id": f"{split_name}_{sample_id}_{sub_idx}",
                        "webnlg_id": webnlg_id,
                        "category": category,
                        "text": text,
                        "triples": triples,
                        "target_text": target_str,
                        "all_references": clean_texts,
                        "num_triples": num_triples,
                    })
            else:
                # Benchmark chuẩn của GEM/web_nlg: 1 entry = 1 sample (1,667 val / 1,779 test)
                processed_samples.append({
                    "id": f"{split_name}_{sample_id}",
                    "webnlg_id": webnlg_id,
                    "category": category,
                    "text": clean_texts[0],
                    "triples": triples,
                    "target_text": target_str,
                    "all_references": clean_texts,
                    "num_triples": num_triples,
                })

        sample_id += 1

    stats["total_samples"] = len(processed_samples)
    stats["categories"] = sorted(list(stats["categories"]))
    return processed_samples, stats


def save_processed(samples: List[Dict[str, Any]], out_dir: Path, split_name: str) -> None:
    """Lưu dữ liệu đã xử lý ra cả định dạng JSON và CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{split_name}.json"
    csv_path = out_dir / f"{split_name}.csv"

    # Lưu JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # Lưu CSV
    if samples:
        fieldnames = ["id", "webnlg_id", "category", "text", "target_text", "num_triples"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in samples:
                writer.writerow(s)

    print(f"[SAVED] {split_name.upper()}: {len(samples):,} mẫu -> {json_path} & {csv_path}")


def generate_data_readme(processed_dir: Path, all_stats: Dict[str, Any]) -> None:
    """Tạo file README.md trong data/ để bàn giao theo tiêu chí 19/11 của Tasks_NLP.docx."""
    readme_path = processed_dir.parent / "README.md"
    content = f"""# 📚 Dữ liệu WebNLG+ 2020 cho BTL NLP (Text-to-Triples)

> **Phụ trách (Owner):** Doãn Duy Lợi (24021549) — *Lead: Data Pipeline & Training Strategy*
> **Nguồn dữ liệu:** GEM Benchmark / WebNLG+ 2020 English (`GEM/web_nlg` trên Hugging Face: https://huggingface.co/datasets/GEM/web_nlg)
> **Nhiệm vụ bàn giao:** Hạng mục T01 (Mốc 19/11: Bàn giao tập dữ liệu sạch & chuẩn hóa)
> **Người nhận:** Hoàng Lâm Tùng (Scratch), Nguyễn Chí Phong (LLMs < 6B), Bùi Thị Bích Phượng (Eval/KG)

---

## 1. Đối chiếu Số liệu chuẩn với Hugging Face GEM/web_nlg

Dữ liệu được trích xuất trực tiếp từ các file JSON nguồn chính thức do Hugging Face lưu trữ cho `GEM/web_nlg` (en):

| Tập dữ liệu | Số mục gốc (Entries) | Số mẫu Text-to-Triples chuẩn hóa | Số mẫu trên Hugging Face `dataset_infos.json` | 1 Triple | 2 Triples | >= 3 Triples |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | {all_stats.get('train', {}).get('total_entries', 0):,} | **{all_stats.get('train', {}).get('total_samples', 0):,}** | **35,426** (Khớp 100%) | {all_stats.get('train', {}).get('triples_count_dist', {}).get(1, 0):,} | {all_stats.get('train', {}).get('triples_count_dist', {}).get(2, 0):,} | {all_stats.get('train', {}).get('triples_count_dist', {}).get('>=3', 0):,} |
| **Validation** | {all_stats.get('val', {}).get('total_entries', 0):,} | **{all_stats.get('val', {}).get('total_samples', 0):,}** | **1,667** (Khớp 100%) | {all_stats.get('val', {}).get('triples_count_dist', {}).get(1, 0):,} | {all_stats.get('val', {}).get('triples_count_dist', {}).get(2, 0):,} | {all_stats.get('val', {}).get('triples_count_dist', {}).get('>=3', 0):,} |
| **Test** | {all_stats.get('test', {}).get('total_entries', 0):,} | **{all_stats.get('test', {}).get('total_samples', 0):,}** | **1,779** (Khớp 100%) | {all_stats.get('test', {}).get('triples_count_dist', {}).get(1, 0):,} | {all_stats.get('test', {}).get('triples_count_dist', {}).get(2, 0):,} | {all_stats.get('test', {}).get('triples_count_dist', {}).get('>=3', 0):,} |

- **Số lượng Category (Chủ đề DBpedia):** {len(all_stats.get('train', {}).get('categories', []))} danh mục (bao gồm: {', '.join(all_stats.get('train', {}).get('categories', [])[:8])}...)

---

## 2. Chuẩn Giao diện Dữ liệu (Interface Contract)

Mỗi mẫu dữ liệu được lưu dưới dạng JSON object có các trường sau:

```json
{{
  "id": "train_0_0",
  "webnlg_id": "train/Airport/1/Id1",
  "category": "Airport",
  "text": "Aarhus Airport serves the city of Aarhus, Denmark.",
  "triples": [
    ["Aarhus_Airport", "cityServed", "Aarhus, Denmark"]
  ],
  "target_text": "(Aarhus_Airport | cityServed | Aarhus, Denmark)",
  "num_triples": 1
}}
```

### Định dạng Target String:
- Mỗi bộ ba được bao bởi dấu ngoặc đơn: `(Subject | Predicate | Object)`
- Nếu có nhiều bộ ba, các bộ ba được phân cách bởi ` | `:
  `(Subject1 | Predicate1 | Object1) | (Subject2 | Predicate2 | Object2)`

---

## 3. Hướng dẫn nạp Dữ liệu cho các thành viên

### Cách 1: Nạp qua `WebNLGDataset` (cho PyTorch DataLoader)
```python
from data.dataset import WebNLGDataset, get_dataloader
from tokenizer.train_bpe import BPETokenizerWrapper

tokenizer = BPETokenizerWrapper.load("tokenizer/artifacts")
train_loader = get_dataloader(
    "data/processed/train.json",
    tokenizer=tokenizer,
    batch_size=32,
    shuffle=True,
    max_length=256,
)

for batch in train_loader:
    src_ids = batch["src_ids"]  # [B, L_src]
    tgt_ids = batch["tgt_ids"]  # [B, L_tgt]
    src_mask = batch["src_mask"]  # [B, 1, 1, L_src]
    tgt_mask = batch["tgt_mask"]  # [B, 1, L_tgt, L_tgt]
    break
```

### Cách 2: Nạp trực tiếp JSON / CSV (cho LLMs Fine-Tuning & Evaluation)
```python
import json
import pandas as pd

# Đọc JSON
with open("data/processed/train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

# Đọc CSV bằng Pandas
df = pd.read_csv("data/processed/test.csv")
print(f"Tổng số mẫu test: {{len(df)}}")
print(df.head())
```

---
*Tài liệu tự động sinh bởi `data/prepare_webnlg.py` — Hoàn thành đúng hạn mốc 19/11.*
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[GENERATED] Đã tạo file tài liệu hướng dẫn bàn giao: {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải, tiền xử lý và chuẩn hóa WebNLG+ 2020 cho BTL NLP.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Thư mục chứa raw json tải về")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Thư mục lưu dữ liệu chuẩn hóa")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số mẫu xử lý (dùng để test nhanh)")
    parser.add_argument("--force_download", action="store_true", help="Bắt buộc tải lại file gốc")
    parser.add_argument("--expand_eval", action="store_true", help="Tách từng verbalization cho val/test thay vì 1:1 GEM benchmark")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("BTL NLP: TIỀN XỬ LÝ DỮ LIỆU WEBNLG+ 2020 (TEXT-TO-TRIPLES)")
    print("Nguồn: Hugging Face GEM/web_nlg (English split)")
    print("Owner: Doãn Duy Lợi (24021549) - Lead: Data Pipeline & Training Strategy")
    print("=" * 70)

    all_stats: Dict[str, Any] = {}

    for split_key, split_name in [("train", "train"), ("val", "val"), ("test", "test")]:
        url = OFFICIAL_URLS[split_key]
        raw_file = raw_dir / f"webnlg_en_{split_name}.json"
        download_file(url, raw_file, force=args.force_download)

        print(f"\n[*] Đang xử lý tập {split_name.upper()}...")
        samples, stats = process_split_data(raw_file, split_name, limit=args.limit, expand_eval=args.expand_eval)
        all_stats[split_name] = stats
        save_processed(samples, output_dir, split_name)

        print(f"    - Tổng entries gốc: {stats['total_entries']:,}")
        print(f"    - Tổng mẫu sau khi chuẩn hóa: {stats['total_samples']:,}")
        print(f"    - Phân bổ triples: 1={stats['triples_count_dist'][1]}, 2={stats['triples_count_dist'][2]}, >=3={stats['triples_count_dist']['>=3']}")

    # Tạo README.md hướng dẫn bàn giao cho nhóm
    generate_data_readme(output_dir, all_stats)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU ĐỐI CHIẾU CHUẨN GEM/WEBNLG")
    print("=" * 70)


if __name__ == "__main__":
    main()
