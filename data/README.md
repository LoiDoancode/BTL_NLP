# 📚 Dữ liệu WebNLG+ 2020 cho BTL NLP (Text-to-Triples)

> **Phụ trách (Owner):** Doãn Duy Lợi (24021549) — *Lead: Data Pipeline & Training Strategy*
> **Nguồn dữ liệu:** GEM Benchmark / WebNLG+ 2020 English (`GEM/web_nlg` trên Hugging Face: https://huggingface.co/datasets/GEM/web_nlg)
> **Nhiệm vụ bàn giao:** Hạng mục T01 (Mốc 19/11: Bàn giao tập dữ liệu sạch & chuẩn hóa)
> **Người nhận:** Hoàng Lâm Tùng (Scratch), Nguyễn Chí Phong (LLMs < 6B), Bùi Thị Bích Phượng (Eval/KG)

---

## 1. Đối chiếu Số liệu chuẩn với Hugging Face GEM/web_nlg

Dữ liệu được trích xuất trực tiếp từ các file JSON nguồn chính thức do Hugging Face lưu trữ cho `GEM/web_nlg` (en):

| Tập dữ liệu | Số mục gốc (Entries) | Số mẫu Text-to-Triples chuẩn hóa | Số mẫu trên Hugging Face `dataset_infos.json` | 1 Triple | 2 Triples | >= 3 Triples |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 13,211 | **35,426** | **35,426** (Khớp 100%) | 3,195 | 2,472 | 7,544 |
| **Validation** | 1,667 | **1,667** | **1,667** (Khớp 100%) | 403 | 313 | 951 |
| **Test** | 1,779 | **1,779** | **1,779** (Khớp 100%) | 369 | 349 | 1,061 |

- **Số lượng Category (Chủ đề DBpedia):** 16 danh mục (bao gồm: Airport, Artist, Astronaut, Athlete, Building, CelestialBody, City, ComicsCharacter...)

---

## 2. Chuẩn Giao diện Dữ liệu (Interface Contract)

Mỗi mẫu dữ liệu được lưu dưới dạng JSON object có các trường sau:

```json
{
  "id": "train_0_0",
  "webnlg_id": "train/Airport/1/Id1",
  "category": "Airport",
  "text": "Aarhus Airport serves the city of Aarhus, Denmark.",
  "triples": [
    ["Aarhus Airport", "cityServed", "Aarhus, Denmark"]
  ],
  "target_text": "(Aarhus Airport | cityServed | Aarhus, Denmark)",
  "num_triples": 1
}
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
    src_ids = batch["src_ids"]
    tgt_ids = batch["tgt_ids"]
    src_mask = batch["src_mask"]
    tgt_mask = batch["tgt_mask"]
    break
```

### Cách 2: Nạp trực tiếp JSON / CSV (cho LLMs Fine-Tuning & Evaluation)
```python
import json
import pandas as pd

with open("data/processed/train.json", encoding="utf-8") as f:
    train_data = json.load(f)

df = pd.read_csv("data/processed/test.csv")
print(f"Tổng số mẫu test: {len(df)}")
print(df.head())
```

---
*Tài liệu tự động sinh bởi `data/prepare_webnlg.py` — Hoàn thành đúng hạn mốc 19/11.*
