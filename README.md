# End-to-End Knowledge Graph Extraction from Unstructured Text (Text-to-Triples)

> **Học phần:** Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)  
> **Đề tài:** Comparative Study of From-Scratch Transformers and Parameter-Efficient Fine-Tuning of Small LLMs on WebNLG+ 2020  
> **Thời gian:** 4 tuần (16/11/2026 – 13/12/2026)  
> **Nhóm thực hiện:**
> 1. **Doãn Duy Lợi (24021549)** — *Lead: Data Pipeline & Training Strategy*
> 2. **Hoàng Lâm Tùng (24021661)** — *Lead: Transformer From-Scratch Core*
> 3. **Nguyễn Chí Phong (24020268)** — *Lead: Pretrained LLMs (< 6B) Fine-tuning*
> 4. **Bùi Thị Bích Phượng (24021601)** — *Lead: Evaluation & KG Fusion*

---

## 📌 Cấu trúc Dự án

```text
BTL_NLP/
├── configs/
│   └── config.yaml               # Cấu hình siêu tham số chung (seed=42, paths, model configs)
├── data/
│   ├── prepare_webnlg.py         # Script tải và tiền xử lý WebNLG+ 2020
│   ├── dataset.py                # WebNLGDataset & dynamic padding collate_fn
│   ├── README.md                 # Hướng dẫn chi tiết nạp dữ liệu cho các nhánh
│   └── processed/                # {train, val, test}.{json, csv} đã chuẩn hóa
├── tokenizer/
│   ├── train_bpe.py              # Huấn luyện Custom Byte-Level BPE trên train split
│   └── artifacts/                # tokenizer.json, vocab.json, tokenizer_config.json
├── models/
│   ├── scratch/                  # Nhánh 1: Transformer PyTorch thuần (6 enc/6 dec)
│   └── llm_under6b/              # Nhánh 2: Pipeline SFT QLoRA 4-bit cho LLMs < 6B
├── eval/
│   ├── parser.py                 # Triplet Parser & Chuẩn hóa định dạng (s | p | o)
│   ├── metrics.py                # Micro/Macro F1, Exact Match, Unparseable Rate
│   └── hallucination_filter.py   # Lọc thực thể ảo giác
├── kg_fusion/
│   ├── fusion.py                 # Hợp nhất đồ thị tri thức
│   └── visualize.py              # Trực quan hóa Knowledge Graph (.gexf / .json)
├── train_scratch.py              # Vòng lặp huấn luyện Transformer Scratch (Teacher Forcing, AdamW)
├── decode.py                     # Thuật toán Greedy Search & Beam Search
├── docs/
│   ├── Tasks_NLP.docx            # Kế hoạch phân công nhiệm vụ chi tiết 4 tuần
│   ├── BTL_NLP_Proposal_Report.md# Đề cương kỹ thuật chi tiết
│   └── WEEKLY_PROGRESS.md        # Bảng theo dõi tiến độ công việc hàng tuần
└── tests/
    └── test_data_pipeline.py     # Bộ unit test tự động cho dữ liệu, tokenizer & training
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Cài đặt Môi trường
```bash
pip install -r requirements.txt
```

### 2. Tiền xử lý Dữ liệu WebNLG+ 2020 (Hạng mục T01 - Doãn Duy Lợi)
```bash
python data/prepare_webnlg.py
```
- Dữ liệu được tải từ nguồn chính thức của Hugging Face GEM/web_nlg.
- Chuyển đổi sang tác vụ Text-to-Triples: Đầu vào là văn bản tiếng Anh, đầu ra là danh sách bộ ba `(Subject | Predicate | Object)`.
- Kết quả được lưu tại `data/processed/train.json`, `val.json`, `test.json` (và định dạng CSV tương ứng).

### 3. Huấn luyện Custom BPE Tokenizer (Hạng mục T02 - Doãn Duy Lợi)
```bash
python tokenizer/train_bpe.py --vocab_size 16000
```
- Huấn luyện thuật toán Byte-Level BPE **CHỈ trên tập train split** (chống rò rỉ dữ liệu).
- Special tokens: `[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`, `[SEP]=4`.
- Artifacts được lưu tại `tokenizer/artifacts/`.

### 4. Huấn luyện Transformer From-Scratch (Hạng mục T04 - Phối hợp)
```bash
# Chạy thử nghiệm nhanh (Dry run)
python train_scratch.py --dry_run

# Chạy huấn luyện chính thức (Cấu hình trong configs/config.yaml)
python train_scratch.py --config configs/config.yaml
```
- Tự động lưu checkpoint tốt nhất vào `outputs/checkpoints/scratch/best_model.pt`.
- Ghi nhật ký huấn luyện tại `outputs/checkpoints/scratch/train_log.json`.

### 5. Chạy Toàn bộ Unit Tests
```bash
python -m pytest tests/test_data_pipeline.py -v
```

---

## 📄 Chuẩn Giao diện Dữ liệu (Interface Contract)
Mọi mô hình (Transformer From-Scratch & 02 Pretrained LLMs < 6B) sinh đầu ra chuẩn:
```text
(Subject | Predicate | Object)
```
Nếu mẫu có nhiều bộ ba:
```text
(Subject1 | Predicate1 | Object1) | (Subject2 | Predicate2 | Object2)
```
Mỗi bộ ba được tách biệt bởi ` | ` hoặc dòng mới, giúp bộ đánh giá `eval/parser.py` dễ dàng phân tách và tính toán metric mà không bị sai lệch cú pháp.
