# 📋 Theo dõi Tiến độ Nhiệm vụ Hàng tuần — BTL NLP

> **Đề tài:** Xây dựng Đồ thị Tri thức từ Văn bản (Text-to-Triples) trên tập dữ liệu WebNLG+ 2020 (`GEM/web_nlg`)  
> **Thời gian thực hiện:** 4 tuần (`11/11/2026` – `08/12/2026`)  
> **Tài liệu tham khảo:** [BTL_NLP_Proposal_Report.md](./BTL_NLP_Proposal_Report.md)

---

## 👥 Danh sách Thành viên & Vai trò

| STT | Họ và tên | MSSV | Vai trò chính (Lead) |
|:---:|-----------|------|----------------------|
| 1 | **Doãn Duy Lợi** | `24021549` | Data Pipeline & Training Strategy |
| 2 | **Hoàng Lâm Tùng** | `24021661` | Transformer From-Scratch Core |
| 3 | **Nguyễn Chí Phong** | `24020268` | Pretrained LLMs Fine-tuning |
| 4 | **Bùi Thị Bích Phượng** | `24021601` | Evaluation & KG Fusion |

---

## 🏷️ Quy ước Trạng thái

- ⬜ **Chưa bắt đầu** (`Not Started`)
- 🔄 **Đang thực hiện** (`In Progress`)
- ✅ **Hoàn thành** (`Done`)
- ⚠️ **Vướng mắc / Cần hỗ trợ** (`Blocked`)

---

## 📊 Tổng quan Tiến độ Theo Tuần

| Tuần | Thời gian | Giai đoạn | Mục tiêu chính | Trạng thái chung | Tiến độ (%) |
|:----:|-----------|-----------|----------------|:----------------:|:-----------:|
| **Tuần 1** | `11/11` – `17/11` | Setup & From-Scratch | Tiền xử lý WebNLG+ 2020, BPE Tokenizer, dựng lõi Transformer Scratch & Baseline | ✅ Hoàn thành | 100% |
| **Tuần 2** | `18/11` – `24/11` | SFT 2 LLMs & Cải tiến | Pipeline QLoRA (`Qwen3-4B`, `Llama-3.1-Nemotron-Nano-4B`), Pre-LN, Weight Tying, CoT Prompting | 🔄 Đang thực hiện | 50% |
| **Tuần 3** | `25/11` – `01/12` | Đánh giá & KG Fusion | Đo Micro/Macro P/R/F1, Exact-Set, lọc ảo giác, hợp nhất KG, Error Analysis | ⬜ Chưa bắt đầu | 0% |
| **Tuần 4** | `02/12` – `08/12` | Báo cáo & Thuyết trình | Hoàn thiện báo cáo IEEE 6 trang (EN), Slide bảo vệ, chuẩn hóa GitHub repo | ⬜ Chưa bắt đầu | 0% |

---

## 📅 Tuần 1 (`11/11` – `17/11`): Setup & Transformer From-Scratch

**Đầu ra cần đạt (Deliverables):** Checkpoint Baseline Transformer, Log loss huấn luyện, Tokenizer artifact, Khung repo GitHub.

### 1. Doãn Duy Lợi (`24021549`) — *Data Pipeline & Training Strategy*
- [x] Khởi tạo cấu trúc mã nguồn chuẩn trên GitHub repo & cập nhật `README.md`.
- [x] Tải, làm sạch và chuẩn hóa tập dữ liệu WebNLG+ 2020 (`GEM/web_nlg`).
- [x] Huấn luyện Custom BPE Tokenizer (chỉ sử dụng tập `train` split) và lưu artifact.
- [x] Xây dựng khung vòng lặp huấn luyện (`train_scratch.py`): Teacher Forcing, Cross-Entropy Loss, AdamW + Warmup Scheduler.

### 2. Hoàng Lâm Tùng (`24021661`) — *Transformer From-Scratch Core*
- [x] Cài đặt `ScaledDotProductAttention` & `MultiHeadAttention` bằng PyTorch thuần (không dùng `torch.nn.Transformer`).
- [x] Cài đặt `PositionalEncoding` (Sinusoidal) & `FeedForwardNetwork`.
- [x] Cài đặt `Padding Mask` và `Causal Mask`.
- [x] Ghép hoàn chỉnh `EncoderLayer`, `DecoderLayer`, `TransformerEncoder`, `TransformerDecoder` bản Baseline.

### 3. Nguyễn Chí Phong (`24020268`) — *Pretrained LLMs Fine-tuning*
- [x] Chuẩn bị môi trường chạy QLoRA 4-bit (RTX 4090 / Colab T4) & khảo sát VRAM cho `Qwen/Qwen3-4B-Instruct-2507` (4.02B) và `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` (4.51B).
- [x] Thiết kế định dạng Prompt chuẩn (Input text $\rightarrow$ Structured Triples) & kiểm tra tokenization trên 2 LLM.
- [x] Chạy thử nghiệm baseline inference (few-shot) để làm mốc đối chiếu trước SFT.

### 4. Bùi Thị Bích Phượng (`24021601`) — *Evaluation & KG Fusion*
- [x] Xây dựng bộ parser chuyển đổi chuỗi output sinh ra thành danh sách bộ ba `(subject, predicate, object)`.
- [x] Cài đặt bộ tính toán metric chuẩn: Full-Triple Micro/Macro Precision, Recall, F1 và Exact-Set Accuracy.
- [x] Kiểm thử unit test cho parser & evaluator trên các mẫu ground-truth của WebNLG+.

### 📝 Tổng kết Tuần 1
| Thành viên | Trạng thái | % Hoàn thành | Sản phẩm / Link Commit | Ghi chú / Vướng mắc |
|------------|:----------:|:------------:|------------------------|---------------------|
| Doãn Duy Lợi | ✅ | 100% | `data/prepare_webnlg.py`, `tokenizer/train_bpe.py`, `train_scratch.py`, `data/README.md` | Hoàn thành sớm trước hạn bàn giao 19/11 & 21/11 |
| Hoàng Lâm Tùng | ✅ | 100% | `models/scratch/*`, `tests/test_scratch_core.py` | Đã hoàn thiện core Transformer, Pre-LN, Weight Tying, unit tests & tương thích `train_scratch.py` |
| Nguyễn Chí Phong | ✅ | 100% | `configs/config.yaml`, `models/llm_under6b/*` (`50801c7`) | Chốt 2 LLM `< 6B` (`Qwen3-4B-Instruct-2507` & `Llama-3.1-Nemotron-Nano-4B-v1.1`), đo VRAM 4-bit trên RTX 4090 & chạy baseline few-shot (`Qwen3-4B` Micro F1 = 0.2706) |
| Bùi Thị Bích Phượng | ✅ | 100% | `eval/parser.py`, `eval/metrics.py`, `eval/hallucination_filter.py`, `tests/test_eval_parser.py`, `tests/test_metrics.py` (`267f79d`) | Đã hoàn thiện parser, metrics, hallucination filter & pass toàn bộ unit tests |

---

## 📅 Tuần 2 (`18/11` – `24/11`): SFT 2 LLMs & Cải tiến Kiến trúc

**Đầu ra cần đạt (Deliverables):** Adapters LoRA đã lưu (`Qwen2.5-3B` & `Llama-3.2-3B`), Checkpoint Transformer cải tiến, Bảng loss so sánh.

### 1. Doãn Duy Lợi (`24021549`) — *Data Pipeline & Training Strategy*
- [x] Hoàn thiện module giải mã tự hồi quy: Greedy Search và Beam Search ($B=4$).
- [x] Triển khai chiến lược Curriculum Learning (lập lịch huấn luyện theo độ phức tạp 1, 2, $\ge 3$ triples).
- [x] Chạy huấn luyện Transformer cải tiến & lưu log loss so sánh với bản Baseline.

### 2. Hoàng Lâm Tùng (`24021661`) — *Transformer From-Scratch Core*
- [ ] Tích hợp cải tiến **Pre-LayerNorm (Pre-LN)** vào các khối Encoder/Decoder.
- [ ] Tích hợp cải tiến **Weight Tying** giữa embedding và linear projection đầu ra.
- [ ] Kiểm tra gradient flow, ổn định hội tụ và hỗ trợ debug huấn luyện cùng Data/Training Lead.

### 3. Nguyễn Chí Phong (`24020268`) — *Pretrained LLMs Fine-tuning*
- [x] Xây dựng pipeline SFT & QLoRA hoàn chỉnh cho `Qwen/Qwen3-4B-Instruct-2507` và `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1`.
- [x] Triển khai kỹ thuật **Target-only Loss Masking** (chỉ tính loss trên phần output triples).
- [x] Thử nghiệm **Entity-Guided CoT Prompting** & tinh chỉnh siêu tham số LoRA ($r$, $\alpha$, lr, scheduler).
- [x] Lưu lại các LoRA adapters tốt nhất & ghi nhận chi phí phần cứng (Peak VRAM, thời gian train).

### 4. Bùi Thị Bích Phượng (`24021601`) — *Evaluation & KG Fusion*
- [ ] Xây dựng **Entity Hallucination Filter** (bộ lọc loại bỏ bộ ba chứa thực thể/quan hệ bịa đặt).
- [ ] Thiết kế và cài đặt module **Knowledge Graph Fusion** (hợp nhất bộ ba từ nhiều câu, loại bỏ trùng lặp).
- [ ] Chạy đánh giá sơ bộ trên checkpoint Tuần 1 & các kết quả đầu ra đầu tiên của Tuần 2.

### 📝 Tổng kết Tuần 2
| Thành viên | Trạng thái | % Hoàn thành | Sản phẩm / Link Commit | Ghi chú / Vướng mắc |
|------------|:----------:|:------------:|------------------------|---------------------|
| Doãn Duy Lợi | ✅ | 100% | `decode.py`, `train_scratch.py`, `scripts/compare_scratch_runs.py`, `tests/test_decode_and_curriculum.py`, `outputs/checkpoints/scratch_comparison.json` | Đã hoàn thiện Greedy & Beam Search ($B=4$), Curriculum Learning 3 giai đoạn (1 → 1-2 → full), và chạy so sánh Baseline vs Improved (`val_loss` giảm `9.3493` → `8.8163`, `-5.70%`) |
| Hoàng Lâm Tùng | ⬜ | 0% | | |
| Nguyễn Chí Phong | ✅ | 100% | `scripts/sft_qlora.py`, `scripts/compare_llm_runs.py`, `outputs/checkpoints/llm_comparison.json` | Hoàn tất SFT QLoRA full `35,426` mẫu trên RTX 4090: `Qwen3-4B_cot` tốt nhất (`val_loss=0.01298`, Micro F1 `0.5683`, `13.45 GB` VRAM), `Qwen3-4B_std` (`0.5584`), `Qwen3-4B_r16` (`0.5536`), `Nemotron-4B_std` (`0.5411`) |
| Bùi Thị Bích Phượng | ⬜ | 0% | | |

---

## 📅 Tuần 3 (`25/11` – `01/12`): Đánh giá Toàn diện & KG Fusion

**Đầu ra cần đạt (Deliverables):** Bảng số liệu thực nghiệm đầy đủ (3 mô hình), Biểu đồ KG trực quan (`.gexf`/`.json`), Báo cáo Error Analysis.

### 1. Doãn Duy Lợi (`24021549`) — *Data Pipeline & Training Strategy*
- [ ] Chốt kết quả cuối cùng của mô hình Transformer From-Scratch (điều chỉnh Beam Search / Decoding nếu cần).
- [ ] Tổng hợp đồ thị Training/Validation Loss của Transformer Scratch (Baseline vs. Pre-LN + Weight Tying + Curriculum).
- [ ] Đóng băng code data pipeline & chuẩn bị script tái lập kết quả (`reproduce`).

### 2. Hoàng Lâm Tùng (`24021661`) — *Transformer From-Scratch Core*
- [ ] Thực hiện ablation study nhỏ (nếu cần) cho các thành phần Pre-LN / Weight Tying.
- [ ] Vẽ sơ đồ khối kiến trúc Transformer Pre-LN chuẩn để đưa vào báo cáo IEEE & Slide.
- [ ] Review code lõi Transformer, bổ sung docstring và type hints rõ ràng.

### 3. Nguyễn Chí Phong (`24020268`) — *Pretrained LLMs Fine-tuning*
- [ ] Sinh dự đoán trên toàn bộ tập test WebNLG+ 2020 cho `Qwen2.5-3B-Instruct` và `Llama-3.2-3B-Instruct`.
- [ ] Hoàn thiện bảng đo đạc tài nguyên phần cứng (Peak GPU VRAM, training time, inference latency).
- [ ] Tổng hợp bảng siêu tham số tối ưu của 2 mô hình để đưa vào Mục IV.B & IV.C.

### 4. Bùi Thị Bích Phượng (`24021601`) — *Evaluation & KG Fusion*
- [ ] Chạy đánh giá tổng hợp 3 mô hình (From-Scratch vs. Qwen2.5-3B vs. Llama-3.2-3B): Micro/Macro P, R, F1, Exact-Set.
- [ ] Thống kê phân tầng theo số lượng bộ ba (`1`, `2`, `>= 3` triples) và theo domain/category.
- [ ] Xuất đồ thị tri thức hoàn chỉnh sang định dạng `.gexf` / `.json` và tạo hình ảnh trực quan hóa KG.
- [ ] Thực hiện **Error Analysis** định tính (hallucination, lỗi cú pháp parse, thiếu quan hệ).

### 📝 Tổng kết Tuần 3
| Thành viên | Trạng thái | % Hoàn thành | Sản phẩm / Link Commit | Ghi chú / Vướng mắc |
|------------|:----------:|:------------:|------------------------|---------------------|
| Doãn Duy Lợi | ⬜ | 0% | | |
| Hoàng Lâm Tùng | ⬜ | 0% | | |
| Nguyễn Chí Phong | ⬜ | 0% | | |
| Bùi Thị Bích Phượng | ⬜ | 0% | | |

---

## 📅 Tuần 4 (`02/12` – `08/12`): Báo cáo IEEE & Slide Thuyết trình

**Đầu ra cần đạt (Deliverables):** File PDF báo cáo IEEE (tiếng Anh, $\ge 6$ trang), Slide `.pptx`, GitHub repo chuẩn hóa có thể tái lập.

### 1. Doãn Duy Lợi (`24021549`) — *Data Pipeline & Training Strategy*
- [ ] Viết **Section I (Abstract)**, **Section II (Introduction)**, **Section III (Related Work)** báo cáo IEEE.
- [ ] Chuẩn hóa toàn bộ cấu trúc GitHub repo (seed cố định, `requirements.txt`, script chạy end-to-end, `README.md`).
- [ ] Thiết kế Slide phần Giới thiệu bài toán & Dữ liệu.

### 2. Hoàng Lâm Tùng (`24021661`) — *Transformer From-Scratch Core*
- [ ] Viết **Section IV.A (Methodology — From-Scratch Transformer)** với công thức toán học & sơ đồ khối Pre-LN.
- [ ] Thiết kế Slide phần Kiến trúc Transformer Scratch.

### 3. Nguyễn Chí Phong (`24020268`) — *Pretrained LLMs Fine-tuning*
- [ ] Viết **Section IV.B & IV.C (Pretrained LLMs, QLoRA, Loss Masking & Prompting)** kèm bảng siêu tham số & chi phí GPU.
- [ ] Thiết kế Slide phần Fine-tuning Pretrained LLMs.

### 4. Bùi Thị Bích Phượng (`24021601`) — *Evaluation & KG Fusion*
- [ ] Viết **Section V (Experiments & Results)**, **Section VI (Discussion & Error Analysis)**, **Section VII (Conclusion)**.
- [ ] Thiết kế Slide phần Thực nghiệm, Phân tích lỗi & Demo KG.

### 🤝 Công việc chung cả nhóm (Tuần 4)
- [ ] Rà soát chéo toàn bộ bài báo IEEE (format 2 cột, citation, hình ảnh, độ dài tối thiểu 6 trang).
- [ ] Ghép và thống nhất giao diện Slide thuyết trình (`.pptx`).
- [ ] Tổng duyệt thuyết trình thử (dry-run) và kiểm tra demo.

### 📝 Tổng kết Tuần 4
| Thành viên | Trạng thái | % Hoàn thành | Sản phẩm / Link Commit | Ghi chú / Vướng mắc |
|------------|:----------:|:------------:|------------------------|---------------------|
| Doãn Duy Lợi | ⬜ | 0% | | |
| Hoàng Lâm Tùng | ⬜ | 0% | | |
| Nguyễn Chí Phong | ⬜ | 0% | | |
| Bùi Thị Bích Phượng | ⬜ | 0% | | |

---

## 🗒️ Nhật ký Họp nhóm & Vướng mắc (Blockers)

| Ngày | Nội dung trao đổi chính | Vướng mắc (Blockers) | Hướng xử lý / Người phụ trách |
|------|--------------------------|----------------------|-------------------------------|
| `15/11/2026` | Khởi tạo file theo dõi tiến độ tuần cho 4 thành viên | — | Cả nhóm cập nhật checklist sau mỗi mốc công việc |
