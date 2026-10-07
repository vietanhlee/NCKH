# Hướng 3: Context-Aware Weak Supervision — Gộp Nhãn Yếu Đa Nguồn Cho Giám Sát Giao Thông Đô Thị

> **Paper Title Candidate:** *Context-Aware Markov Label Aggregation: Weakly-Supervised Traffic Congestion Assessment from Imperfect Heuristics on City-Scale Surveillance Networks*  
> **Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / CVPR / ECCV / NeurIPS  

---

## 1. Giới thiệu & Động lực Khoa học
Trong hệ thống giám sát giao thông đô thị quy mô lớn (như TP.HCM với hàng ngàn camera), việc gán nhãn thủ công (ground truth) cho từng khung hình camera 24/7 là bất khả thi về mặt nhân lực và chi phí. Tuy nhiên, chúng ta có sẵn nhiều **nguồn tín hiệu yếu (weak heuristics / labeling functions - LF)**:
1. **LF1 (Detector Bounding Box Area):** Tỷ lệ diện tích xe so với diện tích mặt đường $R_i$.
2. **LF2 (Background Difference Ratio):** Chênh lệch tuyệt đối trung bình so với ảnh nền $\bar{\Delta}_t$.
3. **LF3 (Temporal Frame Differencing):** Biến động quang thông/sai khác giữa các frame liên tiếp $|\mathbf{I}_t - \mathbf{I}_{t-1}|$ (đo vận tốc lưu thông xấp xỉ).
4. **LF4 (Historical Peak Profile):** Hồ sơ lịch sử thống kê theo giờ/ngày trong tuần.
5. **LF5 (Multimodal VLM - Gemini/Qwen2.5-VL):** Đánh giá mức độ qua truy vấn ngôn ngữ thị giác (chỉ chạy lấy mẫu do chi phí tính toán).

**Thách thức cốt lõi:**
- Các nguồn nhãn yếu này đều có **sai số hệ thống**: LF2 bị lỗi bóng đổ và ghost vehicle khi kẹt xe kéo dài; LF1 bỏ sót xe do góc khuất hoặc lóa đèn đêm; LF4 sai khi trời mưa bất chợt; LF5 đắt và đôi khi hallucinate.
- Các LF có độ tin cậy **thay đổi phụ thuộc mạnh vào Ngữ cảnh (Context)**: LF1 rất tốt ban ngày nhưng kém ban đêm; LF2 tốt khi nền sạch nhưng hỏng khi camera bị lệch góc.
- Trạng thái kẹt xe có **tính liên tục thời gian (Markovian Dynamics)**: Tắc đường không xuất hiện và biến mất trong 1 giây, mà chuyển dịch trạng thái dần dần ($0 \to 1 \to 2 \to 3$).

---

## 2. Kiến trúc Phương pháp

### 2.1. Phân loại Ngữ cảnh (Context Extraction)
Không gian ngữ cảnh được phân nhỏ thành 54 tổ hợp:
- **Ánh sáng:** Ngày, Chiều tối, Đêm (hỗ trợ phân biệt chế độ hồng ngoại IR / lóa đèn).
- **Khung giờ:** Cao điểm sáng, Cao điểm chiều, Thấp điểm ban ngày, Đêm muộn.
- **Phân loại đường:** Trục lộ lớn (High-capacity), Đường hẹp nội đô.
- **Tình trạng Camera:** Độ tin cậy nền $r_i$ (Cao / Thấp).

### 2.2. Context-Aware Markov Label Model
- **Biến ẩn:** $y_t \in \{0, 1, 2, 3\}$ (0: Thông thoáng, 1: Đông nhẹ, 2: Ùn ứ, 3: Kẹt xe nghiêm trọng).
- **Chuyển trạng thái:** Ma trận Markov $\mathbf{A} \in \mathbb{R}^{4 \times 4}$, khởi tạo đường chéo 0.85 (tính trơn liên tục).
- **Hàm phát xạ theo ngữ cảnh:** $\pi_j^{(c)}(\lambda_j \mid y) = P(\lambda_j \mid y, \text{context } c)$.
- **Cơ chế Abstain:** $\lambda_j = -1$ khi nguồn không chắc chắn $\implies$ không đóng góp vào likelihood.
- **Thuật toán EM:** Tối ưu hóa kỳ vọng cực đại (Expectation-Maximization) với thuật toán **Forward-Backward trong không gian log-sum-exp** (ngăn chặn underflow tràn số).

### 2.3. End Model (DINOv3 + Causal GRU)
- Nhãn mềm đầu ra $q(y_t) = [q_0, q_1, q_2, q_3]$ được dùng để huấn luyện **End Model**.
- Backbone: Meta DINOv3 ViT trích xuất đặc trưng thị giác cấp cao.
- Temporal: Causal 1-way GRU (chỉ nhìn về quá khứ, không rò rỉ tương lai).
- Hàm mất mát: **Soft Cross-Entropy** với ngưỡng tự tin $\max(q) \ge 0.40$.
- Khi triển khai thực tế, End Model **tự chủ 100%**, chỉ nhận ảnh camera đầu vào mà **không phụ thuộc vào bất kỳ LF hay background nào**.

---

## 3. Cấu trúc Thư mục
```
direction3_weak_supervision/
├── lfs/
│   ├── lf_detector.py      # LF1: Tỷ lệ diện tích phát hiện xe
│   ├── lf_background.py    # LF2: Sai khác ảnh nền có cổng tin cậy r_i
│   ├── lf_temporal.py      # LF3: Chênh lệch khung hình liên tiếp
│   ├── lf_history.py       # LF4: Mật độ lịch sử theo khung giờ
│   └── lf_vlm.py           # LF5: VLM Prompting có abstention protocol
├── context.py              # Bộ trích xuất 54 ngữ cảnh
├── label_model.py          # Context-Aware Markov Label Model (EM algorithm)
├── end_model.py            # DINOv3 + Causal GRU End Model
├── evaluate.py             # So sánh đối sánh với Majority Vote, Dawid-Skene
└── README.md
```

---

## 4. Hướng dẫn Thực nghiệm

### Bước 1: Trích xuất ma trận nhãn yếu $\mathbf{L} \in \mathbb{R}^{T \times J}$ và Ngữ cảnh $\mathbf{C} \in \mathbb{R}^T$
Chạy các LF trên chuỗi video camera giao thông.

### Bước 2: Huấn luyện Label Model
```python
from direction5_weak_supervision.label_model import ContextAwareMarkovLabelModel

model = ContextAwareMarkovLabelModel(num_classes=4, num_lfs=5, num_contexts=54)
history = model.fit_em(sequences_lf, sequences_ctx, max_iters=50, verbose=True)
soft_labels = model.predict_soft_labels(test_lf, test_ctx)
```

### Bước 3: Đánh giá so sánh thuật toán gộp nhãn
```bash
python direction5_weak_supervision/evaluate.py
```

### Bước 4: Huấn luyện End Model
Sử dụng `WeakSupervisionEndModel` cùng `SoftCrossEntropyLoss` trên tập nhãn mềm đã gộp.
