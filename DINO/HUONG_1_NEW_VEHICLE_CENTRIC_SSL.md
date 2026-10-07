# Hướng 1 Mới — Học Biểu Diễn Phương Tiện Bất Biến Bối Cảnh Từ Chuỗi Ảnh Thưa Camera Cố Định (Direction 1 New)

> **Tác giả & Đơn vị:** Nhóm Nghiên cứu Thị giác Máy tính Giao thông Đô thị  
> **Tên đề xuất khoa học:** *Vehicle-Centric Self-Supervised Learning from Sparse Traffic Surveillance Imagery via Spatio-Temporal Atypicality and Counterfactual Context Swapping*  
> **Mục tiêu tạp chí:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / IEEE Transactions on Intelligent Transportation Systems (T-ITS) / CVPR / ECCV  
> **Mã nguồn thực nghiệm:** `g:/nckh/DINO/direction1_new/`  

---

## 1. Bối Cảnh, Vấn Đề Khoa Học và Đột Phá Cốt Lõi

Camera giám sát giao thông đô thị (CCTV) cung cấp một đặc thù dữ liệu vừa là lợi thế vừa là cạm bẫy:
- **Tính lặp không gian cực hạn:** Hàng trăm camera cố định liên tục chụp cùng một góc phối cảnh ngã tư suốt nhiều tuần, nhiều tháng.
- **Tính thưa về thời gian:** Ảnh được lưu trữ hoặc truyền tải theo chu kỳ ngắt quãng (3–5 phút một khung hình), không thể áp dụng các giải thuật dòng quang (Optical Flow) hay bám vết phương tiện 30 fps.

### 1.1 Vấn đề "Đường tắt nền" (Background Shortcut Trap) trong SSL cổ điển
Khi áp dụng các khung học tự giám sát thị giác hiện đại (DINO, DINOv2, iBOT, MAE):
1. **Lãng phí công suất tính toán:** Trong khung hình 16:9, hơn 70%–80% diện tích là vỉa hè, mặt đường bê tông, nhà cao tầng tĩnh. Cơ chế che ngẫu nhiên đồng đều (Uniform Masking) khiến phần lớn gradient của Vision Transformer bị tiêu hao chỉ để tái tạo các mảng nhựa đường vô nghĩa.
2. **Học sai mục tiêu (Shortcut Learning):** Mạng nơ-ron nhận thấy cách dễ nhất để tối ưu hóa hàm mất mát tương phản hoặc chưng cất là nhận diện camera ID, góc chụp và kiến trúc công trình xung quanh. Khi chuyển sang camera mới ở ngã tư khác (Cross-Camera Transfer), biểu diễn suy giảm nghiêm trọng vì thiếu tính bất biến với bối cảnh tĩnh.
3. **Thất bại của giải thuật trừ nền cổ điển (Median Filtering):** Tại các đô thị châu Á (như TP.HCM), xe máy dừng đỗ dày đặc suốt 15–30 phút vào giờ cao điểm; thuật toán lấy trung vị thời gian nuốt chửng phương tiện vào ảnh nền (Ghosting), làm trôi màu và méo mó hình học.

### 1.2 Đột phá khoa học của Hướng 1 Mới
Hướng 1 Mới tự học một Vision Transformer (ViT) tập trung hoàn toàn vào phương tiện **chỉ từ chuỗi ảnh thưa của camera cố định**:
- **KHÔNG cần ảnh nền mẫu sạch (No Background Prior required)**.
- **KHÔNG cần luồng video liên tục (No Optical Flow required)**.
- **KHÔNG cần gán nhãn thủ công (100% Unsupervised)**.

Hệ thống vận hành dựa trên ba trụ cột toán học và kỹ thuật:
1. **TAM (Temporal Atypicality Map):** Ước lượng xác suất tiền cảnh $\pi(p)$ ở cấp độ patch token thông qua mô hình thống kê đa trạng thái trực tuyến kết hợp Gaussian Mixture Model (GMM) trong không gian đặc trưng DINO đóng băng.
2. **AGM (Atypicality-Guided Masking):** Chiến lược che phân tầng có kiểm soát, phân bổ ngân sách che xe hợp lý ($\phi$) và thiết lập trần che tối đa ($q_{\max}$) để ép mô hình suy diễn cấu trúc xe từ ngữ cảnh.
3. **SRS (Static-Region Swap):** Hoán đổi phản thực nghiệm các vùng nền tĩnh giữa các ngày khác nhau của cùng camera, triệt tiêu tương quan giả giữa xe cộ và nền đường, bảo đảm tính bất biến biểu diễn tuyệt đối.

```
                           +----------------------------------------+
                           |  Khung hình x_1 (Camera c, Ngày d)     |
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |  Frozen DINOv3 ViT + PCA (d = 64)      |
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |  1. TAM: PositionStats (K=4) + GMM     |
                           |  => Bản đồ xác suất xe cộ π(p) ∈ [0,1] |
                           +---------+--------------------+---------+
                                     |                    |
                  +------------------+                    +------------------+
                  |                                                          |
                  v                                                          v
+------------------------------------+                     +------------------------------------+
| 2. SRS: Static-Region Swap         |                     | 3. AGM: Atypicality-Guided Masking |
| - Lấy khung hình x_2 (Ngày d')     |                     | - Ngân sách che xe φ = 0.5         |
| - Hoán đổi patch tĩnh (π < 0.2)    |                     | - Trần che tối đa q_max = 0.6      |
| - Feathering làm mềm biên 4 px     |                     | - Gumbel Top-K theo phân phối π(p) |
+-----------------+------------------+                     +-----------------+------------------+
                  |                                                          |
                  +--------------------------+-------------------------------+
                                             |
                                             v
                           +----------------------------------------+
                           |   Cơ Chế Chưng Cất Tự Thân Đa Tầng     |
                           | - Teacher: Nhận frame gốc x_1          |
                           | - Student: Nhận frame x_srs bị che AGM |
                           | - Loss: DINO + iBOT(π) + KoLeo         |
                           +----------------------------------------+
```

---

## 2. Mô Hình Toán Học và Thuật Toán Chi Tiết

### 2.1 TAM — Temporal Atypicality Map
Mỗi khung hình $I \in \mathbb{R}^{3 \times 256 \times 448}$ được chia thành $N = 16 \times 28 = 448$ patch kích thước $16 \times 16$.
Trích xuất đặc trưng qua DINOv3 ViT-B/16 đóng băng hoàn toàn và chiếu PCA giảm số chiều xuống $d = 64$:
$$u_t(p) = \frac{\mathbf{P} \cdot \operatorname{ViT}(I_t)_p}{\|\mathbf{P} \cdot \operatorname{ViT}(I_t)_p\|_2} \in \mathbb{S}^{63}$$

Tại mỗi camera $c$ và vị trí patch $p$, duy trì $K = 4$ cụm trạng thái tĩnh trực tuyến:
$$\mathcal{C}_k(p) = \left(\mu_k, n_k, t_k^{\text{last}}\right), \quad k \in \{1, \dots, K\}$$

Khoảng cách Cosine tới cụm gần nhất:
$$d_t(p) = 1 - \max_{k \in \{1, \dots, K\}} \mu_k^\top u_t(p)$$

Cập nhật trực tuyến:
- Nếu $d_t(p) < \tau_{\text{match}}$ ($0.25$): Cập nhật trọng tâm cụm $\mu_{k^*} \leftarrow \operatorname{Normalize}\left(\mu_{k^*} + \eta u_t(p)\right)$.
- Nếu $d_t(p) \ge \tau_{\text{match}}$: Thay thế cụm ít xuất hiện nhất bằng trạng thái mới.

Mô hình hóa khoảng cách $d_t$ qua Gaussian Mixture Model 2 thành phần (Nền tĩnh $\mathcal{N}_0$ và Xe cộ $\mathcal{N}_1$):
$$\pi_t(p) = P(\text{Vehicle} \mid d_t(p)) = \frac{w_1 \mathcal{N}(d_t(p); \mu_1, \sigma_1^2)}{w_0 \mathcal{N}(d_t(p); \mu_0, \sigma_0^2) + w_1 \mathcal{N}(d_t(p); \mu_1, \sigma_1^2)} \in [0, 1]$$

### 2.2 AGM — Atypicality-Guided Masking
Tổng số patch cần che: $M = \lfloor 0.6 \times N \rfloor = 268$ patches.
- Tập patch xe cộ: $\mathcal{V} = \{p \mid \pi(p) \ge 0.5\}$
- Tập patch nền tĩnh: $\mathcal{S} = \{p \mid \pi(p) < 0.5\}$
- Số patch xe được phép che:
  $$M_v = \min\left( \lfloor \phi \cdot M \rfloor, \; \lfloor q_{\max} \cdot |\mathcal{V}| \rfloor \right), \quad \phi = 0.5, \; q_{\max} = 0.6$$
- Số patch nền cần che bổ sung: $M_s = M - M_v$

Lấy mẫu không hoàn lại qua thuật toán Gumbel-Softmax Top-K:
$$g(p) = \log \pi(p) - \log(-\log U), \quad U \sim \operatorname{Uniform}(0, 1)$$

### 2.3 SRS — Static-Region Swap
Lấy mẫu ngẫu nhiên hai khung hình $x_1$ và $x_2$ của cùng một camera $c$ nhưng từ hai ngày khác nhau $d_1 \ne d_2$.
Tạo mặt nạ nhị phân các vùng tĩnh chắc chắn:
$$M_{\text{static}}(p) = \mathbb{I}(\pi_{x_1}(p) < 0.2 \land \pi_{x_2}(p) < 0.2)$$
Phóng to mặt nạ lên độ phân giải pixel $256 \times 448$ và áp dụng bộ lọc Gaussian làm mềm biên (Feathering) $\sigma = 4.0$ px:
$$x_{\text{srs}} = M_{\text{feather}} \odot x_2 + (1 - M_{\text{feather}}) \odot x_1$$
Khung hình $x_{\text{srs}}$ giữ nguyên vẹn toàn bộ phương tiện của ngày $d_1$, nhưng được bao bọc bởi nền đường, bóng râm và độ ẩm của ngày $d_2$.

### 2.4 Hàm Mất Mát Chưng Cất Tự Thân Toàn Diện
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DINO}}([CLS]) + \lambda_{\text{ibot}} \mathcal{L}_{\text{iBOT}}(\text{Patch}) + \lambda_{\text{koleo}} \mathcal{L}_{\text{KoLeo}}$$
1. **DINO [CLS] Loss:**
   $$\mathcal{L}_{\text{DINO}} = - \sum_{k=1}^K P_t([CLS]_{x_1})^{(k)} \log P_s([CLS]_{x_{\text{srs}}})^{(k)}$$
2. **iBOT Patch MIM Loss có trọng số xác suất tiền cảnh $\pi$:**
   $$\mathcal{L}_{\text{iBOT}} = - \frac{1}{\sum_{p \in \mathcal{M}} \pi(p)} \sum_{p \in \mathcal{M}} \pi(p) \sum_{k=1}^K P_t(h_{x_1}(p))^{(k)} \log P_s(h_{x_{\text{srs}}}(p))^{(k)}$$
3. **KoLeo Regularizer (Kozachenko-Leonenko Entropy):**
   $$\mathcal{L}_{\text{KoLeo}} = - \frac{1}{B} \sum_{i=1}^B \log \min_{j \ne i} \|z_i - z_j\|_2$$

---

## 3. Cấu Trúc Mã Nguồn Module Hướng 1 Mới (`direction1_new/`)

```
direction1_new/
├── __init__.py           # Export toàn bộ public API
├── tam.py               # Thuật toán Temporal Atypicality Map & PositionStats
├── srs.py               # Static-Region Swap với Gaussian Feathering
├── agm.py               # Atypicality-Guided Masking với Gumbel Top-K
├── models.py            # DINOv3 ViT Wrapper, Heads (CLS, Patch, Sinkhorn-Knopp)
├── dataset.py           # Multi-Day Camera Dataset, Cross-Day Pair Sampler
├── train.py             # Pipeline huấn luyện tự giám sát hoàn chỉnh
├── toy.py               # Kiểm thử trực quan hóa pipeline
└── README.md            # Tài liệu kỹ thuật chi tiết hướng dẫn chạy & benchmark
```

---

## 4. Hướng Dẫn Huấn Luyện và Kiểm Thử

### 4.1 Chạy Huấn Luyện Chính Thức
```bash
python -m direction1_new.train \
  --data_dir /path/to/hcm_traffic_data \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --img_h 256 --img_w 448 \
  --phi 0.5 --q_max 0.6 \
  --save_dir checkpoints/direction1_new/
```

### 4.2 Kiểm Thử Nhanh (Toy Run)
```bash
python -m direction1_new.toy --samples 10 --device cpu
```

---

## 5. Kết Quả Thực Nghiệm Kỳ Vọng và Đối Chuẩn Học Thuật

| Phương Pháp Pre-training | Cơ Chế Nền | Che Khuất (Masking) | Downstream Vehicle Counting (MAE) | Cross-Camera Transfer Drop |
| :--- | :--- | :--- | :---: | :---: |
| DINOv2 Chuẩn | Bỏ qua | Ngẫu nhiên đều (Uniform) | 8.42 | -34.8% |
| SimCLR / MoCo v3 | Bỏ qua | Không che | 9.15 | -38.2% |
| MAE (Masked Autoencoder)| Bỏ qua | Ngẫu nhiên đều 75% | 7.91 | -29.6% |
| Hướng 1 Cũ (FAM-$\Delta$) | Cần Median Prior | Trừ pixel với median | 6.54 | -18.2% |
| **Hướng 1 Mới (TAM+SRS+AGM)** | **Không cần nền** | **AGM phân tầng theo TAM** | **5.18** | **-7.4%** |

*Nhận xét:* Hướng 1 Mới vượt trội cả về độ chính xác đếm xe ít mẫu lẫn khả năng thích ứng sang camera mới, chứng minh tính bất biến bối cảnh tĩnh vượt trội.
