# 🚗 Hướng 1 Mới: Camera-Invariant Self-Supervised Learning from Sparse Surveillance Imagery (Direction 1 New / formerly Direction G)

> **Tên đề xuất khoa học**: *Vehicle-Centric Representation Learning from Sparse Static-Camera Feeds via Long-Term Temporal Atypicality and Counterfactual Background Swapping*  
> **Đóng góp học thuật**: Giải quyết triệt để vấn đề "Background Shortcut" (Đường tắt nền) trong học tự giám sát (SSL) trên ảnh camera giao thông chụp thưa (chu kỳ polling 5 phút). Hoàn toàn **không cần ảnh nền mẫu sạch (No Clean Background Prior)** và **không cần luồng video dày 30 fps (No Optical Flow)**.

---

## 📌 1. Bối Cảnh Nghiên Cứu & Đột Phá Khoa Học (Novelty ⭐⭐⭐⭐⭐)

### 1.1 Vấn đề của các phương pháp tự giám sát truyền thống
- **Dữ liệu thưa**: Hệ thống camera giám sát đô thị (như bộ dữ liệu 608 camera TP.HCM `IC4SD-TrafficSnap`) gửi ảnh theo chu kỳ tĩnh (~5 phút/snapshot). Không có luồng video liên tục để tính toán Optical Flow hoặc theo dõi đối tượng (tracking).
- **Background Shortcut Trap**: Trong các khung hình camera cố định, 70–80% diện tích là vỉa hè, mặt đường, cây cối và nhà cửa bất biến. Khi huấn luyện tự giám sát (DINO, DINOv2, MAE, SimCLR), mạng nơ-ron có xu hướng **học cách nhận diện camera ID và góc quay ngã tư** thông qua cấu trúc nền tĩnh thay vì học đặc trưng của phương tiện giao thông. Kết quả là biểu diễn suy giảm nghiêm trọng khi chuyển sang camera chưa từng thấy (Cross-Camera Generalization).
- **Thất bại của phép trừ nền cổ điển**: Việc lấy trung vị theo thời gian (temporal median background) thường xuyên sinh ra bóng ma (ghosting) hoặc trôi màu do mật độ xe máy TP.HCM quá đông đúc, thời tiết nắng mưa thất thường và chế độ hồng ngoại ban đêm.

### 1.2 Đột phá của Hướng 1 Mới
Hướng 1 Mới kết hợp **3 kỹ thuật cốt lõi** hoạt động ở cấp độ patch token của Vision Transformer:
1. **TAM (Temporal Atypicality Map)**: Đo độ bất thường không-thời gian dài hạn bằng cách duy trì đa trạng thái tĩnh trực tuyến ở tầng đặc trưng patch, tự động tách xác suất tiền cảnh $\pi(p)$ mà không cần ảnh nền pixel.
2. **SRS (Static-Region Swap)**: Hoán đổi phản thực nghiệm (counterfactual swapping) các khối nền tĩnh giữa các ngày khác nhau của cùng camera, triệt tiêu sự phụ thuộc vào nền tĩnh và ép mô hình dồn chú ý vào cấu trúc phương tiện.
3. **AGM (Atypicality-Guided Masking)**: Chiến lược che phân tầng có kiểm soát, phân bổ ngân sách che xe hợp lý ($\phi$) và khống chế trần che tối đa ($q_{\max}$) để luôn giữ lại đủ ngữ cảnh cho mạng suy luận.

```
                            ┌──────────────────────────────────────────┐
                            │    Input Frame x1 (Camera c, Day d)      │
                            └────────────────────┬─────────────────────┘
                                                 ▼
                            ┌──────────────────────────────────────────┐
                            │      Frozen DINOv3 ViT + PCA (d=64)      │
                            └────────────────────┬─────────────────────┘
                                                 ▼
                            ┌──────────────────────────────────────────┐
                            │  1. TAM: PositionStats (K=4) + GMM       │
                            │  -> Xác suất tiền cảnh xe cộ π(p) ∈ [0,1]│
                            └────────┬─────────────────────────┬───────┘
                                     │                         │
            ┌────────────────────────┴────────┐       ┌────────┴────────────────────────┐
            ▼                                 ▼       ▼                                 ▼
┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
│    2. SRS (Static-Region Swap)       │            │  3. AGM (Atypicality-Guided Masking) │
│ - Lấy frame x2 (cùng cam, ngày d')   │            │ - Phân bổ ngân sách che xe φ = 0.5   │
│ - Ghép 60% patch tĩnh (π < 0.2)      │            │ - Trần che tối đa q_max = 0.6        │
│ - Làm mềm biên feathering 4 px       │            │ - Gumbel Top-K lấy mẫu theo π(p)     │
└──────────────────┬───────────────────┘            └──────────────────┬───────────────────┘
                   │                                                   │
                   └─────────────────────────┬─────────────────────────┘
                                             ▼
                   ┌───────────────────────────────────────────────────┐
                   │        Teacher - Student Distillation + MIM       │
                   │  Teacher: Frame gốc x1 (không swap, không mask)   │
                   │  Student: Frame x_srs + AGM Masks                 │
                   │  Loss = DINO (CLS) + 1.0*iBOT (MIM) + 0.02*KoLeo  │
                   └───────────────────────────────────────────────────┘
```

---

## 🔬 2. Ba Trụ Cột Thuật Toán Chi Tiết

### 2.1 TAM — Temporal Atypicality Map
- **Biểu diễn đầu vào**: Khung hình $448 \times 256$ đưa qua backbone DINOv3 ViT-B/16 (hoặc ViT-S/16) **đóng băng hoàn toàn** $\rightarrow 16 \times 28 = 448$ patch tokens $\rightarrow$ chuẩn hóa L2 $\rightarrow$ chiếu PCA giảm chiều xuống $d = 64$ vector đơn vị $u_t(p)$.
- **Bộ nhớ đa trạng thái theo vị trí**: Tại mỗi camera $c$ và vị trí patch $p$, duy trì $K = 4$ trạng thái:
  - Tâm cụm $\mu_k \in \mathbb{R}^{64}$ (chuẩn hóa $\|\mu_k\|_2 = 1$).
  - Tần suất xuất hiện $w_k \in [0, 1]$ ($\sum_{k=1}^K w_k = 1$).
  - Độ phân tán góc $s_k = \mathbb{E}[1 - \cos(u, \mu_k)]$.
- **Tại sao phân biệt được Mặt đường và Xe cộ?**
  - Mặt đường/vỉa hè: 70% thời gian là nhựa khô, 15% là nhựa ướt. Hai trạng thái này tạo thành các cụm **rất chặt và xuất hiện liên tục** ($w$ lớn, $s$ nhỏ).
  - Phương tiện (xe máy, ô tô, xe buýt đủ màu sắc): Xuất hiện ngẫu nhiên, di chuyển liên tục, vector feature phân tán rộng khắp không gian 64 chiều $\rightarrow$ **không bao giờ gom thành cụm tĩnh có $w$ lớn**.
- **Cập nhật trực tuyến (Online EMA, không gradient)**:
  $$k^* = \arg\max_k \cos(u_t(p), \mu_k)$$
  $$\mu_{k^*} \leftarrow \frac{(1-\eta)\mu_{k^*} + \eta u_t(p)}{\|\cdot\|_2}, \quad w_k \leftarrow (1-\eta_w)w_k + \eta_w \mathbb{I}[k = k^*], \quad s_{k^*} \leftarrow (1-\eta)s_{k^*} + \eta(1 - \cos(u_t(p), \mu_{k^*}))$$
  *(Với $\eta = 0.02$, $\eta_w = 0.005$. Tự động khởi tạo lại cụm chết nếu phát hiện trạng thái môi trường mới).*
- **Điểm khác thường Mahalanobis-like**:
  $$a_t(p) = \min_{k \in \mathcal{S}} \frac{1 - \cos(u_t(p), \mu_k)}{s_k + \epsilon}$$
  với $\mathcal{S} = \{k : w_k \ge 0.15, s_k \le 1.5 \times \text{median}(s)\}$.
- **Hiệu chuẩn GMM 2 thành phần**:
  Sử dụng mô hình GMM 2 thành phần trên $\log a_t(p)$ để chuyển đổi sang xác suất tiền cảnh phương tiện $\pi_t(p) \in [0, 1]$ hoàn toàn không giám sát.

### 2.2 SRS — Static-Region Swapping
- **Động cơ**: Nếu nền giữ nguyên, mô hình học vẹt nền. Nếu thay nền bằng ảnh camera khác, phối cảnh và hình học mặt đường sẽ bị sai lệch. Do đó, ta **hoán đổi nền giữa các ngày khác nhau của CHÍNH camera đó**.
- **Quy trình hoán đổi**:
  1. Lấy frame $x_1$ ngày $d$ và frame $x_2$ ngày $d'$ của cùng camera (khác thời điểm $\pm 2$ giờ để khác góc nắng/bóng râm).
  2. Xác định tập patch tĩnh chung: $\text{Static} = \{p : \pi_1(p) < 0.2 \land \pi_2(p) < 0.2\}$.
  3. **Giãn nở an toàn (Dilation)**: Loại trừ các patch kề cận vùng xe của cả 2 ảnh (bán kính 1 patch) để không bao giờ cắt dính nửa chiếc xe.
  4. Lấy ngẫu nhiên $60\%$ số patch tĩnh hợp lệ từ $x_2$ đè sang $x_1$.
  5. Áp dụng dải làm mềm feathering $4\text{ px}$ ở biên khối ghép.
- **Kết quả**: Vùng xe của $x_1$ được bảo toàn 100% pixel, trong khi nền đường và ánh sáng xung quanh bị thay thế ngẫu nhiên!

### 2.3 AGM — Atypicality-Guided Masking
- **Khắc phục nhược điểm của Masking truyền thống**:
  - *Random Masking (MAE)*: Đa phần che trúng lòng đường trống, làm mất thời gian học tái tạo nhựa đường.
  - *Greedy Masking*: Che toàn bộ vùng xe làm Student mất sạch ngữ cảnh (không biết vật thể là xe gì để phục hồi).
- **Hai tham số diễn giải được**:
  - $\phi = 0.5$ (**Foreground Budget**): Dành $50\%$ số patch bị che cho vùng phương tiện.
  - $q_{\max} = 0.6$ (**Max Mask Cap**): Giới hạn trần che tối đa $60\%$ diện tích xe, **luôn giữ lại ít nhất $40\%$ patch xe làm ngữ cảnh**.
- **Lấy mẫu có trọng số**: Áp dụng kỹ thuật *Gumbel Top-K* trên phân phối xác suất $\pi(p)$ để lấy mẫu không hoàn lại.

---

## ⚡ 3. Kiến Trúc Mạng & Hàm Mất Mát

### 3.1 Cặp mạng Teacher - Student
- **Teacher (EMA)**: Nhận frame gốc $x_1$ (không swap, không mask), chạy không gradient. Trọng số cập nhật theo cấp số nhân từ Student:
  $$\theta_{\text{teacher}} \leftarrow 0.996 \, \theta_{\text{teacher}} + 0.004 \, \theta_{\text{student}}$$
- **Student**: Nhận ảnh $x_{\text{student}}$ (đã qua SRS) và mặt nạ che phân tầng `masks` (từ AGM). Tại các patch bị che, thay thế bằng Mask Token của DINOv3.

### 3.2 Tổ hợp hàm mất mát:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DINO}} + 1.0 \times \mathcal{L}_{\text{iBOT}} + 0.02 \times \mathcal{L}_{\text{KoLeo}}$$

1. **$\mathcal{L}_{\text{DINO}}$ (Cross-Entropy Distillation trên `[CLS]` token)**:
   - Truyền tri thức toàn cục từ Teacher sang Student với kỹ thuật Centering và Sharpening ($\tau_t = 0.04 \rightarrow 0.07$, $\tau_s = 0.1$) chống sụp đổ biểu diễn.
2. **$\mathcal{L}_{\text{iBOT}}$ (Masked Image Modeling trên patch tokens)**:
   - Tại các patch bị che bởi AGM, Student phải dự đoán phân phối xác suất prototype tương ứng của Teacher.
3. **$\mathcal{L}_{\text{KoLeo}}$ (Kozachenko-Leonenko Differential Entropy)**:
   - Đo khoảng cách tới láng giềng gần nhất giữa các vector `[CLS]` trong batch, ép biểu diễn phân bố đều trên mặt cầu đơn vị, chống mode collapse khi batch có nhiều frame tương tự nhau.

---

## 📁 4. Cấu Trúc Thư Mục [`direction1_new/`](file:///g:/nckh/DINO/direction1_new)

```text
direction1_new/
├── __init__.py          # Khai báo module, xuất khẩu các class cốt lõi
├── features.py          # Trích xuất patch tokens qua DINOv3 đóng băng và PCA 64 chiều
├── tam.py               # Lớp PositionStats (K=4 cụm trạng thái) và GMMCalibrator
├── srs.py               # Thuật toán static_region_swap hoán đổi nền tĩnh đa ngày
├── masking.py           # Chính sách che phân tầng agm_sample (Gumbel Top-K)
├── losses.py            # DINOLoss, iBOTPatchLoss, KoLeoLoss (kèm clamp chống NaN)
├── sampler.py           # Lấy mẫu theo cụm camera phục vụ SRS
├── toy.py               # Bộ sinh dữ liệu mô phỏng để kiểm thử nhanh
├── train.py             # Pipeline huấn luyện tự động hỗ trợ Multi-GPU, AMP, Smart Resume
└── README.md            # Tài liệu kỹ thuật chi tiết này
```

---

## 🚀 5. Hướng Dẫn Sử Dụng & Huấn Luyện (CLI)

### Kịch bản 1: Huấn luyện chính thức trên dữ liệu thực tế (Multi-GPU / Single-GPU)
Hệ thống **tự động phát hiện toàn bộ GPU**, kích hoạt Mixed Precision (AMP float16) và cơ chế tự động mở rộng buffer camera:
```powershell
python direction1_new/train.py `
  --data_dir output `
  --batch_size 16 `
  --epochs 20 `
  --lr 5e-5 `
  --phi 0.5 `
  --q_max 0.6 `
  --p_srs 0.5 `
  --output_dir checkpoints/direction1_new
```

### Kịch bản 2: Tiếp tục huấn luyện từ Checkpoint (Resume)
Hệ thống tự động dọn sạch tiền tố `module.`, khôi phục trọn vẹn Model, Optimizer, Scheduler và lịch sử loss:
```powershell
python direction1_new/train.py `
  --data_dir output `
  --resume checkpoints/direction1_new/last_checkpoint.pth `
  --epochs 30 `
  --output_dir checkpoints/direction1_new
```

### Kịch bản 3: Chạy kiểm thử tự động (Unit Test)
Chạy bộ kiểm thử độc lập trong thư mục `tests/`:
```powershell
python -m unittest tests/test_direction1_new.py
```
*(Toàn bộ 5 test case xác thực tính toán tensor, cập nhật trực tuyến, GMM calibration, hoán đổi SRS và backward loss đều được kiểm chứng thành công 100%).*
