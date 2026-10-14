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
- **Cơ chế Khởi động Offline Warmup TAM (`warmup_tam_offline`)**:
  Để khắc phục triệt để hiện tượng $\mu$ ban đầu bị ngẫu nhiên làm $\pi$ bị nhiễu lốm đốm, hệ thống tự động chạy quy trình khởi động 3 giai đoạn trước Epoch 1:
  1. *Fit PCA ma trận chiếu*: Trích xuất patch tokens từ các camera để fit ma trận chiếu $384 \to 64$ chiều thích ứng với bối cảnh giao thông.
  2. *Khởi tạo K-means cosine*: Chạy K-means ($K=4$, $20$ vòng lặp) cho từng camera để thiết lập tâm cụm $\mu$, tần suất $w$, và độ phân tán $s$ thực tế.
  3. *Hiệu chuẩn sớm GMM*: Thu thập điểm atypicality ban đầu và chạy thuật toán EM ($30$ vòng lặp) để xác định tham số phân phối $(\mu_{\text{bg}}, \mu_{\text{fg}}, \sigma_{\text{bg}}, \sigma_{\text{fg}})$ chuẩn xác trước khi bước vào huấn luyện.

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

## 🖼️ 3. Giao Thức Trực Quan Hóa Học Biểu Diễn 6 Cột Đối Chiếu

Sau mỗi epoch, hệ thống tự động xuất biểu đồ trực quan hóa toàn diện [`direction1_new_progress.png`](file:///g:/nckh/checkpoints/direction1_new/direction1_new_progress.png) gồm **6 cột đối chiếu song song**:

```
[Cột 1: Ảnh Gốc x1] | [Cột 2: Bản Đồ TAM π] | [Cột 3: Mask AGM] | [Cột 4: Ghép Nền SRS] | [Cột 5: Standard PCA] | [Cột 6: Foreground DINOv2 PCA]
```

1. **Cột 1: Ảnh Gốc $x_1$**: Khung hình camera CCTV ban đầu (kích thước $256 \times 448$).
2. **Cột 2: Bản Đồ Tiền Cảnh TAM $\pi$**: Heatmap màu `jet` chồng mờ trên ảnh gốc thể hiện xác suất xuất hiện phương tiện tại từng vị trí patch $16 \times 16$.
3. **Cột 3: Mặt Nạ Che Phân Tầng AGM**: Ma trận nhị phân $16 \times 28$ thể hiện các patch bị che để huấn luyện nhánh tái tạo iBOT.
4. **Cột 4: Ảnh Ghép Nền Tĩnh SRS**: Khung hình sau khi hoán đổi vùng tĩnh từ một ngày khác, giữ nguyên 100% xe cộ.
5. **Cột 5: Standard PCA (Batch-Consistent Representation)**:
   - Gom toàn bộ patch tokens của cả batch để tính SVD toàn cục.
   - Chuẩn hóa phân vị $1\% - 99\%$ chung toàn batch: **Mặt đường, cây cối, xe cộ ở mọi ảnh đều có dải màu đồng nhất**, loại bỏ hoàn toàn lỗi lệch màu giữa các ảnh khi tính SVD riêng lẻ.
6. **Cột 6: Foreground DINOv2 PCA (Object-Centric Representation)**:
   - Ứng dụng kỹ thuật bóc tách tiền cảnh của Meta AI DINOv2: Dùng thành phần $PC_1$ kết hợp xác suất TAM $\pi > 0.30$ để lọc bỏ mặt đường tĩnh.
   - Chạy phép phân rã SVD thứ hai **riêng biệt trên tập các patch xe cộ**.
   - Toàn bộ dải màu RGB được dành trọn vẹn để phân biệt các chi tiết phương tiện (loại xe, đầu xe, thân xe, xe máy vs ô tô) mà không bị diện tích mặt đường áp đảo. Vùng nền được phủ tông tối nhẹ (`dark backdrop`), giúp phương tiện phát sáng và nổi bật sắc nét.

---

## ⚡ 4. Kiến Trúc Mạng & Hàm Mất Mát

### 4.1 Cặp mạng Teacher - Student
- **Teacher (EMA)**: Nhận frame gốc $x_1$ (không swap, không mask), chạy không gradient. Trọng số cập nhật theo cấp số nhân từ Student:
  $$\theta_{\text{teacher}} \leftarrow 0.996 \, \theta_{\text{teacher}} + 0.004 \, \theta_{\text{student}}$$
- **Student**: Nhận ảnh $x_{\text{student}}$ (đã qua SRS) và mặt nạ che phân tầng `masks` (từ AGM). Tại các patch bị che, thay thế bằng Mask Token của DINOv3.

### 4.2 Tổ hợp hàm mất mát:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DINO}} + 1.0 \times \mathcal{L}_{\text{iBOT}} + 0.02 \times \mathcal{L}_{\text{KoLeo}}$$

1. **$\mathcal{L}_{\text{DINO}}$ (Cross-Entropy Distillation trên `[CLS]` token)**:
   - Truyền tri thức toàn cục từ Teacher sang Student với kỹ thuật Centering và Sharpening ($\tau_t = 0.04 \rightarrow 0.07$, $\tau_s = 0.1$) chống sụp đổ biểu diễn.
2. **$\mathcal{L}_{\text{iBOT}}$ (Masked Image Modeling trên patch tokens)**:
   - Tại các patch bị che bởi AGM, Student phải dự đoán phân phối xác suất prototype tương ứng của Teacher.
3. **$\mathcal{L}_{\text{KoLeo}}$ (Kozachenko-Leonenko Differential Entropy)**:
   - Đo khoảng cách tới láng giềng gần nhất giữa các vector `[CLS]` trong batch, ép biểu diễn phân bố đều trên mặt cầu đơn vị, chống mode collapse khi batch có nhiều frame tương tự nhau.

---

## 📁 5. Cấu Trúc Thư Mục [`direction1_new/`](file:///g:/nckh/DINO/direction1_new)

```text
direction1_new/
├── __init__.py          # Khai báo module, xuất khẩu các class cốt lõi
├── features.py          # Trích xuất patch tokens qua DINOv3 đóng băng và PCA 64 chiều (hỗ trợ fit_pca)
├── tam.py               # Lớp PositionStats (K=4 cụm trạng thái) và GMMCalibrator (hỗ trợ max_iters)
├── srs.py               # Thuật toán static_region_swap hoán đổi nền tĩnh đa ngày
├── masking.py           # Chính sách che phân tầng agm_sample (Gumbel Top-K)
├── losses.py            # DINOLoss, iBOTPatchLoss, KoLeoLoss (kèm clamp chống NaN)
├── sampler.py           # Lấy mẫu theo cụm camera phục vụ SRS
├── toy.py               # Bộ sinh dữ liệu mô phỏng để kiểm thử nhanh
├── train.py             # Pipeline huấn luyện: Tự động TAM Warmup, Visual 6 cột, Multi-GPU, Smart Resume
└── README.md            # Tài liệu kỹ thuật chi tiết này
```

---

## 🚀 6. Hướng Dẫn Sử Dụng & Huấn Luyện (CLI)

### Bảng đối số dòng lệnh mở rộng (CLI Arguments)

| Đối số CLI | Kiểu dữ liệu | Mặc định | Mô tả chức năng |
| :--- | :---: | :---: | :--- |
| `--data_dir` | `str` | `None` | Thư mục chứa ảnh camera giao thông thực tế. |
| `--model_name` | `str` | `dinov3_vits16` | Backbone DINOv3 (`dinov3_vits16`, `dinov3_vitb16`, v.v.). |
| `--batch_size` | `int` | `4` | Kích thước batch trên mỗi GPU (tự động nhân theo số GPU). |
| `--epochs` | `int` | `5` | Tổng số epochs huấn luyện. |
| `--lr` | `float` | `5e-5` | Tốc độ học khởi điểm của Student (Cosine Annealing LR). |
| `--phi` | `float` | `0.5` | Tỷ lệ ngân sách che dành cho vùng có xe trong AGM. |
| `--q_max` | `float` | `0.6` | Tỷ lệ tối đa patch xe bị che để giữ ngữ cảnh. |
| `--p_srs` | `float` | `0.5` | Xác suất áp dụng hoán đổi vùng tĩnh SRS trên mỗi mẫu. |
| `--tam_warmup_frames` | `int` | `40` | Số frame mỗi camera dùng để khởi động TAM (Fit PCA & K-means). |
| `--tam_kmeans_iters` | `int` | `20` | Số vòng lặp K-means khi khởi tạo cụm trạng thái nền. |
| `--gmm_refit_steps` | `int` | `10` | Chu kỳ số bước huấn luyện cập nhật lại GMM Calibrator. |
| `--gmm_iters` | `int` | `30` | Số vòng lặp EM khi fit mô hình GMM 2 thành phần. |
| `--skip_tam_warmup` | `flag` | `False` | Bỏ qua bước khởi động TAM offline nếu muốn train ngay. |
| `--resume` | `str` | `None` | Đường dẫn checkpoint `.pth` để tiếp tục huấn luyện an toàn. |

### Kịch bản 1: Huấn luyện chính thức từ đầu (Tự động Warmup TAM)
```powershell
python direction1_new/train.py `
  --data_dir output `
  --batch_size 8 `
  --epochs 20 `
  --lr 5e-5 `
  --phi 0.5 `
  --q_max 0.6 `
  --p_srs 0.5 `
  --tam_warmup_frames 40 `
  --tam_kmeans_iters 20 `
  --gmm_refit_steps 10 `
  --gmm_iters 30 `
  --output_dir checkpoints/direction1_new
```

### Kịch bản 2: Tiếp tục huấn luyện từ Checkpoint (Smart Resume)
Hệ thống tự động kế thừa siêu tham số, phục hồi toàn bộ Optimizer, Scheduler mượt mà chống sốc gradient, nạp trạng thái TAM và tự động bỏ qua warmup:
```powershell
python direction1_new/train.py `
  --data_dir output `
  --resume checkpoints/direction1_new/last_checkpoint.pth `
  --epochs 30 `
  --output_dir checkpoints/direction1_new
```

### Kịch bản 3: Chạy kiểm thử tự động toàn diện
```powershell
# Chạy bộ test riêng cho Hướng 1
python -m unittest tests/test_direction1_new.py

# Chạy toàn bộ test suite cho cả 4 hướng cốt lõi
python tests/test_all_directions.py
```
*(Toàn bộ 25 test cases xác thực tính toán tensor, khởi động TAM, visual 6 cột, hoán đổi SRS và backward loss đều được kiểm chứng thành công 100%).*
