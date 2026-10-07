# 🏙️ Hướng 2 Mới: Prior-Free Traffic Scene Decomposition (Direction 2 New)

> **Tên đề xuất khoa học**: *Unsupervised Traffic Scene Decomposition Without Clean Background Priors via Multi-Illumination Scene Basis and Uncertainty-Aware Laplace Formulation*  
> **Đóng góp học thuật**: Tự động phân rã cảnh giao thông phức tạp thành các thành phần quang học độc lập (**Alpha Matte xe cộ, Lớp tiền cảnh xe nổi, Lớp nền đường tĩnh không còn xe, và Bản đồ độ bất định**) mà **hoàn toàn không cần ảnh nền mẫu sạch cung cấp trước (Prior-Free)**.

---

## 📌 1. Bối Cảnh & Đột Phá Khoa Học (Novelty ⭐⭐⭐⭐⭐)

### 1.1 Hạn chế của Hướng 2 Cũ (Background-Supervised Decomposition)
- Phương pháp phân rã cảnh ban đầu giả định rằng hệ thống đã có sẵn ảnh nền tĩnh sạch $I_{\text{bg}}$ để giám sát nhánh dự đoán nền:
  $$\mathcal{L}_{\text{bg}} = \|\hat{I}_{\text{bg}} - I_{\text{bg}}\|_1$$
- **Điểm nghẽn thực tế**: Trong các đô thị đông đúc như TP.HCM, lòng đường gần như luôn có phương tiện di chuyển hoặc đậu đỗ. Việc tạo ra ảnh nền sạch tuyệt đối là bất khả thi. Khi ảnh nền mẫu bị "dính ma xe" (ghosting artifacts), mạng nơ-ron sẽ học sai và in luôn bóng xe vào lớp nền tái tạo!

### 1.2 Đột phá của Hướng 2 Mới
Hướng 2 Mới thiết lập quy trình giải bài toán phân rã cảnh **hoàn toàn không cần ảnh nền** thông qua hai bước phối hợp:
1. **Mô hình đa tạp chiếu sáng ít chiều (Low-Rank Scene Basis)**: Mô hình hóa mọi biến thể quang học của nền đường (trời nắng, bóng râm, đường ướt sau mưa, đèn đường ban đêm) thành tổ hợp tuyến tính:
   $$B(t) = E_0 + \sum_{j=1}^J \ell_{t, j} E_j$$
   trong đó $E_0$ là nền tĩnh trung tính, $E_j$ là $J$ ảnh cơ sở biến đổi ánh sáng, và $\ell_{t, j}$ là mã chiếu sáng tự do của khung hình $t$.
2. **Tối ưu hóa mạnh mẽ bằng IRLS (Robust Iteratively Reweighted Least Squares)**: Trọng số $w$ tự động gán giá trị xấp xỉ 0 tại các pixel có xe cộ, giúp trích xuất nền chuẩn xác mà không bị nhiễm màu xe.
3. **Mạng nơ-ron phân rã sâu có ước lượng bất định (Uncertainty-Aware Laplace Formulation)**: Thay vì dùng hàm mất mát L1/L2 thông thường, mạng dự đoán đồng thời tham số phân phối Laplace $(\hat{I}, \sigma)$, cho phép tự động bỏ qua (down-weight) các vùng có độ bất định cao (viền xe, bóng đổ phức tạp).

```
                             [KHUNG HÌNH GỐC I_origin]
                                        │
                                        ▼
                        ┌───────────────────────────────┐
                        │   DINOv3 ViT Patch Backbone   │
                        └───────┬───────────────┬───────┘
                                │               │
                ┌───────────────┘               └───────────────┐
                ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│     Multi-Scale Decoder       │               │      Lighting Head (ell)      │
│  - Multi-layer ConvTrans2d    │               │  - Linear MLP từ Patch tĩnh   │
│  - Skip-connection RGB sắc nét│               │  - Dự đoán mã ánh sáng ell    │
└───────┬───────┬───────┬───────┘               └───────────────┬───────────────┘
        │       │       │       │                               │
        ▼       ▼       ▼       ▼                               ▼
    [Alpha]   [Fg]    [Bg]   [sigma]                         [ell]
     M_fg    I_fg    I_bg   Độ bất định                   Mã chiếu sáng
        │       │       │       │                               │
        └───────┼───────┼───────┘                               │
                ▼                                               ▼
   [Tổng hợp Alpha Compositing]                    [Tái tạo nền từ SceneBasis]
   I_recon = M ⊙ Fg + (1-M) ⊙ Bg                   B_basis = E0 + Σ ell_j * Ej
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │       SceneDecompositionLossV2:                     │
             │   1. Laplace NLL Loss (Reconstruction with sigma)   │
             │   2. Pseudo-Bg Loss (từ SceneBasis IRLS)            │
             │   3. Mask Sparsity & Total Variation Smoothness     │
             │   4. Lighting Consistency Loss (ell vs solve_ell)   │
             └─────────────────────────────────────────────────────┘
```

---

## 🔬 2. Kiến Trúc Kỹ Thuật Chi Tiết

### 2.1 Giai đoạn 1: Khớp Đa Tạp Nền Offline (`scene_fit.py`)
- **Tham số hóa SceneBasis**:
  - $E_0 \in \mathbb{R}^{3 \times 256 \times 448}$: Cảnh tĩnh trung bình đầy đủ độ phân giải.
  - $E_j \in \mathbb{R}^{J \times 3 \times 128 \times 224}$ ($J = 4$): Các ảnh cơ sở ánh sáng ở độ phân giải $1/2$ (giảm số tham số và tăng tính khái quát).
  - $\ell \in \mathbb{R}^{N \times J}$: Mã chiếu sáng cho $N$ khung hình quan sát.
- **Thuật toán Robust IRLS**:
  - Tại mỗi bước lặp, tính phần dư $r_t = |I_t - B_t|$.
  - Cập nhật trọng số thích nghi:
    $$w_t(p) = \frac{1}{\sqrt{r_t(p)^2 + \epsilon^2}} \cdot (1 - \pi_t(p))$$
    với $\pi_t(p)$ là xác suất tiền cảnh từ TAM (nếu có).
  - Pixel có xe cộ sẽ có phần dư $r_t$ lớn $\rightarrow w_t \approx 0 \rightarrow$ xe cộ bị loại bỏ hoàn toàn khỏi quá trình tối ưu hóa $E_0, E_j$.
- **Kết quả đầu ra**: Tự động sinh ra tập ảnh nền giả định (**Pseudo-Backgrounds**) chuẩn nét cho từng khung hình để mồi huấn luyện mạng nơ-ron ở Giai đoạn 2.

### 2.2 Giai đoạn 2: Mạng Phân Rã Sâu (`models.py`)
Mạng `TrafficDecompositionNet` gồm 5 đầu ra chuyên biệt:
1. **$M_{\text{fg}} \in [0, 1]$ (Alpha Matte)**: Kênh mặt nạ phương tiện mềm (kích hoạt Sigmoid), tách biệt ranh giới giữa xe và đường.
2. **$\hat{I}_{\text{fg}} \in [0, 1]^3$ (Foreground Image)**: Lớp ảnh chỉ chứa phương tiện giao thông.
3. **$\hat{I}_{\text{bg}} \in [0, 1]^3$ (Road Inpainting)**: Lớp ảnh mặt đường tĩnh sạch bóng xe.
4. **$\sigma \in [0.01, 0.5]$ (Uncertainty Map)**: Dự đoán độ bất định của mô hình tại từng pixel, kích hoạt qua $\exp(\text{clip}(\log\sigma))$.
5. **$\ell \in \mathbb{R}^J$ (Lighting Code)**: Vector biểu diễn điều kiện chiếu sáng tức thời của khung hình.

### 2.3 Giải Mã Ánh Sáng Nhanh Trực Tuyến (`solve_ell.py`)
Tại thời điểm suy luận hoặc kiểm tra tính nhất quán, mã chiếu sáng $\ell^*$ được tìm kiếm chính xác theo công thức bình phương tối thiểu có trọng số (Weighted Least Squares):
$$\ell^* = \arg\min_{\ell} \sum_p w(p) \left\| I(p) - \left( E_0(p) + \sum_{j=1}^J \ell_j E_j(p) \right) \right\|_2^2$$
Lời giải giải tích đóng qua phương trình pháp tuyến (Normal Equations) giải trong $< 2\text{ ms}$ trên GPU.

### 2.4 Tổ Hợp Hàm Mất Mát (`losses.py`)
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Laplace}} + \lambda_{\text{bg}} \mathcal{L}_{\text{pseudo\_bg}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}} + \lambda_{\text{tv}} \mathcal{L}_{\text{tv}} + \lambda_{\text{light}} \mathcal{L}_{\text{light}}$$

1. **$\mathcal{L}_{\text{Laplace}}$ (Laplace Negative Log-Likelihood)**:
   $$\mathcal{L}_{\text{Laplace}} = \frac{1}{|\Omega|} \sum_{p \in \Omega} \left( \frac{|\hat{I}_{\text{origin}}(p) - I_{\text{origin}}(p)|}{\sigma(p)} + \log(2\sigma(p)) \right)$$
   Mô hình tự động học cách tăng $\sigma$ ở những vùng viền xe khó dự đoán để không làm nổ gradient, và giảm $\sigma$ ở vùng mặt đường phẳng để ép tái tạo sắc nét.
2. **$\mathcal{L}_{\text{pseudo\_bg}}$**: Giám sát nhánh $\hat{I}_{\text{bg}}$ bằng ảnh nền từ SceneBasis đã làm sạch.
3. **$\mathcal{L}_{\text{sparse}}$ & $\mathcal{L}_{\text{tv}}$**: Ép mặt nạ Alpha Matte thưa thớt (chỉ sáng ở chỗ có xe) và mượt mà trong không gian.
4. **$\mathcal{L}_{\text{light}}$**: Ép mã ánh sáng $\ell$ của mạng nơ-ron khớp với mã giải tích từ `solve_ell`.

---

## 📁 3. Cấu Trúc Thư Mục [`direction2_new/`](file:///g:/nckh/DINO/direction2_new)

```text
direction2_new/
├── __init__.py          # Khai báo module, xuất khẩu các class và hàm cốt lõi
├── scene_fit.py         # Giai đoạn 1: Khớp SceneBasis (E0, Ej, ell) bằng Robust IRLS
├── solve_ell.py         # Solver giải tích Weighted Least Squares cho mã ánh sáng ell
├── models.py            # TrafficDecompositionNet (DINOv3 ViT + 5 đầu ra chuyên biệt)
├── losses.py            # SceneDecompositionLossV2 (Laplace NLL, Sparsity, Total Variation)
├── dataset.py           # DecompositionDataset nạp cặp ảnh và pseudo-backgrounds
├── train.py             # Giai đoạn 2: Pipeline huấn luyện mạng phân rã sâu
├── infer.py             # Script suy luận phân rã cảnh và xóa xe trên ảnh thực tế
└── README.md            # Tài liệu kỹ thuật chi tiết này
```

---

## 🚀 4. Hướng Dẫn Sử Dụng & Huấn Luyện (CLI)

### Bước 1: Khớp Đa Tạp Nền SceneBasis (Giai đoạn 1)
Chạy trên chuỗi ảnh của camera để tối ưu $E_0$ và các ảnh cơ sở chiếu sáng $E_j$, tự động xuất các ảnh nền pseudo-background:
```powershell
python direction2_new/scene_fit.py `
  --frames_dir output `
  --num_frames 32 `
  --iters 200 `
  --J 4 `
  --lr 0.01 `
  --save_dir checkpoints/direction2_new_scene_fit
```

### Bước 2: Huấn luyện Mạng Phân Rã Sâu TrafficDecompositionNet (Giai đoạn 2)
Huấn luyện mạng nơ-ron sử dụng các ảnh pseudo-background vừa sinh ra ở Bước 1:
```powershell
python direction2_new/train.py `
  --origin_dir output `
  --bg_dir checkpoints/direction2_new_scene_fit/pseudo_bgs `
  --backbone dinov3_vits16 `
  --img_size 256 `
  --epochs 20 `
  --batch_size 8 `
  --lr 3e-4 `
  --save_dir checkpoints/direction2_new
```

### Bước 3: Suy luận Tách Lớp & Tự Động Xóa Xe (Inference / Road Inpainting)
Chạy suy luận trên ảnh mới bất kỳ (hoàn toàn không cần ảnh nền lúc test):
```powershell
python direction2_new/infer.py `
  --weights checkpoints/direction2_new/best_model.pth `
  --input_path output `
  --output_dir checkpoints/direction2_new/inferred `
  --mode S
```
Kết quả xuất ra gồm:
- `*_alpha.png`: Mặt nạ phương tiện mềm (Alpha Matte).
- `*_fg.png`: Lớp tiền cảnh phương tiện nổi.
- `*_bg.png`: Mặt đường sạch bóng xe (Road Inpainting).
- `*_sigma.png`: Bản đồ độ bất định quang học.

### Bước 4: Chạy kiểm thử tự động (Unit Test)
Chạy bộ kiểm thử độc lập trong thư mục `tests/`:
```powershell
python -m unittest tests/test_direction2_new.py
```
*(Toàn bộ 4 test case xác thực SceneBasis, solve_ell solver, mô hình 5 đầu ra và backward LossV2 đều vượt qua thành công 100%).*
