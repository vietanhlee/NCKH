# ST-BMB: Spatio-Temporal Background Memory Bank Suite

> **Tên đề xuất khoa học:** *Streaming Traffic Scene Decomposition and Vehicle Segmentation via Point-wise Spatio-Temporal Background Memory Bank (ST-BMB)*  
> **Lấy cảm hứng từ:** Triết lý **Memory Bank** trong **SAM 2** (Segment Anything Model 2) nhưng lật ngược lại cho bài toán giám sát giao thông qua Camera CCTV cố định: **Mặt đường là thực thể bất biến, phương tiện là biến cố thời gian**.  
> **Tác giả:** Nghiên cứu Khoa học Camera Giao thông TP.HCM (IC4SD-TrafficSuite).

---

## 1. Triết Lý & Nguyên Tắc Cốt Lõi (Fixed CCTV Surveillance, Khung Hình Cách Nhau 5 Phút)

Coi BMB là **bộ nhớ quan sát của một cảnh tĩnh**, không phải bộ nhớ chuyển động thông thường:
1. **Đọc theo cùng vị trí không gian (Point-wise Temporal Attention):** Do camera cố định, patch $l$ ở frame hiện tại tương ứng chính xác với patch $l$ trong quá khứ. Thay vì attention toàn cục $O(L^2 \cdot T)$, cơ chế chỉ tính attention dọc theo trục $T$ tại từng vị trí $L$, giảm độ phức tạp xuống $O(L \cdot T)$ ($256 \times$ nhanh hơn).
2. **Ghi có cổng theo độ tin cậy nền (Gated Write):** Chỉ ghi nhận các token ở vùng sạch bóng xe dựa trên `valid` mask $(B, T, L) \in [0, 1]$ (với $\text{valid} = 1 - \alpha_p$). Chỗ đang có xe tuyệt đối không được ghi đè vào nền.
3. **Chọn frame theo độ tương đồng ánh sáng (Photometric Similarity):** Thay vì hàng đợi FIFO máy móc, ưu tiên duy trì các frame có phân bố ánh sáng tương đồng hoặc đa dạng để đối chiếu.
4. **Token Detach bắt buộc:** Cắt đứt hoàn toàn đồ thị gradient lịch sử `bg_tokens.detach()` khi ghi vào BMB, chống phình VRAM và triệt tiêu triệt để hiện tượng gian lận shortcut.
5. **Bản đồ nền dài hạn (Long-term Background Memory EMA):**
   $$M = (1 - \beta) M + \beta \cdot z.\text{detach}(), \quad \text{với } \beta = \eta \cdot (1 - \alpha_p)$$
   Vùng có xe ($\alpha_p \to 1 \Rightarrow \beta \to 0$) bảo toàn nền chuẩn, vùng sạch ($\alpha_p \to 0 \Rightarrow \beta = \eta$) cập nhật mượt mà.
6. **Anchor Selection thông minh:** Frame có tỷ lệ phương tiện nhỏ nhất ($\text{mean}(\alpha_p)$ nhỏ nhất) trong chuỗi tự động được chọn làm anchor chuẩn dài hạn.

---

## 2. Kiến Trúc Luồng Hoạt Động (Read - Predict - Write)

```
[Frame t] ──> ViT Backbone + 2D Sin-Cos PE ──> Visual Tokens Q (B, L, D)
                                                     │
                                                     ▼
[BMB: (B, T, L, D) + Valid Mask] ──(READ)──> Point-wise Temporal Attention
                                                     │ (Bias: log(valid))
                                                     ▼
                                        Conditioned Visual Tokens
                                                     │
                                                     ▼
                                         MultiScaleDecompDecoder
                                                     │
                 ┌───────────────────────────────────┼───────────────────────────────────┐
                 ▼                                   ▼                                   ▼
          Mặt nạ xe (α_t)                     Tiền cảnh xe (F_t)                  Mặt đường (B_t)
                 │                                   │                                   │
                 └───────────────────────────────────┼───────────────────────────────────┘
                                                     ▼
                                         Background Memory Encoder
                                                     │
                                            (GATED WRITE & DETACH)
                                                     ▼
                                         [BMB: Update cho Frame t+1]
```

### 2.1 Cấu Trúc Bộ Nhớ Background Memory Bank (BMB)
* **Tensor bộ nhớ 4D:** Lưu trữ dạng `(B, T, L, D)` kèm valid mask `(B, T, L)`.
* **Spatial Positional Embedding:** Nhúng tọa độ không gian 2D Sin-Cos chuẩn cho từng vị trí patch $(H_p, W_p)$.
* **Temporal Embedding thực tế:** Thay thế chỉ số 0..4 vô nghĩa bằng thông tin vật lý:
  $$f_{\text{time}} = \left[\log(1 + \Delta t_{\text{minutes}}),\, \sin\left(\frac{\text{hour}}{24} \cdot 2\pi\right),\, \cos\left(\frac{\text{hour}}{24} \cdot 2\pi\right)\right] \xrightarrow{\text{MLP}} \mathbb{R}^D$$

### 2.2 Point-wise Temporal Attention & Background Familiarity Map
* **Attention Bias:**
  $$s_{b, h, l, t} = \frac{Q_{b, l, h} \cdot K_{b, t, l, h}^\top}{\sqrt{d_h}} + \log(\max(\text{valid}_{b, t, l}, 10^{-3}))$$
  $$w = \operatorname{Softmax}_t(s), \quad \text{Output}_{b, l} = \sum_{t} w_{b, h, l, t} V_{b, t, l, h}$$
* **Point-wise Background Familiarity Map:**
  $$\text{sim}_{b, t, l} = \cos(Q_{b, l}, K^{\text{raw}}_{b, t, l}) + (\text{valid}_{b, t, l} - 1) \cdot 1.5$$
  $$M_{\text{fam}} = \sigma\left(\tau_{\text{temp}} \cdot (\max_t(\text{sim}_{b, t, l}) - \tau_{\text{shift}})\right)$$
  (Tính trước khi cộng Temporal PE, chuẩn hóa bằng tham số học được).

---

## 3. Hệ Thống Hàm Mất Mát Chống Suy Biến

### 3.1 Cross-Frame Background Consistency Loss ($\mathcal{L}_{\text{cross}}$)
Triệt tiêu hoàn toàn 2 nghiệm suy biến kinh điển ($B=I, \alpha=0$ và $\alpha=1, F=I$). Nền dự đoán $\hat{B}_t$ ở frame $t$ phải giải thích được frame $s$ tại những vùng frame $s$ không có xe:
$$\text{vis}_s = (1 - \hat{\alpha}_s).\text{detach}()$$
$$\mathcal{L}_{\text{cross}} = \frac{1}{W(W-1)} \sum_{t \neq s} \frac{\sum_{c, h, w} \text{vis}_s \odot |\operatorname{Align}(\hat{B}_t \to s) - I_s|}{3 \sum_{h, w} \text{vis}_s + \epsilon}$$

### 3.2 Temporal Uncertainty Loss ($\mathcal{L}_{\text{temp\_bg}}$)
* **Detach frame trước:** $\hat{B}_{t-1}.\text{detach}()$, $\hat{\sigma}_{t-1}.\text{detach}()$.
* **Đồng bộ mẫu số:** $\bar{\sigma} = 0.5(\hat{\sigma}_t + \hat{\sigma}_{t-1}.\text{detach}())$, với $\hat{\sigma} = \operatorname{softplus}(x) + 0.01$.
* **Chroma Invariance:** Nâng sàn mẫu số lên $\ge 0.05$ chống bất ổn trong bóng râm và ban đêm.
* **Khóa chặn Adaptor:** Regularizer $\|g - 1\|^2 + \|b\|^2 + (1 - w_{\text{illum}})^2$ kéo $w_{\text{illum}} \to 1$.

### 3.3 One-Sided Familiarity Loss ($\mathcal{L}_{\text{fam}}$)
$$\mathcal{L}_{\text{fam}} = \operatorname{mean}\left(\operatorname{ReLU}(M_{\text{fam}} - 0.85) \odot \hat{\alpha}\right)$$
Chỉ phạt khi vùng cực kỳ quen thuộc với nền đường ($M_{\text{fam}} > 0.85$) mà bị gán là xe cộ.

### 3.4 Edge-Aware Total Variation ($\mathcal{L}_{\text{edge\_tv}}$)
Làm mịn mặt nạ xe có định hướng theo cạnh biên ảnh gốc:
$$\mathcal{L}_{\text{edge\_tv}} = \operatorname{mean}\left(|\nabla_x \alpha| e^{-10 |\nabla_x I|} + |\nabla_y \alpha| e^{-10 |\nabla_y I|}\right)$$

---

## 4. Cấu Trúc Mã Nguồn

```
direction_streaming_bmb/
├── __init__.py          # Khởi tạo package và xuất khẩu API chuẩn
├── memory.py            # BackgroundMemoryBank (Tensor 4D, Gated Write, EMA, Real-time PE)
├── attention.py         # Point-wise Temporal Attention & Background Familiarity Map
├── models.py            # StreamingDecompositionNet, MemoryEncoder, Decoder, Adaptor
├── losses.py            # StreamingDecompositionLoss & Cross-Frame Loss
├── dataset.py           # SlidingWindowTrafficDataset (Regex an toàn, max_gap, Cam-split)
├── train.py             # Pipeline huấn luyện chuỗi thời gian & Warm-up schedule
├── infer.py             # Pipeline suy luận streaming bóc tách xe & xuất GIF/metrics
└── README.md            # Tài liệu khoa học kỹ thuật
```

---

## 5. Hướng Dẫn Sử Dụng

### 5.1 Huấn luyện (Training)
```bash
python direction_streaming_bmb/train.py \
    --data_dir direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences \
    --save_dir checkpoints/direction_streaming_bmb \
    --backbone dinov3_vits16 \
    --weights checkpoints/dinov3_vits16_model.safetensors \
    --window_size 5 \
    --stride 1 \
    --img_size 256 \
    --epochs 15 \
    --warmup_epochs 3 \
    --memory_dropout 0.1 \
    --lambda_cross 1.5 \
    --lambda_temp 2.0 \
    --batch_size 2 \
    --lr 2e-4
```

### 5.2 Suy luận Streaming & Bóc tách Xe cộ (Inference)
```bash
python direction_streaming_bmb/infer.py \
    --checkpoint checkpoints/direction_streaming_bmb/best_streaming_model.pth \
    --input_dir direction_data_article/zenodo_bundle/sample_preview/sample_camera_sequences \
    --output_dir direction_streaming_bmb/output_inference \
    --img_size 256 \
    --save_gif
```

---

## 6. Kiểm Thử Đơn Vị (Unit Tests)

Chạy toàn bộ test suite đã cập nhật:
```bash
python -m unittest tests/test_direction_streaming_bmb.py
```
Kết quả kiểm thử bao phủ 100%:
* `test_01`: Background Memory Bank (Spatial PE, Real-time PE, Gated Write, Detach & EMA).
* `test_02`: Point-wise Temporal Attention $O(L \cdot T)$, bias $\log(\text{valid})$, Familiarity Map & Fallback.
* `test_03`: StreamingDecompositionNet forward sequence, patch $\alpha_p$, softplus $\sigma \ge 0.01$.
* `test_04`: Cross-Frame Background Loss, detached prev-frame, Adaptor regularizer, Backward gradient.
* `test_05`: Dataset safe regex timestamp, max_gap filtering, camera-based split, Letterbox resize.
* `test_06`: Photometric Illumination Adaptor gain/bias/soft_weight estimation.
* `test_07`: End-to-end training loop với Memory Dropout.
