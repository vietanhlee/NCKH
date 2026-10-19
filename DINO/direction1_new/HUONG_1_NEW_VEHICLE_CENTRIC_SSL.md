# Hướng 1 Mới — Vehicle-Centric SSL Từ Chuỗi Ảnh Thưa CCTV (TAM + AGM + SRS + RIC + CamAdv)

> **Tên đề xuất khoa học:** *Vehicle-Centric Self-Supervised Learning from Sparse Traffic Surveillance Imagery via Spatio-Temporal Atypicality, Counterfactual Context Swapping, and Same-Camera Contrastive Learning*  
> **Mục tiêu tạp chí:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / IEEE Transactions on Intelligent Transportation Systems (T-ITS) / CVPR / ECCV  
> **Mã nguồn thực nghiệm:** `DINO/direction1_new/`  

---

## 1. Bối Cảnh, Lỗ Hổng Lý Thuyết Cũ & Đột Phá Mới

### 1.1 Thách thức từ Camera Giám Sát Cố Định (CCTV) Snapshot Thưa
- Camera CCTV tại đô thị (như 608 camera TP.HCM) chụp ảnh ngắt quãng 3–5 phút/frame, không có luồng video liên tục 30 fps $\implies$ Không thể dùng Optical Flow hay tracking liên tục.
- 70%–85% khung hình là cảnh nền tĩnh (nhựa đường, vỉa hè, nhà cửa).
- Giờ cao điểm hoặc đèn đỏ, xe máy dừng đỗ dày đặc suốt 15–30 phút $\implies$ Thuật toán trừ nền median cổ điển bị ghosting (coi xe là nền).

### 1.2 Phản Biện Học Thuật & Đột Phá Kỹ Thuật Đã Giải Quyết Triệt Để
1. **Triệt tiêu "Camera ID Shortcut Trap"**:
   - *Vấn đề cũ*: SRS hoán nền cùng camera chỉ bất biến thời tiết; góc quay, vỉa hè và công trình không đổi khiến mô hình vẫn học tắt Camera ID.
   - *Đột phá mới*:
     + Kích hoạt **`CameraGroupedSampler`**: gom nhiều frame khác nhau của cùng camera vào một batch.
     + Kích hoạt **`RICLoss`** (Regional Instance Contrastive): Tính vector vùng xe $\mathbf{z}_{\text{veh}} = \frac{\sum_p \pi_p \mathbf{h}_p}{\sum_p \pi_p + \epsilon}$. Dùng các frame khác của cùng camera làm *hard negatives* (nền giống hệt nhau, chỉ khác xe) $\implies$ Bắt buộc mạng phải dồn toàn bộ sự chú ý vào xe để phân biệt.
     + Tích hợp **`CameraAdversarialHead`** với **Gradient Reversal Layer (GRL)**: Phạt trực tiếp việc mã hóa Camera ID trong CLS token với hệ số đảo ngược gradient $-\lambda_{\text{adv}}$.
2. **Chống Ô Nhiễm Xe ở Làn Đông trong TAM**:
   - *Vấn đề cũ*: Xe máy ở làn đông lặp lại thường xuyên có thể tạo cụm tĩnh $w \ge 0.15$; update online trên frame có xe gây thiên lệch tự bao gồm (self-inclusion bias).
   - *Đột phá mới*:
     + **Gated Online Update**: Điều biến tốc độ cập nhật cụm tĩnh theo $\eta_{\text{eff}}(p) = \eta \cdot (1 - \pi(p))$. Vị trí nào đang có xe ($\pi \to 1$) thì tốc độ cập nhật triệt tiêu về 0, bảo vệ hoàn toàn trạng thái nền tĩnh $\mu_k$.
     + **Tính Atypicality Trước Khi Update**: Loại trừ hoàn toàn self-inclusion bias.
     + **Frame-level Photometric Normalization**: Trừ dịch chuyển đồng biến toàn frame trước khi tính atypicality để khử nhiễu bóng mây / đèn đêm.
     + **Vectorized PyTorch K-means**: Khởi tạo $P=448$ patch song song trên tensor GPU, tăng tốc 50x-100x.
3. **Nâng cấp AGM Masking**:
   - Bổ sung **Block-wise Masking (`--mask_mode block`)**: Che kín các khối $2 \times 2$ quanh tâm cụm xe thay vì các patch rải rác, ép mô hình phải học ngữ nghĩa hình học trọn vẹn của xe.
4. **Sửa Mâu Thuẫn Code vs Tài Liệu**:
   - Truyền `--ibot_beta 0.5` vào `iBOTPatchLoss` để kích hoạt nhánh weighting theo $\pi$.
   - Lịch trình **Cosine EMA Momentum Schedule** ($0.996 \to 1.0$) cho Teacher.
   - Tách biệt LR: Backbone LR thấp hơn Head LR (`--backbone_lr_scale 0.2`) và weight decay $= 0.0$ cho bias/LayerNorm.

---

## 2. Kiến Trúc Hệ Thống & Hàm Mất Mát

```
                          [ Frame x1 (Cam c, Time t1) ]
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
         [ Frozen DINOv3 + PCA ]             [ Random Photometric Jitter ]
                     │                                 │
                     ▼                                 ▼
          [ TAM Gated Update ]                [ SRS Swap Nền Liên Tục ]
       => Foreground Map π ∈ [0, 1]              từ Frame x2 (Time t2)
                     │                                 │
                     ├────────────────────────┐        │
                     ▼                        ▼        ▼
           [ AGM Block Mask ]               [ Student ViT ] (Ảnh x_srs + Mask)
                     │                        ├── s_cls  ──► [ CamAdv GRL Head ]
                     │                        ├── s_patches ──► [ Vehicle Pooling z_s ]
                     │                        └── s_logits (DINO & iBOT)
                     │                                 ▲
                     │          Chưng Cất              │
                     └────────────────────────► [ Teacher ViT EMA ] (Ảnh gốc x1)
                                                  ├── t_cls
                                                  ├── t_patches ──► [ Vehicle Pooling z_t ]
                                                  └── t_logits
```

### Hàm Mất Mát Toàn Phần:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}}(\beta) + 0.02 \mathcal{L}_{\text{KoLeo}} + \lambda_{\text{ric}} \mathcal{L}_{\text{RIC}} + \lambda_{\text{adv}} \mathcal{L}_{\text{CamAdv}}$$

1. **$\mathcal{L}_{\text{DINO}}$**: Cross-entropy distillation giữa Student CLS logits và Teacher CLS logits với Centering & Sharpening.
2. **$\mathcal{L}_{\text{iBOT}}(\beta)$**: MIM loss trên các patch bị che bởi AGM với trọng số tiền cảnh:
   $$\mathcal{L}_{\text{iBOT}} = -\frac{1}{|\mathcal{M}|} \sum_{p \in \mathcal{M}} (1 + \beta \cdot \pi_p) \sum_{k} P_t(h_p)^{(k)} \log P_s(h_p)^{(k)}$$
3. **$\mathcal{L}_{\text{RIC}}$** (Regional Instance Contrastive): Kéo gần $\mathbf{z}_{\text{veh}}^{(s)}$ và $\mathbf{z}_{\text{veh}}^{(t)}$ của cùng frame, đồng thời đẩy xa các frame khác của cùng camera (hard negatives).
4. **$\mathcal{L}_{\text{CamAdv}}$**: Dự đoán Camera ID qua Gradient Reversal Layer để backbone tự triệt tiêu đặc trưng định danh camera.

---

## 3. Cấu Trúc Mã Nguồn Module (`DINO/direction1_new/`)

```
direction1_new/
├── __init__.py               # Public API exports
├── features.py               # FrozenExtractor (DINOv3 feature + PCA 64D)
├── tam.py                    # PositionStats (Gated Update, Vectorized K-means) + GMMCalibrator
├── masking.py                # AGM (Stratified Gumbel Top-K, Block-wise, Random mode)
├── srs.py                    # Static-Region Swap (Continuous BG swap, Color Match)
├── losses.py                 # DINOLoss, iBOTPatchLoss, KoLeoLoss, RICLoss, CameraAdversarialHead
├── sampler.py                # CameraGroupedSampler (gom nhóm camera cho hard-negatives)
├── train.py                  # Huấn luyện SSL hoàn chỉnh, Resume C^0, Smart Hyperparams
├── eval_camera_probe.py      # Linear Probe đo độ rò rỉ Camera ID (chứng minh chống shortcut)
├── eval_downstream.py        # Đo Silhouette score phân tách xe/nền & Effective Rank
├── run_ablations.py          # Ma trận tự động chạy 7 thí nghiệm ablation đối chứng
├── toy.py                   # Demo trực quan hóa nhanh
└── README.md                 # Tài liệu hướng dẫn sử dụng chi tiết
```

---

## 4. Hướng Dẫn Thực Thi (Đứng Tại Thư Mục `DINO`)

> **Lưu ý đường dẫn:** Đứng tại thư mục `DINO` và truyền `--data_dir /workspace/output`.

### 4.1 Huấn Luyện Chính Thức Hướng 1 Mới
```bash
python direction1_new/train.py \
    --data_dir /workspace/output \
    --model_name dinov3_vits16 \
    --output_dir checkpoints/direction1_new \
    --epochs 12 \
    --session_epochs 4 \
    --batch_size 4 \
    --lr 5e-5 \
    --backbone_lr_scale 0.2 \
    --mask_mode block \
    --mask_ratio 0.35 \
    --srs_continuous \
    --ibot_beta 0.5 \
    --use_ric \
    --lambda_ric 0.15 \
    --use_cam_adv \
    --lambda_cam_adv 0.05 \
    --use_cam_sampler \
    --cams_per_batch 4 \
    --frames_per_cam 2
```

### 4.2 Đánh Giá Shortcut Camera ID (Camera-ID Linear Probe)
Đo độ chính xác phân loại Camera ID để chứng minh mô hình không bị "học tắt" góc máy:
```bash
# 1. Đánh giá trên checkpoint Hướng 1 mới
python direction1_new/eval_camera_probe.py \
    --data_dir /workspace/output \
    --checkpoint checkpoints/direction1_new/last_checkpoint.pth \
    --output_json checkpoints/direction1_new/probe_trained.json

# 2. Đánh giá trên DINOv3 gốc (Pretrained Baseline để so sánh)
python direction1_new/eval_camera_probe.py \
    --data_dir /workspace/output \
    --output_json checkpoints/direction1_new/probe_baseline.json
```

### 4.3 Đánh Giá Chất Lượng Biểu Diễn Phương Tiện (Downstream Evaluation)
Đo khoảng cách Cosine phân tách Xe vs Nền và Effective Rank (chống Dimensional Collapse):
```bash
python direction1_new/eval_downstream.py \
    --data_dir /workspace/output \
    --checkpoint checkpoints/direction1_new/last_checkpoint.pth \
    --output_json checkpoints/direction1_new/downstream_metrics.json
```

### 4.4 Chạy Ma Trận 7 Thí Nghiệm Ablation Study
Tự động chạy và so sánh 7 cấu hình đối chứng khoa học:
```bash
# Xem trước danh sách và câu lệnh các thí nghiệm (Dry-Run):
python direction1_new/run_ablations.py --data_dir /workspace/output --dry_run

# Chạy tự động toàn bộ 7 biến thể (mỗi biến thể 2 epochs):
python direction1_new/run_ablations.py \
    --data_dir /workspace/output \
    --output_root checkpoints/ablations \
    --epochs 2 \
    --batch_size 4 \
    --run_all
```
