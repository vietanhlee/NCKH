# Hướng 2: Self-Supervised Scene Decomposition Network (Traffic-Decompose)

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐⭐⭐)
Đây là hướng nghiên cứu có độ mới (Novelty) và giá trị khoa học cao nhất, hướng tới các hội nghị thị giác máy tính hàng đầu (CVPR / ECCV / NeurIPS):
- Huấn luyện mạng nơ-ron nhận đầu vào duy nhất là ảnh giao thông $I_{\text{origin}}$, tự động phân rã không giám sát thành **3 thành phần vật lý**:
  1. $\hat{I}_{\text{bg}}$: Lớp nền đường tĩnh không còn xe (**Unsupervised Road Inpainting**).
  2. $\hat{I}_{\text{fg}}$: Lớp phương tiện nổi (chỉ chứa xe cộ).
  3. $M_{\text{fg}} \in [0, 1]$: Mặt nạ xác suất phân bố phương tiện (Alpha Mask).
- **Tín hiệu giám sát vật lý độc nhất**: Sử dụng ảnh background thật $I_{\text{bg}}$ từ camera tĩnh làm ground-truth để giám sát nhánh $\hat{I}_{\text{bg}}$.
- Công thức tổng hợp quang học (Alpha Compositing Formulation):
  $$\hat{I}_{\text{origin}} = M_{\text{fg}} \odot \hat{I}_{\text{fg}} + (1 - M_{\text{fg}}) \odot \hat{I}_{\text{bg}}$$
- **Hàm mất mát kết hợp**:
  $$\mathcal{L} = \|\hat{I}_{\text{origin}} - I_{\text{origin}}\|_1 + \lambda_{\text{bg}} \|\hat{I}_{\text{bg}} - I_{\text{bg}}\|_1 + \lambda_{\text{sparse}} \|M_{\text{fg}}\|_1 + \lambda_{\text{tv}} \text{TV}(M_{\text{fg}})$$

---

## 2. Cấu trúc File trong Thư mục
- `dataset.py`: `DecompositionDataset` (Đồng bộ không gian và hình học giữa ảnh Origin và Background).
- `models.py`: `TrafficDecompositionNet` (Shared DINOv3 ViT, 3 nhánh Decoder: Nền đường, Xe cộ, Alpha Mask).
- `losses.py`: `DecompositionLoss` (Reconstruction Loss + Background Supervision + Mask Sparsity + Total Variation Smoothness).
- `train.py`: Pipeline huấn luyện tự động đa GPU (DataParallel, Cosine LR, Smart Checkpoint).
- `infer.py`: Script suy luận tách lớp và tự động xóa xe (Road Inpainting) trên ảnh mới bất kỳ (không cần background lúc test).

---

## 3. Hướng Dẫn Chạy Chi Tiết (CLI Execution)

### Kịch bản 1: Chạy thử nghiệm nhanh (Quick Test)
Kiểm tra khả năng học tách lớp trên 20 mẫu ảnh trong 2 epochs:
```bash
python DINO/direction2_scene_decomposition/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --epochs 2 \
    --batch_size 4 \
    --max_samples 20 \
    --save_dir checkpoints/direction2_scene_decomp/test_run \
    --device cuda
```

---

### Kịch bản 2: Huấn luyện đầy đủ trên Đa GPU (Kaggle 2x T4 / Server Lab)
Hệ thống **tự động phát hiện toàn bộ số GPU**, phân phối 3 nhánh decoder song song trên tất cả các GPU:
```bash
python DINO/direction2_scene_decomposition/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --match_strategy route_hourly \
    --backbone dinov3_vits16 \
    --img_size 256 \
    --epochs 20 \
    --batch_size 8 \
    --lr 3e-4 \
    --lambda_bg 1.5 \
    --lambda_sparse 0.05 \
    --lambda_tv 0.1 \
    --save_dir checkpoints/direction2_scene_decomp \
    --device cuda
```
*Lưu ý:* Kiểm tra ảnh tiến trình học tách lớp trực quan sinh ra sau mỗi epoch tại:
`checkpoints/direction2_scene_decomp/visual_progress/epoch_XXX.png`.

---

### Kịch bản 3: Chạy suy luận tách lớp & Xóa sạch xe (Unsupervised Road Inpainting)
Sau khi huấn luyện, bạn có thể áp dụng mô hình lên **bất kỳ ảnh giao thông nào** để tự động xuất ra 3 lớp (mặt đường sạch xe, ảnh xe, mặt nạ):

```bash
# Chạy trên 1 file ảnh cụ thể:
python DINO/direction2_scene_decomposition/infer.py \
    --weights checkpoints/direction2_scene_decomp/best_decomposition_model.pth \
    --input_path output/1_1755698811.jpg \
    --output_dir checkpoints/direction2_scene_decomp/inferred \
    --img_size 256 \
    --device cuda

# Hoặc chạy trên toàn bộ thư mục ảnh mới:
python DINO/direction2_scene_decomposition/infer.py \
    --weights checkpoints/direction2_scene_decomp/best_decomposition_model.pth \
    --input_path output \
    --output_dir checkpoints/direction2_scene_decomp/inferred_batch \
    --img_size 256 \
    --device cuda
```

*Kết quả đầu ra sinh ra gồm 3 file cho mỗi ảnh:*
- `{stem}_clean_road.jpg`: Ảnh mặt đường tĩnh sạch bóng xe (Road Inpainting).
- `{stem}_vehicles_only.jpg`: Lớp phương tiện được cô lập.
- `{stem}_density_mask.png`: Mặt nạ mật độ phương tiện liên tục.

---

## 4. Bảng Giải Thích Tham Số (Arguments)

| Tham Số | Mặc Định | Ý Nghĩa Kỹ Thuật |
|:---|:---:|:---|
| `--lambda_bg` | `1.5` | Trọng số giám sát nền từ ảnh background thật (tăng nếu mặt đường tái tạo chưa sạch xe). |
| `--lambda_sparse` | `0.05` | Trọng số ép mặt nạ thưa (L1 sparsity) — xe cộ chỉ chiếm một phần mặt đường. |
| `--lambda_tv` | `0.1` | Trọng số Total Variation (TV) ép viền mặt nạ mượt mà, không lốm đốm nhiễu hạt. |
| `--freeze_backbone` | `True` | Đóng băng DINOv3 ViT backbone để huấn luyện nhanh và tiết kiệm VRAM. |
| `--img_size` | `256` | Kích thước ảnh đầu vào và tái tạo (khuyến nghị 256x256). |
