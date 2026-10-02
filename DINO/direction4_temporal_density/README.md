# Hướng 4: Spatio-Temporal DINO for Continuous Road Space Occupancy & Congestion Level of Service (LoS) Estimation

## 1. Giới thiệu & Đóng góp Khoa học (Novelty ⭐⭐⭐⭐⭐)
Thay vì đếm từng xe (dễ sai lệch nghiêm trọng khi tắc đường xe máy che khuất nhau), bài toán kỹ thuật giao thông hiện đại (ITS) tập trung vào hai chỉ số cốt lõi:
1. **Tỷ lệ chiếm dụng mặt đường liên tục** $\rho(t) \in [0.0, 1.0]$ (Road Space Occupancy Ratio): Diện tích mặt đường bị phương tiện che phủ.
2. **Cấp độ dịch vụ giao thông (Level of Service - LoS)** theo chuẩn Highway Capacity Manual (HCM):
   - **LoS 0 (Free-Flow, Thông thoáng)**: $\rho \le 0.15$
   - **LoS 1 (Moderate, Dòng xe ổn định)**: $0.15 < \rho \le 0.35$
   - **LoS 2 (Slow, Đông đúc / Sắp quá tải)**: $0.35 < \rho \le 0.60$
   - **LoS 3 (Gridlock, Ùn tắc nghiêm trọng)**: $\rho > 0.60$
3. **Động lực học biến thiên ùn tắc** $\frac{\partial \rho}{\partial t}$: Dự báo xu hướng tắc nghẽn đang hình thành hay giải tỏa.

### Giải pháp kỹ thuật đột phá:
* **Mỏ neo vật lý quang học tự thân (Physical Ground Truth)**: Tính toán trực tiếp $\rho_{\text{phys}}(t) = \frac{\sum \mathbb{I}(\Delta > \tau)}{H \times W}$ từ trường sai khác $\Delta = |I - I_{\text{bg}}|$, cung cấp nhãn liên tục khách quan mà **không cần con người gán nhãn thủ công**.
* **Spatio-Temporal Bi-GRU Network**: DINO ViT CLS token kết hợp CNN $\Delta$-Encoder và mạng tuần hoàn Bi-GRU qua $K$ khung hình để mô hình hóa tính trơn và quán tính của dòng xe.
* **Đầu ra đa nhiệm**: Ước lượng đồng thời $\hat{\rho}(t)$, phân loại LoS 4 mức và xu hướng biến thiên.

---

## 2. Cấu trúc File
- `dataset.py`: `TemporalTrafficDataset` quét chuỗi khung hình theo timestamp, tự động trích xuất $\Delta$, tính mỏ neo $\rho_{\text{phys}}$ và nhãn LoS HCM.
- `models.py`: `SpatioTemporalDensityNet` (DINO ViT + DeltaSpatialEncoder + Bi-GRU + Occupancy Head + LoS Head + Trend Head).
- `losses.py`: `SpatioTemporalDensityLoss` (Smooth L1 Occupancy + Cross-Entropy LoS + Trend MSE + Temporal Smoothness).
- `train.py`: Pipeline huấn luyện hoàn chỉnh hỗ trợ Multi-GPU, Mixed Precision (AMP), Resume Checkpoint, báo cáo MAE Occupancy và LoS Accuracy.

---

## 3. Hướng Dẫn Chạy (CLI Execution)

```bash
python direction4_temporal_density/train.py \
    --bg_dir traffic_backgrounds \
    --origin_dir output \
    --save_dir checkpoints/direction4_temporal_density \
    --backbone dinov3_vits16 \
    --window_size 4 \
    --batch_size 8 \
    --epochs 20 \
    --lr 3e-4 \
    --device cuda
```
Hoặc tiếp tục huấn luyện từ checkpoint:
```bash
python direction4_temporal_density/train.py \
    --resume checkpoints/direction4_temporal_density/best_temporal_model.pth \
    --epochs 10
```
