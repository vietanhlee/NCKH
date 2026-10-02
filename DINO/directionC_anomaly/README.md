# Hướng C: Phát Hiện Sự Cố Bất Thường Kéo Dài Trong Giám Sát Giao Thông Đô Thị

> **Paper Title Candidate:** *Persistence-Aware, Camera-Conditioned Anomaly Detection for City-Scale Traffic Surveillance under Sparse Sampling*  
> **Target:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / Transportation Research Part C / Pattern Recognition / EAAI  

---

## 1. Giới thiệu & Động lực Khoa học
Các đô thị lớn như TP.HCM thường xuyên đối mặt với các sự cố bất thường giao thông nghiêm trọng:
- **Ngập lụt mặt đường:** Do triều cường hoặc mưa giông nhiệt đới kéo dài.
- **Xe chết máy / Tai nạn dừng đỗ:** Gây phong tỏa cục bộ làn đường.
- **Vật cản nguy hiểm / Công trình tạm thời:** Rào chắn thi công, cây ngã đổ.
- **Lỗi kỹ thuật camera:** Góc quay bị rung giật hoặc gió thổi xoay hướng, ống kính bị che mờ.

**Thách thức lớn:**
- Dữ liệu camera giao thông lấy mẫu thưa (1 frame mỗi 10-60 giây), không thể dùng Optical Flow hay Object Tracking liên tục.
- Sự cố giao thông rất hiếm (long-tail event) và không có nhãn gán sẵn.
- Cần phân biệt rõ:
  1. *Xe cộ đang di chuyển bình thường* (chỉ lướt qua trong 1-2 frames $\implies$ hiện tượng thoáng qua).
  2. *Sự cố giao thông thực sự* (vật cản nằm yên, vũng nước ngập duy trì qua nhiều frames).
  3. *Lỗi camera* (vùng ngoài đường như tòa nhà, cột đèn bị dịch chuyển/che mờ).

---

## 2. Kiến trúc Phương pháp

### 2.1. Feature Extraction & Road-Aware Masking (`features.py`)
- Sử dụng backbone **Meta DINOv3 (đóng băng)** để trích xuất biểu diễn patch-level giàu ngữ nghĩa không gian.
- Chiếu đặc trưng xuống không gian compact 128 chiều bằng phép chiếu trực giao ngẫu nhiên (Orthogonal Projection) nhằm bảo toàn khoảng cách Euclidean trong khi giảm mạnh tiêu thụ bộ nhớ.
- Hạ mẫu Mặt nạ Lòng đường (`Road Mask`) về cấp độ patch: phân tách rạch ròi giữa patch mặt đường $p \in R$ và patch tĩnh ngoài đường $p \in \text{Static}$.

### 2.2. Temporal Feature Pooling (`pooling.py`)
Gộp đặc trưng qua cửa sổ trượt $W$ khung hình bằng trung vị (Median Filter) trong không gian đặc trưng:
$$\tilde{F}_t(p) = \operatorname{median}_{w=0}^{W-1} f_{t-w}(p)$$
- **Triệt tiêu hoàn toàn xe cộ di chuyển thoáng qua** vì xe chỉ ghé qua patch trong thời lượng ngắn.
- **Bảo toàn và làm sắc nét các biến đổi kéo dài** (ngập nước, vật cản, tai nạn dừng đỗ).

### 2.3. Coreset Normal Memory Bank (`bank.py`)
- Ngân hàng vector đặc trưng chuẩn được phân vùng theo từng Camera và Khung giờ trong ngày (Đêm, Cao điểm sáng, Trưa, Cao điểm chiều).
- Áp dụng thuật toán **K-Center Greedy Selection (PatchCore-style)** nén 90% kích thước bộ nhớ mà vẫn bảo toàn tối đa bán kính bao phủ không gian đặc trưng bình thường.

### 2.4. Anomaly Scoring & Camera Fault Disentanglement (`score.py`, `camera_fault.py`)
- Điểm bất thường patch: Khoảng cách Euclidean $L_2$ tới láng giềng gần nhất trong Memory Bank:
  $$a_t(p) = \min_{m \in \text{Bank}} \|\tilde{F}_t(p) - m\|_2$$
- Điểm bất thường mặt đường $s_{\text{road}}$: Trung bình Top 5% patch có điểm cao nhất trong Road Mask.
- Điểm bất thường ngoại cảnh $s_{\text{static}}$: Trung bình Top 5% patch ngoài Road Mask.
- **Tách lỗi camera:** Nếu $s_{\text{static}}$ tăng vọt so với $s_{\text{road}}$ $\implies$ Cảnh báo `CAMERA_FAULT`. Nếu chỉ $s_{\text{road}}$ tăng vọt $\implies$ Cảnh báo `TRAFFIC_INCIDENT`.

### 2.5. Persistence Filtering & Event Tracking (`events.py`)
- Ngưỡng kích hoạt được tự động hiệu chuẩn bằng phân vị 99.5% trên các ngày bình thường (đảm bảo tỷ lệ báo động sai < 1 lần/camera/ngày).
- Chỉ xác nhận sự kiện khi điểm bất thường duy trì vượt ngưỡng liên tục trong ít nhất $N$ cửa sổ ($N \ge 3$).

---

## 3. Cấu trúc Thư mục
```
directionC_anomaly/
├── features.py          # Trích xuất patch token DINOv3 + Road mask
├── pooling.py           # Temporal Feature Pooling (Median cửa sổ W)
├── bank.py              # Coreset Normal Memory Bank (K-Center Greedy)
├── score.py             # Điểm bất thường k-NN & sinh Heatmap
├── camera_fault.py      # Phân tách Sự cố Giao thông vs Lỗi Camera
├── events.py            # Persistence Filter & Quản lý vòng đời sự kiện
├── synth_events.py      # Bộ sinh sự cố tổng hợp có Ground-truth
├── evaluate.py          # Pipeline đánh giá định lượng (AUROC, Delay, False Alarms)
└── README.md
```

---

## 4. Hướng dẫn Chạy Thử nghiệm

```bash
# Chạy pipeline đánh giá mẫu
python directionC_anomaly/evaluate.py
```
