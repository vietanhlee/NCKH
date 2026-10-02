# Hướng D: Dự Báo Ùn Tắc Giao Thông Quy Mô Toàn Thành Phố Trên Đồ Thị Camera

> **Paper Title Candidate:** *City-Scale Congestion Forecasting from Surveillance Camera Networks in Motorbike-Dominant Traffic*  
> **Target:** Transportation Research Part C: Emerging Technologies / IEEE Transactions on Intelligent Transportation Systems (T-ITS) / IEEE TKDE  

---

## 1. Giới thiệu & Động lực Khoa học
Phần lớn các nghiên cứu dự báo giao thông kinh điển thế giới (DCRNN, STGCN, Graph WaveNet) đều sử dụng dữ liệu từ cảm biến vòng từ (loop detectors) trên các tuyến đường cao tốc Bắc Mỹ (như METR-LA, PEMS-BAY). Tại các siêu đô thị Đông Nam Á như TP.HCM:
- **Không có mạng lưới cảm biến vòng từ ngầm**, nhưng có **hàng ngàn camera giám sát đô thị** trải rộng khắp các giao lộ.
- **Giao thông xe máy hỗn hợp cực kỳ đặc thù**: Khả năng len lỏi cao, động học ùn tắc hình thành rất nhanh, lan truyền phức tạp qua các ngã tư ngã năm và phụ thuộc mật thiết vào các cơn mưa giông nhiệt đới bất chợt.
- **Thách thức kỹ thuật thực tế**:
  1. Dữ liệu suy diễn từ hình ảnh có nhiễu (không phải số đếm cảm biến tuyệt đối).
  2. Camera giao thông thường xuyên rớt mạng, mất tín hiệu, mất điện cục bộ (dữ liệu khuyết thiếu lớn).
  3. Góc quay và độ cao lắp đặt của mỗi camera hoàn toàn khác nhau.

---

## 2. Kiến trúc Phương pháp

### 2.1. Spatial Graph Construction (`graph.py`)
- **Nút:** Mỗi camera là 1 nút trên đồ thị ($N$ nút).
- **Cạnh vật lý:** Tính khoảng cách đường bộ ngắn nhất (OpenStreetMap / Haversine) kết hợp nhân Gauss:
  $$W_{ij} = \exp\left(-\frac{\text{dist}(i, j)^2}{\sigma^2}\right)$$
- **Ma trận kề Thích ứng (Adaptive Adjacency Matrix):**
  $$\tilde{\mathbf{A}}_{\text{adp}} = \operatorname{Softmax}\left(\operatorname{ReLU}(\mathbf{E}_1 \mathbf{E}_2^T)\right)$$
  Học các liên kết giao thông tiềm ẩn giữa các nút camera không nhất thiết gần nhau về mặt địa lý.

### 2.2. Spatio-Temporal Dataset & Chronological Split (`dataset.py`)
- Dữ liệu được đưa về lưới thời gian đều $\Delta t = 5$ phút.
- Mỗi nút camera có vector đặc trưng kết hợp đa phương thức:
  - Chỉ số chiếm dụng $\rho_t^i$ hoặc phân phối nhãn mềm $q_t^i$.
  - Biểu diễn thị giác $e_t^i$: Token [CLS] từ DINOv3 nén 32 chiều.
  - Mặt nạ quan sát $m_t^i \in \{0, 1\}$ (bằng 0 khi mất tín hiệu).
- **Chia dữ liệu nghiêm ngặt theo trật tự thời gian (Chronological Split):** 70% tuần đầu Train, 10% tuần giữa Val, 20% tuần cuối Test (chống rò rỉ tương lai).
- **Node Dropout lúc huấn luyện:** Tự động che ngẫu nhiên 10–50% camera để mô hình rèn luyện tính bền vững khi mạng lưới có nhiều camera offline.

### 2.3. ST-GraphWaveNet Model (`models.py`)
- **Temporal Layer:** Gated Dilated TCN với Causal Padding:
  $$\mathbf{H} = \tanh(\mathbf{\Theta}_1 * \mathbf{X}) \odot \sigma(\mathbf{\Theta}_2 * \mathbf{X})$$
- **Spatial Layer:** Graph Convolution trên cả 2 ma trận kề (Vật lý + Thích ứng).
- **3 Đầu ra Đa nhiệm:**
  1. *Continuous Horizon Forecasting:* Dự báo $\rho$ tại $h \in \{15', 30', 60'\}$ phút.
  2. *Ordinal Congestion Classification:* Phân loại 4 mức ùn tắc có tính đến trật tự.
  3. *Congestion Onset Warning:* Dự báo xác suất bùng phát kẹt xe trong $h$ phút tới khi hiện tại đường đang thông thoáng.

### 2.4. Loss Functions (`losses.py`)
- **Masked MAE:** Chỉ tính mất mát trên các thời điểm và camera có dữ liệu thực tế ($m=1$).
- **Focal Onset Loss:** Giải quyết hiện tượng mất cân bằng mẫu cực đoan của sự kiện bùng phát kẹt xe.

---

## 3. Cấu trúc Thư mục
```
directionD_forecasting/
├── graph.py        # Xây dựng ma trận kề địa lý & Adaptive Adjacency Layer
├── dataset.py      # Lưới thời gian 5 phút, Missing mask, Chronological split
├── models.py       # ST-GraphWaveNet với Gated TCN & Multi-Task Heads
├── losses.py       # Masked MAE, Focal Onset Loss, Multi-task Loss
├── evaluate.py     # Đánh giá đa tầm h=15', 30', 60' & Khảo sát camera offline 10%-50%
└── README.md
```

---

## 4. Hướng dẫn Chạy Thử nghiệm

```bash
# Chạy đánh giá mô phỏng
python directionD_forecasting/evaluate.py
```
