# Hướng 4: Giám Sát & Định Vị Sự Cố Giao Thông Bất Thường Kéo Dài Kết Hợp Điều Phối Flycam Tuần Tra Tự Hành

> **Paper Title Candidate:** *Graph-Linked Persistent Traffic Anomaly Detection and Autonomous UAV Inspection for City-Scale Surveillance Networks*  
> **Target Conferences / Journals:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / Transportation Research Part C: Emerging Technologies / IEEE Transactions on Automation Science and Engineering (T-ASE)  

---

## 1. Bối cảnh & Động lực Thực tiễn Đô thị

Hệ thống giám sát giao thông đô thị quy mô lớn (như mạng lưới 600+ camera CCTV cố định tại TP.HCM) đối mặt với một nghịch lý lớn trong vận hành:
1. **Camera cố định có góc nhìn hạn chế (Blind Spots):** Camera chỉ nhìn thấy một nút giao hoặc một góc ngã tư; toàn bộ đoạn đường nối giữa hai nút camera (inter-camera edge dài 500m – 3km) là "vùng mù" thị giác.
2. **Tốc độ lấy mẫu thưa (1 frame mỗi 10 – 60 giây):** Khung hình snapshot gián đoạn không thể áp dụng Optical Flow hoặc theo dõi xe liên tục (tracking), dễ bị đánh lừa bởi phương tiện dừng đèn đỏ hoặc bóng đổ mây trời.
3. **Phản ứng thụ động & Tốn kém nhân lực:** Khi xảy ra ngập lụt, tai nạn liên hoàn hoặc cây đổ phong tỏa lòng đường, trung tâm điều hành (TOC) thường chỉ phát hiện khi dòng xe đã dồn ứ kéo dài hàng km và phải cử CSGT chạy xe máy đến hiện trường xác minh.

### 🎯 Ứng Dụng Thực Tiễn Trọng Tâm: "Early Warning & Autonomous UAV Dispatch"
Hệ thống kết hợp **Camera CCTV cố định ở các nút giao** làm cảm biến cảnh giới sớm (Early Sentinel), kết hợp cùng **Đồ thị mạng lưới đường bộ (Road Network Graph)** để tự động điều phối **Flycam/Drone tự hành** tuần tra xác minh:

```
[Camera Cố Định Nút u]                     [Camera Cố Định Nút v]
     (Trạm A)                                   (Trạm B)
        │                                          │
        ▼                                          ▼
   Phát hiện bất thường                     Lưu lượng đột ngột
   kéo dài (s_road cao)                     giảm về 0 (đứt gãy dòng xe)
        │                                          │
        └───────────────────┬──────────────────────┘
                            ▼
           [ENGINE ĐỊNH VỊ PHÂN ĐOẠN ĐƯỜNG (u, v)]
           Khẳng định sự cố nghẽn tắc trên cạnh nối
                            │
                            ▼
              [TRẠM SẠC DRONE / UAV TỰ HÀNH]
           Tự động xuất tọa độ GPS và lộ trình bay
                            │
                            ▼
              [FLYCAM TUẦN TRA ĐOẠN ĐƯỜNG (u, v)]
         Truyền video live góc nhìn trên cao (top-down view):
         - Đo độ sâu vùng ngập nước
         - Xác định vị trí tai nạn chặn làn nào
         - Dẫn đường cho xe cứu hộ & phân luồng từ xa
```

---

## 2. Các Kịch Bản Ứng Dụng Thực Tế (Real-World Use Cases)

### 🚁 Kịch bản 1: Điều phối Flycam tuần tra điểm mù giữa 2 Node camera
* **Hiện tượng:** Camera tại Nút $u$ báo điểm bất thường lòng đường kéo dài ($s_{\text{road}} > \tau$ liên tục qua 3 cửa sổ thời gian), trong khi camera tại Nút $v$ kế tiếp ghi nhận dòng xe biến mất bất thường.
* **Hành động tự động:** Hệ thống suy luận đoạn đường $(u, v)$ bị phong tỏa hoàn toàn. Lệnh điều phối (dispatch mission) lập tức được gửi tới Docking Station của Flycam gần nhất:
  * Flycam cất cánh tự động theo waypoint GPS dọc hành lang đường $(u, v)$.
  * Camera trên Flycam truyền video góc nhìn thẳng từ trên cao (Bird's Eye View), phân loại chính xác nguyên nhân (xe tải chết máy giữa cầu, sạt lở hay va chạm giao thông) và truyền tọa độ chính xác cho lực lượng cứu hộ.

### 🌊 Kịch bản 2: Cảnh báo sớm & Bản đồ ngập úng đô thị theo thời gian thực
* **Hiện tượng:** Khi mưa lớn hoặc triều cường dâng cao, mặt đường đổi màu và phản xạ mặt nước duy trì ổn định qua nhiều khung hình ($W \ge 5$).
* **Hành động tự động:** Thuật toán phát hiện sự biến đổi phân bố patch DINOv3 trên lòng đường mà không bị nhầm với bóng xe. Hệ thống tự động cập nhật bản đồ ngập lụt của thành phố, đề xuất lộ trình tránh ngập cho các ứng dụng bản đồ giao thông (Google Maps, BusMap).

### 🔧 Kịch bản 3: Tự chẩn đoán & Giám sát sức khỏe mạng lưới camera (Self-Diagnostic)
* **Hiện tượng:** Gió giật làm camera rung lắc xoay góc, hoặc mạng nhện / giọt nước mưa bám ống kính.
* **Hành động tự động:** Thuật toán phân tách rõ:
  * Nếu vùng ngoài đường (tòa nhà, vỉa hè) cũng biến đổi bất thường $\implies$ Báo lỗi kỹ thuật camera (`CAMERA_FAULT`), tự động tạo ticket bảo trì cho đội kỹ thuật.
  * Nếu vùng ngoài đường hoàn toàn ổn định, chỉ mặt đường bị biến đổi $\implies$ Khẳng định sự cố giao thông thực sự (`TRAFFIC_INCIDENT`), ngăn chặn 100% báo động giả.

---

## 3. Kiến Trúc Phương Pháp & Giải Thuật Cốt Lõi

```
 Khung hình Camera t
 (Snapshot 10-60s)
        │
        ▼
 [Meta DINOv3 ViT] ──────► Trích xuất biểu diễn Patch-level ngữ nghĩa cao
        │
        ▼
 [Road-Aware Masking] ───► Phân tách Patch Lòng đường (R) vs Ngoại cảnh (Static)
        │
        ▼
 [Temporal Feature]  ───► Median Pooling qua W khung hình: Triệt tiêu xe chạy,
 [    Pooling     ]       chỉ giữ lại biến đổi kéo dài (ngập, vật cản, tai nạn)
        │
        ▼
 [Coreset Memory]    ───► K-Center Greedy Memory Bank đại diện cho trạng thái
 [     Bank     ]         bình thường theo từng khung giờ trong ngày
        │
        ▼
 [Disentanglement]   ───► Phân tách: s_road (Sự cố) vs s_static (Lỗi camera)
        │
        ▼
 [Persistence   ]    ───► Bộ lọc bền bỉ: Xác nhận sự cố vượt ngưỡng N lần liên tiếp
 [   Filter     ]         (Đảm bảo False Alarm Rate < 1 lần/camera/ngày)
        │
        ▼
 [Graph Dispatch]    ───► Khớp cạnh đồ thị (u, v) & Kích hoạt UAV tuần tra
```

### 3.1. Trích xuất đặc trưng kháng nhiễu (`features.py`)
* Sử dụng **Meta DINOv3 ViT (đóng băng)**: trích xuất các patch token mang thông tin cấu trúc hình học và ngữ nghĩa mặt đường.
* Chiếu trực giao ngẫu nhiên (Orthogonal Random Projection) giảm chiều từ 384/768 xuống **128 chiều**, bảo toàn khoảng cách Euclidean với chi phí bộ nhớ tối thiểu.
* Mặt nạ Lòng đường (`Road Mask`) hạ mẫu xuống lưới patch: phân định rõ patch thuộc làn xe chạy ($p \in R$) và patch tĩnh ngoài đường ($p \in \text{Static}$).

### 3.2. Lọc trung vị đặc trưng theo thời gian (`pooling.py`)
Triệt tiêu phương tiện di chuyển thông thường bằng bộ lọc Median trong không gian đặc trưng qua cửa sổ trượt $W$ khung hình:
$$\tilde{F}_t(p) = \operatorname{median}_{w=0}^{W-1} f_{t-w}(p)$$
* **Xe cộ bình thường:** Chỉ lướt qua patch trong 1-2 khung hình $\implies$ Bị hàm trung vị lọc sạch 100%.
* **Sự cố thực sự (vũng nước ngập, vật cản, xe tai nạn dừng đỗ):** Tồn tại qua phần lớn cửa sổ $W$ $\implies$ Được giữ nguyên và làm nổi bật tín hiệu bất thường.

### 3.3. Coreset Normal Memory Bank (`bank.py`)
* Xây dựng bộ nhớ lưu trữ các trạng thái mặt đường bình thường được phân đoạn theo từng Camera và Khung giờ trong ngày (Sáng, Trưa, Chiều, Đêm).
* Áp dụng thuật toán **K-Center Greedy Selection (phong cách PatchCore)**: nén 90% dung lượng mà vẫn giữ nguyên độ bao phủ đa dạng của thời tiết và ánh sáng bình thường.

### 3.4. Phân tách Sự cố Giao thông vs Lỗi Camera (`camera_fault.py`, `score.py`)
Tính khoảng cách $L_2$ tới láng giềng gần nhất trong Memory Bank:
$$a_t(p) = \min_{m \in \text{Bank}} \|\tilde{F}_t(p) - m\|_2$$
* Điểm bất thường mặt đường $s_{\text{road}}$: Trung bình top 5% patch cao nhất trong Road Mask.
* Điểm bất thường ngoại cảnh $s_{\text{static}}$: Trung bình top 5% patch ngoài Road Mask.
* **Luật phân tách:**
  $$\begin{cases}
  s_{\text{static}} > \tau_{\text{static}} & \implies \mathbf{CAMERA\_FAULT} \text{ (Rung lắc, lệch góc, mờ kính)} \\
  s_{\text{road}} > \tau_{\text{road}} \text{ và } s_{\text{static}} \le \tau_{\text{static}} & \implies \mathbf{TRAFFIC\_INCIDENT} \text{ (Sự cố mặt đường thật sự)}
  \end{cases}$$

### 3.5. Bộ lọc bền bỉ & Quản lý vòng đời sự kiện (`events.py`)
* Ngưỡng phát hiện $\tau$ được tự động hiệu chuẩn ở phân vị 99.5% của các ngày bình thường.
* Cơ chế Persistence: Chỉ phát cảnh báo khi sự cố duy trì liên tục qua ít nhất $N$ cửa sổ liên tiếp ($N \ge 3$), triệt tiêu triệt để báo động giả.

---

## 4. Cấu trúc Mã Nguồn

```
direction4_anomaly_detection/
├── features.py          # Trích xuất DINOv3 patch token + Phân vùng Road/Static mask
├── pooling.py           # Temporal Feature Pooling (Lọc trung vị cửa sổ trượt W)
├── bank.py              # Coreset Normal Memory Bank (Thuật toán K-Center Greedy)
├── score.py             # Tính điểm bất thường patch & tạo Heatmap trực quan
├── camera_fault.py      # Bộ phân tách Sự cố Giao thông vs Lỗi Kỹ thuật Camera
├── events.py            # Persistence Filter & Quản lý vòng đời sự kiện (Start, Peak, End)
├── synth_events.py      # Sinh dữ liệu sự cố mô phỏng (ngập nước, vật cản, tai nạn) để benchmark
├── evaluate.py          # Pipeline đánh giá định lượng (AUROC, False Alarm Rate, Detection Delay)
└── README.md            # Tài liệu thiết kế & Hướng dẫn ứng dụng thực tiễn
```

---

## 5. Hướng dẫn Chạy Đánh giá Thực nghiệm

Chạy pipeline kiểm thử toàn diện trên dữ liệu mô phỏng sự cố thực tế:

```bash
python direction4_anomaly_detection/evaluate.py
```

Kết quả xuất ra gồm:
1. **Chỉ số định lượng:** AUROC phát hiện sự cố, AUROC phân tách lỗi camera, tỷ lệ báo động giả mỗi ngày.
2. **Visual Heatmap:** Bản đồ nhiệt vị trí vật cản/ngập nước chồng lên ảnh gốc camera.
3. **Event Timeline:** Biểu đồ đường thời gian ghi nhận thời điểm bắt đầu, đỉnh điểm và kết thúc của sự cố để kích hoạt lệnh bay cho Flycam.
