# 🚦 TỔNG QUAN VÀ BÁO CÁO TIẾN ĐỘ BÀI BÁO DỮ LIỆU (DATA ARTICLE Q1)
## BỘ DỮ LIỆU: HCMC-TrafficSSL (608 CAMERA TP.HCM - KHÔNG NHÃN)

> **Tài liệu kiểm soát tiến độ & Hướng dẫn sử dụng nhanh cho tác giả**  
> **Cập nhật ngày:** 06/10/2026 | **Tác giả:** Le Viet-Anh et al.

---

## 📌 1. BẢN CHẤT DỰ ÁN & BỘ DỮ LIỆU

- **Tên bộ dữ liệu**: **HCMC-TrafficSSL: A City-Scale Unlabeled Video-Sequence and Spatial Graph Benchmark for Urban Mixed Traffic Representation Learning**.
- **Bản chất**: **Dữ liệu hoàn toàn KHÔNG CÓ NHÃN (Raw Unlabeled Sequential Camera Imagery)**.
  - Không có bounding box, không có nhãn phân loại hay ground-truth đếm xe có giám sát.
  - Là chuỗi hình ảnh thực tế 24/7 thu thập trong **7 ngày liên tục** từ **608 camera giám sát cố định** của TP.HCM.
  - Tổng dung lượng ước tính: $\approx 80 \text{ GB}$ (ảnh định dạng RGB JPEG).
- **Mục đích cốt lõi**: Phục vụ làm nguồn tài nguyên trực tiếp cho toàn bộ hệ sinh thái nghiên cứu trong thư mục [`DINO/`](file:///g:/nckh/DINO):
  - **Hướng H1 & HG** ([`DINO/directionG_camera_ssl`](file:///g:/nckh/DINO/directionG_camera_ssl)): Học tự giám sát bất biến camera (TAM + AGM + SRS), hoán đổi vùng tĩnh đa ngày mà không cần ảnh nền.
  - **Hướng H2** ([`DINO/direction2_scene_decomposition`](file:///g:/nckh/DINO/direction2_scene_decomposition)): Tự bóc tách nền đường tĩnh và tiền cảnh xe cộ động.
  - **Hướng H6** ([`DINO/direction6_anomaly_detection`](file:///g:/nckh/DINO/direction6_anomaly_detection)): Phát hiện lỗi camera và sự cố giao thông không giám sát (Temporal Feature Median Pooling).
  - **Hướng H7** ([`DINO/direction7_traffic_forecasting`](file:///g:/nckh/DINO/direction7_traffic_forecasting)): Học biểu diễn đồ thị không gian - thời gian quy mô 608 nút camera.
  - **Hướng H8** ([`DINO/direction8_bg_conditioning`](file:///g:/nckh/DINO/direction8_bg_conditioning)): Đánh giá tính tổng quát hóa trên camera chưa từng thấy (Unseen Cameras).

---

## 🏛️ 2. TẠP CHÍ MỤC TIÊU & TÌNH TRẠNG BẢN THẢO

| Mục Tiêu | Tạp Chí | Phân Hạng | Tình Trạng Hiện Tại |
| :--- | :--- | :--- | :--- |
| **Đề xuất 1 (Chính)** | **Data in Brief** (Elsevier) | Scopus Q1/Q2, Gold Open Access | **ĐÃ HOÀN TẤT & BIÊN DỊCH PDF THÀNH CÔNG** (9 trang) |
| **Đề xuất 2 (Nâng cao)** | **Scientific Data** (Nature Portfolio) | Q1, Impact Factor ~9.8 | Đầy đủ nội dung theo chuẩn *Data Descriptor* |

### Vị trí các tệp bản thảo bài báo:
- 📄 **File LaTeX chính**: [`paper/data_article/main.tex`](file:///g:/nckh/paper/data_article/main.tex)
- 📚 **File BibTeX trích dẫn**: [`paper/data_article/references.bib`](file:///g:/nckh/paper/data_article/references.bib)
- 📑 **Bảng Specifications (Data in Brief)**: [`paper/data_article/tables/tab_specifications.tex`](file:///g:/nckh/paper/data_article/tables/tab_specifications.tex)
- 📊 **Bảng Thống kê thực nghiệm**: [`paper/data_article/tables/tab_summary_stats.tex`](file:///g:/nckh/paper/data_article/tables/tab_summary_stats.tex)
- 📕 **Tệp PDF bài báo hoàn chỉnh (9 trang)**: [`paper/data_article/main.pdf`](file:///g:/nckh/paper/data_article/main.pdf)

---

## 📦 3. CẤU TRÚC GÓI ĐÓNG GÓI ZENODO ([`zenodo_bundle/`](file:///g:/nckh/zenodo_bundle))

Thư mục này được tổ chức theo tiêu chuẩn dữ liệu mở quốc tế FAIR Data Principles, sẵn sàng để nộp lên Zenodo:

```text
g:/nckh/zenodo_bundle/
├── LICENSE                                    # Giấy phép mở CC BY 4.0
├── CITATION.cff                               # Định dạng trích dẫn chuẩn Citation File Format
├── .zenodo.json                               # Siêu dữ liệu cấu hình Zenodo tự động
├── data_dictionary.md                         # Từ điển dữ liệu chuẩn ISO/W3C
├── README.md                                  # Hướng dẫn quốc tế đầy đủ
├── checksums.sha256                           # Mã băm SHA-256 xác thực tính toàn vẹn 61 tệp
│
├── metadata/                                  # Dữ liệu hình học & topo
│   ├── routes.csv                # ID, tọa độ kinh/vĩ, loại đường, độ cao camera
│   └── road_network_distance.xlsx    # Ma trận khoảng cách định tuyến 608x608 từ OSM
│
├── sample_preview/sample_camera_sequences/    # Tập ảnh mẫu mini (50 ảnh) để chạy thử nghiệm tức thì
│
└── code/                                      # Module Python nạp dữ liệu chuẩn production
    ├── __init__.py                            # Package init
    ├── load_unlabeled_images.py               # Lớp TrafficCameraUnlabeledDataset
    ├── camera_sampler.py                      # CameraGroupedSampler & TemporalSequenceSampler
    └── graph_utils.py                         # Load đồ thị, RBF kernel, Scaled Chebyshev Laplacian
```

---

## 🔬 4. KẾT QUẢ THỰC NGHIỆM ĐO ĐẠC KIỂM ĐỊNH (TECHNICAL VALIDATION)

Toàn bộ các bài kiểm tra chất lượng kỹ thuật trong [`scripts/validation/`](file:///g:/nckh/scripts/validation) đã được thực thi trên dữ liệu thực tế và đưa trực tiếp vào bảng số liệu của bài báo:

| Vòng Kiểm Định | Tệp Mã Nguồn | Chỉ Số Thực Nghiệm Đo Được |
| :--- | :--- | :--- |
| **V1: Tính Toàn Vẹn Ảnh** | [`validate_image_integrity.py`](file:///g:/nckh/scripts/validation/validate_image_integrity.py) | **100% hợp lệ** (loại bỏ hoàn toàn các ảnh lỗi truyền dẫn municipal placeholder 284x177); Dung lượng trung bình: $65.87 \pm 9.19$ KB; Kênh màu: 3-channel RGB. |
| **V2: Độ Bao Phủ Thời Gian** | [`validate_temporal_coverage.py`](file:///g:/nckh/scripts/validation/validate_temporal_coverage.py) | Bao phủ trọn vẹn 168 giờ liên tục qua 7 ngày; Tỷ lệ khung hình: 34.0% ban ngày (06:00--18:00) và 66.0% ban đêm (18:00--06:00). |
| **V3: Đa Dạng Quang Học** | [`validate_visual_diversity.py`](file:///g:/nckh/scripts/validation/validate_visual_diversity.py) | **Mean Luminance (Độ sáng)**: $97.35 \pm 15.03$ (Dải đo $[67.37, 122.37]$); **RMS Contrast**: $45.79 \pm 6.57$; **Shannon Entropy**: $7.30 \pm 0.24$ bits; Cân bằng RGB: (96.0, 98.6, 94.3). |
| **V4: Cấu Trúc Topo Đồ Thị** | [`validate_graph_topology.py`](file:///g:/nckh/scripts/validation/validate_graph_topology.py) | **2,420 liên kết định tuyến có hướng** ($d_{ij} \le 5.0$ km, độ thưa 99.34%); **Tỷ lệ bất đối xứng 73.11%** (1,270 / 1,737 cặp) phản ánh chân thực đường 1 chiều và dải phân cách cứng tại TP.HCM; Khoảng cách định tuyến trung bình: $1.11 \pm 0.98$ km (Trung vị: 0.81 km); Bậc ra trung bình: 3.98 (max: 17). |

---

## ⚡ 5. CÁCH KIỂM TRA NHANH BẰNG DÒNG LỆNH

Bạn có thể mở terminal (PowerShell) và chạy các lệnh kiểm tra ngay tại thư mục gốc:

### 1. Kiểm tra nạp dữ liệu mẫu và tính Laplacian đồ thị:
```powershell
python -c "import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'; from zenodo_bundle.code import TrafficCameraUnlabeledDataset, CameraGroupedSampler, load_road_graph, compute_chebyshev_laplacian; ds = TrafficCameraUnlabeledDataset('zenodo_bundle/sample_preview/sample_camera_sequences'); print('Dataset length:', len(ds)); img, meta = ds[0]; print('Image tensor:', img.shape, 'Station ID:', meta['station_id']); adj, _ = load_road_graph('zenodo_bundle/metadata/road_network_distance.xlsx'); lap = compute_chebyshev_laplacian(adj); print('Adj:', adj.shape, 'Laplacian:', lap.shape)"
```

### 2. Chạy các bài kiểm định kỹ thuật (Technical Validation):
```powershell
python scripts/validation/validate_image_integrity.py
python scripts/validation/validate_temporal_coverage.py
python scripts/validation/validate_visual_diversity.py
python scripts/validation/validate_graph_topology.py
```

### 3. Biên dịch lại bài báo LaTeX (nếu cần chỉnh sửa):
```powershell
cd paper/data_article
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

## ⚖️ 6. ĐẠO ĐỨC & KHUNG PHÁP LÝ

Bản thảo bài báo đã được thiết kế một phần tuyên bố độc lập nghiêm ngặt:
1. **Tuân thủ Nghị định 47/2020/NĐ-CP**: Quản lý, kết nối và chia sẻ dữ liệu số của cơ quan nhà nước tại Việt Nam (Dữ liệu mở phục vụ nghiên cứu phi thương mại).
2. **Tuân thủ Luật Bảo vệ dữ liệu cá nhân 91/2025/QH15**: Camera được gắn trên trụ cao $>6$m, góc quay toàn cảnh mặt đường; độ phân giải và nén ảnh ngăn chặn việc nhận dạng khuôn mặt người đi đường hoặc đọc rõ biển số xe (hoàn toàn **PII-free**).
3. **Chính sách gỡ bỏ (Take-down Policy)**: Quy trình công khai cho phép cá nhân hoặc cơ quan yêu cầu kiểm tra hoặc gỡ bỏ hình ảnh khi có phản ánh.
