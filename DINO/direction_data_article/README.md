# 🚦 HƯỚNG MỚI DINO: DIRECTION_DATA_ARTICLE
## BỘ DỮ LIỆU: HCMC-TrafficSnap (>1.000.000 ẢNH SNAPSHOTS & ĐỒ THỊ 608 CAMERA TP.HCM)

> **Tài liệu kiểm soát tiến độ & Hướng dẫn sử dụng nhanh cho tác giả**  
> **Cập nhật ngày:** 06/10/2026 | **Tác giả:** Viet-Anh Le & Dr. Khanh Nguyen-Trong (IC4SD Lab, PTIT)

---

## 📌 1. BẢN CHẤT DỰ ÁN & VỊ TRÍ TRONG HỆ SINH THÁI DINO

Hướng nghiên cứu này đóng vai trò là **Công trình Công bố Dữ liệu Mở Quốc tế (Data Article)** độc lập, nhằm chính thức xuất bản và chuẩn hóa bộ dữ liệu thị giác giao thông thời gian thực làm nền tảng nuôi sống toàn bộ các hướng nghiên cứu trong [`DINO/`](file:///g:/nckh/DINO):

- **Tên bộ dữ liệu chính thức**: **HCMC-TrafficSnap: A City-Scale Multi-Camera Image Time-Series and Directed Spatial Graph Dataset for Heterogeneous Urban Traffic**.
- **Bản chất**: **Dữ liệu Chuỗi thời gian ảnh Snapshots không nhãn (Unlabeled Image Time-Series / Time-Lapse Snapshots)**.
  - Không có bounding box, không có nhãn phân loại hay ground-truth đếm xe có giám sát.
  - Chuỗi hình ảnh time-lapse thực tế thu thập liên tục trong **7 ngày liên tục (168 giờ, từ Thứ Hai đến Chủ Nhật)** từ **608 camera giám sát cố định** của TP.HCM.
  - Khoảng cách lấy mẫu trung bình: $\Delta T \approx 300\text{ s}$ (5 phút/ảnh $\approx 12$ ảnh/giờ/camera).
  - Quy mô tổng lượng ảnh: **$> 1.000.000$ ảnh** ($\approx 1.22$ triệu ảnh toàn bộ mạng lưới).
  - Độ phân giải gốc: **$512 \times 288$ pixels** (tỷ lệ 16:9, chuẩn streaming qHD của cổng giao thông).
  - Dung lượng trung bình: $65.87 \pm 9.19\text{ KB}$/ảnh; Tổng lưu trữ: $\approx 80\text{ GB}$.
  - Độ trễ crawl trung bình so với đồng hồ camera thực tế: $\Delta t_{\text{lag}} = 15.0 \pm 4.2\text{ s}$.
- **Phục vụ trực tiếp cho các hướng trong thư mục `DINO/`**:
  - **Hướng H1 & HG** (`direction1_bg_guided_dino`, `directionG_camera_ssl`): Học tự giám sát bất biến camera (TAM + AGM + SRS), hoán đổi vùng tĩnh đa ngày mà không cần ảnh nền.
  - **Hướng H2** (`direction2_scene_decomposition`): Tự bóc tách nền đường tĩnh và tiền cảnh xe cộ động không cần nhãn.
  - **Hướng H6** (`direction6_anomaly_detection`): Phát hiện lỗi camera và sự cố giao thông không giám sát (Temporal Feature Median Pooling).
  - **Hướng H7** (`direction7_traffic_forecasting`): Học biểu diễn đồ thị không gian - thời gian quy mô 608 nút camera kết hợp mô hình khuếch tán có hướng (Directed Diffusion Convolutions).
  - **Hướng H8** (`direction8_bg_conditioning`): Đánh giá tính tổng quát hóa trên camera chưa từng thấy (Unseen Cameras).

---

## 🏛️ 2. TẠP CHÍ MỤC TIÊU & TÌNH TRẠNG BẢN THẢO

| Mục Tiêu | Tạp Chí | Phân Hạng | Tình Trạng Hiện Tại |
| :--- | :--- | :--- | :--- |
| **Đề xuất 1 (Chính)** | **Data in Brief** (Elsevier) | Scopus Q1/Q2, Gold Open Access | **ĐÃ HOÀN TẤT & BIÊN DỊCH PDF THÀNH CÔNG** (11 trang, kèm 4 hình & 3 bảng) |
| **Đề xuất 2 (Nâng cao)** | **Scientific Data** (Nature Portfolio) | Q1, Impact Factor ~9.8 | Đầy đủ nội dung theo chuẩn *Data Descriptor* |

### Thông tin Tác giả & Định danh ORCID:
- **Viet-Anh Le** (Đầu mối kỹ thuật & Tác giả thứ nhất): `vietanh.ic4sd@ptit.edu.vn` | ORCID: [0009-0003-5748-0439](https://orcid.org/0009-0003-5748-0439)
- **Dr. Khanh Nguyen-Trong** (Giảng viên hướng dẫn & Corresponding Author): `khanhnt@ptit.edu.vn` | ORCID: [0000-0001-5175-8805](https://orcid.org/0000-0001-5175-8805)
- **Đơn vị công tác**: *Intelligent Computing for Sustainable Development Laboratory (IC4SD), Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam*.

### Vị trí các tệp bản thảo bài báo ([`paper/`](file:///g:/nckh/DINO/direction_data_article/paper)):
- 📄 **File LaTeX chính**: [`paper/main.tex`](file:///g:/nckh/DINO/direction_data_article/paper/main.tex)
- 📚 **File BibTeX trích dẫn**: [`paper/references.bib`](file:///g:/nckh/DINO/direction_data_article/paper/references.bib) *(Đã sửa chính xác Zhang et al. CVPR 2017 và khai báo bài báo liên quan)*
- 📑 **Bảng Specifications (Data in Brief)**: [`paper/tables/tab_specifications.tex`](file:///g:/nckh/DINO/direction_data_article/paper/tables/tab_specifications.tex)
- 📊 **Bảng Thống kê thực nghiệm**: [`paper/tables/tab_summary_stats.tex`](file:///g:/nckh/DINO/direction_data_article/paper/tables/tab_summary_stats.tex)
- 📈 **Bảng So sánh Benchmark toàn diện**: [`paper/tables/tab_dataset_comparison.tex`](file:///g:/nckh/DINO/direction_data_article/paper/tables/tab_dataset_comparison.tex)
- 🖼️ **Thư mục hình vẽ chất lượng xuất bản (300 DPI)**: [`paper/figures/`](file:///g:/nckh/DINO/direction_data_article/paper/figures/)
  - `fig1_camera_spatial_map.png`: Bản đồ phân bố không gian 608 camera tại TP.HCM.
  - `fig2_temporal_and_photometric.png`: Chu kỳ biến thiên ánh sáng ngày/đêm và phân bố khoảng cách lấy mẫu $\Delta T$.
  - `fig3_graph_topology.png`: Phân bố khoảng cách đường đi OSM, bậc ra nút và độ lệch khoảng cách hai chiều.
  - `fig4_sample_snapshots.png`: 4 ảnh mẫu trực quan ngày, trưa, chiều, tối chứng minh PII không thể giải mã.
- 📕 **Tệp PDF bài báo hoàn chỉnh (11 trang camera-ready)**: [`paper/main.pdf`](file:///g:/nckh/DINO/direction_data_article/paper/main.pdf)

---

## 📦 3. CẤU TRÚC GÓI ĐÓNG GÓI ZENODO ([`zenodo_bundle/`](file:///g:/nckh/DINO/direction_data_article/zenodo_bundle))

Thư mục này được tổ chức theo tiêu chuẩn dữ liệu mở quốc tế FAIR Data Principles:

```text
DINO/direction_data_article/zenodo_bundle/
├── LICENSE                                    # Giấy phép mở CC BY 4.0 / Research Use
├── CITATION.cff                               # Định dạng trích dẫn chuẩn Citation File Format
├── .zenodo.json                               # Siêu dữ liệu cấu hình Zenodo tự động
├── data_dictionary.md                         # Từ điển dữ liệu chuẩn ISO/W3C
├── README.md                                  # Hướng dẫn quốc tế đầy đủ
├── checksums.sha256                           # Mã băm SHA-256 xác thực tính toàn vẹn
│
├── metadata/                                  # Dữ liệu hình học & topo
│   ├── routes.csv                             # 608 trạm: Tọa độ, loại đường, độ cao cột (6-15m)
│   └── road_network_distance.xlsx             # Ma trận khoảng cách định tuyến 608x608 từ OSM
│
├── graph/                                     # Ma trận đồ thị định dạng chuẩn máy học
│   ├── distance_m.npy                         # Ma trận khoảng cách tính theo mét (608x608)
│   ├── direction.npy                          # Ma trận hướng liên kết (608x608)
│   └── edges.csv                              # Danh sách 2.420 cạnh định tuyến thực tế
│
├── sample_preview/sample_camera_sequences/    # Tập ảnh mẫu mini để chạy thử nghiệm tức thì
│
└── code/                                      # Module Python nạp dữ liệu chuẩn production
    ├── __init__.py                            # Package init
    ├── load_unlabeled_images.py               # Lớp TrafficCameraUnlabeledDataset
    ├── camera_sampler.py                      # CameraGroupedSampler & TemporalSequenceSampler
    └── graph_utils.py                         # Dual-transition matrices P_f, P_b & Symmetrized Laplacian
```

---

## 🔬 4. KẾT QUẢ THỰC NGHIỆM ĐO ĐẠC KIỂM ĐỊNH ([`validation/`](file:///g:/nckh/DINO/direction_data_article/validation))

Toàn bộ các bài kiểm tra chất lượng kỹ thuật trong [`validation/`](file:///g:/nckh/DINO/direction_data_article/validation) đã được thực thi và khớp 100% với số liệu trong bài báo:

| Vòng Kiểm Định | Tệp Mã Nguồn | Chỉ Số Thực Nghiệm Đo Được |
| :--- | :--- | :--- |
| **V1: Tính Toàn Vẹn Ảnh** | [`validate_image_integrity.py`](file:///g:/nckh/DINO/direction_data_article/validation/validate_image_integrity.py) | **100% hợp lệ** (loại bỏ hoàn toàn các ảnh lỗi truyền dẫn municipal placeholder 284x177); Độ phân giải: **$512 \times 288$ px**; Dung lượng trung bình: $65.87 \pm 9.19$ KB; Kênh màu: 3-channel RGB. |
| **V2: Độ Bao Phủ & Trùng Lặp** | [`validate_temporal_coverage.py`](file:///g:/nckh/DINO/direction_data_article/validation/validate_temporal_coverage.py)<br>[`validate_duplicate_hash.py`](file:///g:/nckh/DINO/direction_data_article/validation/validate_duplicate_hash.py) | Bao phủ trọn vẹn 168 giờ liên tục qua 7 ngày. **Tỷ lệ ảnh lặp SHA-256 là 0.00%** (khung hình biến thiên liên tục, không bị treo luồng). Độ lệch thời gian crawl trung bình: $\Delta t_{\text{lag}} = 15.0 \pm 4.2\text{ s}$. |
| **V3: Đa Dạng Quang Học & Độ Nét** | [`validate_visual_diversity.py`](file:///g:/nckh/DINO/direction_data_article/validation/validate_visual_diversity.py)<br>[`validate_blur_laplacian.py`](file:///g:/nckh/DINO/direction_data_article/validation/validate_blur_laplacian.py) | **Mean Luminance (Độ sáng)**: $97.35 \pm 15.03$ (Dải ngày/đêm $[67.37, 122.37]$); **RMS Contrast**: $45.79 \pm 6.57$; **Shannon Entropy**: $7.30 \pm 0.24$ bits; **Laplacian Variance**: $2836.51 \pm 1252.35$ (100% đạt chuẩn sắc nét). |
| **V4: Cấu Trúc Topo Đồ Thị Có Hướng** | [`validate_graph_topology.py`](file:///g:/nckh/DINO/direction_data_article/validation/validate_graph_topology.py) | **2.420 liên kết định tuyến có hướng** ($d_{ij} \le 5.0$ km, độ thưa 99.34%). Tổng 1.737 cặp nút liên thông:<br>• **1.054 cặp một chiều thuần túy (60.68%)**<br>• **683 cặp hai chiều (39.32%)**, trong đó **232 cặp (33.97%) bị lệch khoảng cách $>50\text{ m}$** do dải phân cách cứng & cầu vượt.<br>• Khoảng cách TB: $1.11 \pm 0.98$ km (Trung vị: 0.81 km); Bậc ra TB: 3.98 (max: 17). |
| **V5: Kiểm Định Quyền Riêng Tư PII** | [`verify_pii_anonymity.py`](file:///g:/nckh/DINO/direction_data_article/validation/verify_pii_anonymity.py) | Độ phân giải mặt đất GSD: $2.73 - 3.25\text{ cm/pixel}$. Biển số xe máy ($19 \times 14\text{ cm}$) chỉ chiếm $\approx 7 \times 5$ px (dưới ngưỡng Nyquist và ALPR $\ge 16$ px). Người đi xe bắt buộc đội mũ bảo hiểm. **Tỷ lệ nhận diện PII khuôn mặt / biển số: 0.00%**. |

---

## ⚡ 5. CÁCH CHẠY THỰC NGHIỆM & SINH BÁO CÁO

Bạn có thể chạy các script trên PowerShell để kiểm tra toàn bộ pipeline:

```powershell
# 1. Chạy sinh lại toàn bộ 4 hình ảnh bài báo (300 DPI):
python DINO/direction_data_article/validation/generate_paper_figures.py

# 2. Chạy kiểm định quyền riêng tư và PII:
python DINO/direction_data_article/validation/verify_pii_anonymity.py

# 3. Chạy kiểm định cấu trúc topo đồ thị có hướng:
python DINO/direction_data_article/validation/validate_graph_topology.py

# 4. Biên dịch lại bài báo LaTeX:
cd DINO/direction_data_article/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
