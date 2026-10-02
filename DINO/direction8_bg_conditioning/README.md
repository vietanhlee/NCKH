# Hướng 8: Thích Ứng Camera Mới Bằng Thống Kê Toàn Cục Của Background

> **Paper Title Candidate:** *Background-Conditioned Generalization to Unseen Traffic Cameras with Unreliable Scene Priors*  
> **Target:** Pattern Recognition / Engineering Applications of Artificial Intelligence (EAAI) / IEEE T-ITS  

---

## 1. Giới thiệu & Động lực Khoa học
Trong mạng lưới camera giám sát quy mô thành phố lớn với hàng ngàn camera:
- Các mô hình thị giác máy tính (đếm xe, phát hiện ùn tắc) khi huấn luyện trên một nhóm camera quen thuộc thường bị **suy giảm độ chính xác nghiêm trọng khi chuyển sang camera mới**:
  - Góc phối cảnh và độ cao lắp đặt khác biệt.
  - Tỉ lệ kích thước xe máy/ô tô theo chiều sâu không gian khác nhau.
  - Bố cục làn đường, vật cản, ánh sáng đèn đường ban đêm hoàn toàn khác.
- Việc gán nhãn lại cho từng camera mới là bất khả thi về mặt chi phí.
- Tuy nhiên, mỗi camera đều có sẵn một tài nguyên miễn phí: **Ảnh nền tĩnh (Background Image)** được ước lượng theo từng khung giờ.
- **Thách thức cốt lõi:**
  Ảnh nền thực tế thường chứa **nhiễu cục bộ** (ghost vehicle khi tắc đường, bóng đổ, lệch vài pixel). Nếu dùng background ở mức pixel thô (như ghép kênh sai khác $\Delta$ ở Hướng 3), mô hình rất dễ bị "đánh lừa" khi background bị hỏng.
  $\implies$ **Giải pháp:** Sử dụng **Thống kê toàn cục cắt tỉa (Trimmed Statistics)** để trích xuất Bản mô tả cảnh $z$ bất biến với nhiễu cục bộ.

---

## 2. Kiến trúc Phương pháp

### 2.1. Robust Scene Descriptor Extraction (`descriptor.py`)
Vector mô tả cảnh toàn cục $z$ được tính từ token patch của DINOv3 (đóng băng):
$$z = \Big[\, \operatorname{TrimMean}_{p \in R} f_{\text{bg}}(p),\; \operatorname{TrimStd}_{p \in R} f_{\text{bg}}(p),\; \operatorname{TrimMean}_{p} f_{\text{bg}}(p) \,\Big]$$
- **Trimmed Mean:** Loại bỏ 10% patch xa trung vị nhất. Ghost vehicle (xe kẹt in vào nền) chỉ chiếm diện tích nhỏ $\implies$ bị loại bỏ sạch sẽ khỏi $z$.
- $z$ mã hóa thành công các thuộc tính toàn cục của camera: độ cao góc máy, độ rộng mặt đường, ánh sáng môi trường.

### 2.2. Conditioning Mechanisms Suite (`conditioning.py`)
Bài báo so sánh đối đầu 3 cơ chế điều kiện hóa:
1. **FiLM (Feature-wise Linear Modulation) - Cơ chế tối ưu:**
   $$\gamma, \beta = \operatorname{MLP}(z), \quad h_{\text{mod}} = (1 + \gamma) \odot \operatorname{LayerNorm}(h) + \beta$$
   Khởi tạo **Zero-Init** ($\gamma=0, \beta=0$) giúp mạng ban đầu hoạt động ổn định như mô hình gốc.
2. **Scene Prompt Tokens:** Nối $K$ visual prompt tokens chiếu từ $z$ vào chuỗi token của ViT.
3. **Cross-Attention:** Tương tác Token-to-Token giữa ảnh hiện tại và ảnh nền (nhạy cảm với nhiễu nhất).

### 2.3. Robust Training Suite (`models.py`)
- **Background-Dropout ($p=0.25$):** Huấn luyện ngẫu nhiên thay $z$ bằng learnable empty vector $z_{\emptyset}$. Đảm bảo mô hình vẫn suy luận tốt ngay cả khi camera mới chưa kịp tạo ảnh nền.
- **Background-Swap:** Tráo đổi ảnh nền với các khung giờ lân cận ($\pm 1-2$h) để chống overfitting camera ID.
- **Đầu ra đa nhiệm:** Hỗ trợ đồng thời 2 tác vụ: Đếm mật độ xe (Regression) và Phân loại 4 mức ùn tắc (Classification).

---

## 3. Cấu trúc Thư mục
```
direction8_bg_conditioning/
├── descriptor.py     # Trích xuất Trimmed Mean/Std z từ DINOv3
├── conditioning.py   # FiLM (zero-init), Prompt tokens, Cross-Attention
├── models.py         # BackgroundConditionedModel + Bg-Dropout
├── evaluate.py       # Đánh giá Unseen Camera Generalization & BDB Stress Test
└── README.md
```

---

## 4. Hướng dẫn Chạy Thử nghiệm

```bash
# Chạy đánh giá thích ứng camera chưa thấy & kiểm thử BDB
python direction8_bg_conditioning/evaluate.py
```
