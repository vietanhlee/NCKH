# Hướng G — Học biểu diễn hướng phương tiện từ chuỗi ảnh thưa của camera cố định

Oct 3, 2026 · @Hung Do

Hướng G tự học một backbone ViT tập trung vào phương tiện chỉ từ chuỗi ảnh nhiều ngày của camera cố định: không cần ảnh nền, không cần video liên tục, không cần nhãn. G thay thế H1 (BG-Guided DINO) trong [tài liệu đề xuất chung](https://claude.ai/code/artifact/0ff3f2c5-4116-47ea-9d41-4fdff8edc4f6); FAM-Δ của H1 chỉ còn là một baseline.

## 1. Bối cảnh, vấn đề và ý tưởng cốt lõi

Camera giao thông cố định cho ra chuỗi ảnh nhìn cùng một cảnh suốt nhiều tháng. Tại một vị trí (i, j) trên ảnh, phần lớn thời gian là mặt đường, cột điện hay tòa nhà; thỉnh thoảng mới có xe đi qua. Cấu trúc này là "giám sát miễn phí" về chỗ nào là phương tiện, nhưng hiện chưa được khai thác đúng cách:

- Các phương pháp tự học hiện đại (DINOv2/v3, iBOT, MAE) coi mỗi ảnh độc lập và che ngẫu nhiên, nên phần lớn công sức học rơi vào mặt đường.
- Các phương pháp che theo chuyển động (MGMAE, MGM, V-JEPA4A) cần video liên tục để tính optical flow hoặc hiệu frame liền kề. Dữ liệu ở đây là ảnh chụp cách nhau vài phút — không dùng được.
- H1 dùng Δ với ảnh nền median, nên chịu toàn bộ lỗi của median: ghost ở giờ kẹt, thay đổi ánh sáng, camera lệch.
- Khi tự học trên dữ liệu có nền dùng chung, mạng dễ "đi đường tắt": học nhận ra nền thay vì đối tượng.

**Vấn đề.** Chỉ từ chuỗi ảnh thưa của camera cố định — không nhãn, không ảnh nền — làm sao để mô hình (1) biết vùng nào nhiều khả năng là phương tiện, (2) dồn năng lực học vào đó một cách hợp lý, và (3) không dựa vào nền để giải bài toán tự học?

**Ý tưởng — 3 thành phần:**

1. **TAM (Temporal Atypicality Map — bản đồ độ khác thường theo vị trí).** Với mỗi camera và mỗi vị trí patch, duy trì vài "trạng thái tĩnh" phổ biến trong không gian feature DINO (mặt đường khô, mặt đường ướt, có bóng cây). Patch hiện tại càng xa mọi trạng thái tĩnh thì càng có khả năng là xe. Mặt đường lặp lại nên tạo cụm chặt và thường xuyên; xe đa dạng về màu, loại và vị trí nên không tạo được cụm như vậy.
2. **AGM (Atypicality-Guided Masking — che phân tầng theo TAM).** Kiểm soát rõ hai đại lượng: tỷ lệ ngân sách che dành cho vùng xe, và tỷ lệ tối đa vùng xe bị che (để phần xe còn lại làm ngữ cảnh). Câu hỏi "che nhiều hay ít vùng xe" trở thành một thí nghiệm, không còn là giả định.
3. **SRS (Static-Region Swap — hoán đổi vùng tĩnh giữa các ngày).** Thay các patch tĩnh của frame ngày d bằng patch cùng vị trí của frame ngày d' cùng camera. Hai ảnh có cùng phương tiện nhưng khác "nền" (ánh sáng, độ ướt, bóng) → buộc biểu diễn bất biến với nền và tập trung vào xe.

**Điểm mạnh so với H1:** không cần ảnh nền; không cần frame liền kề (lấy mẫu thưa đến đâu cũng dùng được, chỉ cần nhiều ngày); TAM tự cải thiện khi feature tốt hơn; mọi thành phần đều đo kiểm được trên tập audit. Sản phẩm cuối chỉ là một backbone ViT, dùng thay DINOv3 gốc cho các hướng khác (đếm xe, mức ùn tắc, phân rã cảnh).

## 2. Câu hỏi nghiên cứu và đóng góp

**Câu hỏi nghiên cứu:**

1. **RQ1:** TAM có định vị phương tiện tốt hơn Δ pixel với ảnh nền median, và tốt hơn phát hiện đối tượng không giám sát trên từng ảnh (MaskCut), đặc biệt ở cao điểm và ban đêm không?
2. **RQ2:** Che theo TAM có cho biểu diễn tốt hơn che ngẫu nhiên, che theo attention và che theo Δ với **cùng compute** không? Và nên che nhiều hơn hay ít hơn vùng xe?
3. **RQ3:** SRS có làm giảm mức dựa vào nền và tăng tổng quát hóa sang camera, vùng, điều kiện chưa thấy không?
4. **RQ4:** Biểu diễn có thật sự "hiểu phương tiện" hơn không — đo bằng probe dày (phân đoạn xe), probe nhận dạng camera và các phép thử can thiệp (đổi nền, xóa xe)?

**Đóng góp dự kiến:**

1. TAM: thống kê feature đa trạng thái theo vị trí cho camera cố định, không cần ảnh nền và không cần chuyển động.
2. AGM: chiến lược che phân tầng với hai tham số diễn giải được, kèm bằng chứng thực nghiệm về hướng che tối ưu.
3. SRS: phép tăng cường dựa trên cấu trúc cùng camera để chống đường tắt nền.
4. Bộ chẩn đoán đường tắt nền cho tự học trên camera cố định: probe camera ID, độ nhất quán khi đổi nền, độ nhạy khi xóa xe.

## 3. Công trình liên quan và khoảng trống

Tra cứu tháng 10/2026. Mục có link đã xác nhận qua trang chính thức hoặc trang bài báo.

| Nhóm | Công trình | Làm gì | Khác ở đâu |
| --- | --- | --- | --- |
| Che theo chuyển động | [MGMAE (Huang et al., ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_MGMAE_Motion_Guided_Masking_for_Video_Masked_Autoencoding_ICCV_2023_paper.html) | Dùng optical flow trực tuyến để tạo khối che nhất quán theo thời gian cho video MAE | Cần video liên tục |
| Che theo chuyển động | [MGM (Fan et al., ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Fan_Motion-Guided_Masking_for_Spatiotemporal_Representation_Learning_ICCV_2023_paper.html) | Lấy motion vector từ video nén để dẫn vị trí che | Cần video nén liên tục |
| Che theo vùng quan trọng, giao thông | [V-JEPA4A — Mask What Matters (Lang, Braun, Valada, arXiv 08/2026)](https://www.alphaxiv.org/abs/2608.17178) | Che theo độ quan trọng ngữ nghĩa và thời gian cho video lái xe | Camera gắn trên xe, video liên tục; xem ghi chú dưới bảng |
| Che học được | [ADIOS (Shi et al., ICML 2022)](https://proceedings.mlr.press/v162/shi22d.html) | Học hàm che bằng mục tiêu đối kháng; che theo mask đối tượng thật tốt hơn hẳn các cách che khác | Không dùng cấu trúc camera cố định — baseline tham khảo |
| Che theo attention | AttMask (Kakogeorgiou et al., ECCV 2022) — chưa kiểm chứng link | Che theo attention của teacher | Chỉ dùng thông tin trong một ảnh |
| Đường tắt nền | [Background Erasing (Wang et al., CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Removing_the_Background_by_Adding_the_Background_Towards_Background_Robust_CVPR_2021_paper.html) | Trộn một frame tĩnh vào mọi frame của video, buộc feature không đổi → chống dựa vào nền | Trộn toàn ảnh ở mức pixel; SRS hoán đổi chọn lọc theo vị trí tĩnh |
| Đường tắt nền | [Jenni & Favaro (arXiv 2010.06218)](https://arxiv.org/pdf/2010.06218) | Ghi nhận nền dùng chung của camera cố định tạo đường tắt; xử lý bằng trừ nền median | Vẫn cần ảnh nền median |
| Ngữ cảnh camera cố định | [Focus on the Positives (Pantazis et al., ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Pantazis_Focus_on_the_Positives_Self-Supervised_Learning_for_Biodiversity_Monitoring_ICCV_2021_paper.html) | Camera bẫy ảnh: dùng ngữ cảnh không gian–thời gian để chọn cặp **dương** | Ta tạo cặp dương có nền khác (SRS) và, trong ablation, cặp **âm** |
| Cùng vị trí, khác thời điểm | [MoCo với temporal positives cho ảnh vệ tinh (arXiv 2210.11815)](https://arxiv.org/pdf/2210.11815) | Ảnh cùng vị trí khác thời điểm làm cặp dương → học đặc trưng cảnh bền vững | Học cái **tĩnh**; ta muốn học cái **động** |
| Tự học dày theo thời gian | [TimeT (Salehi et al., ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Salehi_Time_Does_Tell_Self-Supervised_Time-Tuning_of_Dense_Image_Representations_ICCV_2023_paper.html) | Fine-tune ViT bằng loss phân cụm căn chỉnh theo thời gian trên video | Cần căn chỉnh qua frame liền kề; camera cố định cho căn chỉnh vị trí miễn phí |
| Phát hiện đối tượng không giám sát | [CutLER/MaskCut (Wang et al., CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Cut_and_Learn_for_Unsupervised_Object_Detection_and_Instance_Segmentation_CVPR_2023_paper.pdf); TokenCut; LOST | Cắt đồ thị trên feature DINO để tìm đối tượng trong từng ảnh | Chỉ dùng một ảnh; khó với hàng trăm xe máy nhỏ chồng nhau — baseline cho RQ1 |

**Ghi chú về V-JEPA4A:** theo phần tóm tắt trên trang alphaXiv, nhóm tác giả thấy xác suất che vùng tiền cảnh nên **thấp hơn** vùng nền, vì che xe quá nhiều làm student mất ngữ cảnh. Điều này ngược với giả định "che nhiều vùng xe" của H1 — cần đọc toàn văn để xác nhận. Vì vậy AGM không cố định hướng che mà tham số hóa nó, và thí nghiệm G-AB4 trả lời trực tiếp câu hỏi này.

**Khoảng trống** (trong phạm vi đợt tra cứu này): chưa thấy công trình tự học biểu diễn từ **ảnh chụp thưa** của **camera cố định**, dùng **thống kê feature dài hạn theo vị trí** thay cho ảnh nền và cho chuyển động, để vừa dẫn hướng che vừa chống đường tắt nền. Trước khi nộp phải tra lại, đặc biệt với các từ khóa "static camera self-supervised", "surveillance pretraining", "background bias masked image modeling".

## 4. Tổng quan phương pháp

&#91;embedded content: Luồng huấn luyện của G · TAM dẫn hướng cả việc che (AGM) lẫn việc hoán đổi nền (SRS)\]

Mỗi bước huấn luyện: (1) DINOv3 đóng băng chạy trên khung hình đầy đủ để cập nhật thống kê theo vị trí và tính TAM; (2) SRS ghép patch tĩnh từ một frame ngày khác vào ảnh của student; (3) AGM chọn patch để che theo TAM; (4) teacher nhận ảnh gốc, student nhận ảnh đã biến đổi, loss DINO + iBOT như DINOv2.

## 5. TAM — bản đồ độ khác thường theo vị trí

**Feature đầu vào.** Frame đã căn chỉnh (mục 9), resize về 448 × 256 → DINOv3 ViT-B/16 **đóng băng** → 16 × 28 = 448 token patch, 768 chiều → chuẩn hóa L2 → PCA về d = 64 (fit một lần trên 200k token ngẫu nhiên của camera train) → chuẩn hóa L2 lần nữa, ký hiệu u\_t(p). Dùng mô hình đóng băng để TAM ổn định, không phụ thuộc quá trình huấn luyện; biến thể dùng teacher EMA là một ablation.

**Thống kê theo vị trí.** Với mỗi camera c, chế độ m (ngày / IR) và vị trí p, lưu K = 4 "trạng thái": tâm μ\_k (vector đơn vị d chiều), tần suất w\_k (tổng bằng 1) và độ phân tán s\_k (khoảng cách cosine trung bình của các mẫu thuộc trạng thái đó). Bộ nhớ: 608 camera × 2 chế độ × 448 vị trí × 4 trạng thái × 64 chiều × fp16 ≈ 280 MB, đặt thẳng trên GPU.

**Vì sao phân biệt được mặt đường và xe.** Ví dụ vị trí (5, 12) trên lòng đường: 70% thời gian là nhựa xám khô, 15% là nhựa ướt, 15% là các loại xe khác nhau. Hai trạng thái mặt đường tạo cụm **chặt** và **thường xuyên** (w lớn, s nhỏ). Xe đỏ, xe đen, xe buýt, người đi bộ phân tán khắp không gian feature, không gom thành một trạng thái có w lớn. Vì vậy chỉ coi một trạng thái là "tĩnh" khi nó vừa thường xuyên vừa chặt.

**Cập nhật trực tuyến** (mỗi frame trong batch, không qua gradient):

```latex
k^* = \arg\max_k \cos\big(u_t(p), \mu_k\big), \quad
\mu_{k^*} \leftarrow \frac{(1-\eta)\,\mu_{k^*} + \eta\, u_t(p)}{\lVert \cdot \rVert}, \quad
w_k \leftarrow (1-\eta_w)\, w_k + \eta_w\, \mathbb{1}[k = k^*], \quad
s_{k^*} \leftarrow (1-\eta)\, s_{k^*} + \eta \big(1 - \cos(u_t(p), \mu_{k^*})\big)
```

Với η = 0.02, η\_w = 0.005 (tần suất phản ánh khoảng 200 lần quan sát gần nhất). Nếu cos(u, μ\_{k\*}) < 0.6 và có trạng thái với w < 0.02 thì khởi tạo lại trạng thái đó bằng u (w = 0.05) — cho phép xuất hiện trạng thái mới, ví dụ mặt đường ướt sau cơn mưa đầu mùa.

**Khởi tạo.** Chạy một lượt offline: mỗi (camera, chế độ) lấy 300 frame rải trên ít nhất 10 ngày, chạy k-means cosine K = 4 cho từng vị trí → μ; w = tỷ lệ cụm; s.

**Tập trạng thái tĩnh và độ khác thường:**

```latex
\mathcal{S}(c,m,p) = \{\, k : w_k \ge w_{\min},\; s_k \le s_{\max} \,\}, \qquad
a_t(p) = \min_{k \in \mathcal{S}} \frac{1 - \cos\big(u_t(p), \mu_k\big)}{s_k + \epsilon}
```

Với w\_min = 0.15, s\_max = 1.5 × median của mọi s\_k trong camera, ε = 0.01. Chia cho s\_k giống chuẩn hóa kiểu Mahalanobis: trạng thái "lỏng" (bóng cây lay động) được phép lệch nhiều hơn. Nếu 𝒮 rỗng — vị trí gần như lúc nào cũng có xe hoặc không ổn định — gán a\_t(p) là không xác định; AGM xử lý vị trí đó như che ngẫu nhiên. **Báo cáo tỷ lệ vị trí rỗng theo camera**: đây là giới hạn thật của phương pháp và phải nêu trong paper.

**Lưu ý thiết kế:** không đo bằng khoảng cách tới láng giềng gần nhất như PatchCore. Xe đi qua mọi vị trí nên kho feature chứa sẵn feature xe → xe sẽ trông "bình thường". Phải đo khoảng cách tới **trạng thái tĩnh**, như trên.

**Từ điểm sang tập tiền cảnh.** Với mỗi (camera, chế độ), fit Gaussian Mixture 2 thành phần trên log a\_t(p) của 50.000 giá trị gần nhất (cập nhật mỗi 1.000 bước) → xác suất tiền cảnh π\_t(p) = hậu nghiệm của thành phần có trung bình lớn hơn. Tập tiền cảnh F\_t = {p : π\_t(p) > 0.5}. Không dùng nhãn; ngưỡng chỉ được **kiểm tra** (không chỉnh) trên audit-val.

**Sản phẩm phụ — bản đồ hoạt động của camera:** A\_c(p) = 1 − Σ\_{k∈𝒮} w\_k, tức tỷ lệ thời gian vị trí p không ở trạng thái tĩnh. Đây là bản đồ "làn hay có xe", không cần road mask — dùng được cho mô tả camera và để kiểm tra chéo road mask người vẽ. Hướng 2 mới (phân rã cảnh không cần ảnh nền) dùng trực tiếp π\_t và A\_c.

**Giới hạn cần nêu:**

- Vị trí bị chiếm hơn 85% thời gian bởi vật trông giống nhau (hàng xe máy đậu cố định ở mép đường) sẽ bị coi là tĩnh — về ngữ nghĩa có thể chấp nhận.
- Thay đổi ánh sáng nhanh trong ngày tạo nhiều trạng thái; K = 4 có thể thiếu — ablation K.
- Camera rung làm vị trí patch lệch → mọi thứ thành khác thường. Dựa vào căn chỉnh (mục 9) và phát hiện bằng tỷ lệ patch khác thường trên vùng tĩnh (nếu > 50% thì bỏ frame).

## 6. AGM — che phân tầng theo TAM

**Vấn đề với FAM của H1:** chỉ có một núm α trộn giữa "che theo Δ" và "che đều", nên không trả lời được câu hỏi cốt lõi: nên dành bao nhiêu ngân sách che cho vùng xe, và có nên để lại một phần xe làm ngữ cảnh không. V-JEPA4A gợi ý rằng che tiền cảnh quá nhiều có hại. AGM tách thành hai tham số diễn giải được:

| Tham số | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| φ (phi) | Tỷ lệ số patch bị che thuộc vùng xe F | Ảnh có xe chiếm 20% diện tích. Che ngẫu nhiên → φ ≈ 0.2. Đặt φ = 0.5 → một nửa số patch bị che là xe |
| q\_max | Tỷ lệ tối đa patch xe được phép che | q\_max = 0.6 → luôn còn ít nhất 40% patch xe hiển thị làm ngữ cảnh |

**Thuật toán cho một global crop:**

```python
def agm_sample(pi, valid, ratio, phi, q_max, gen):
    """pi: (N,) xác suất tiền cảnh đã ánh xạ theo crop; valid: (N,) bool (TAM xác định).
    ratio ~ U[0.1, 0.5] (như iBOT/DINOv2). Trả về mask bool (N,)."""
    N = pi.numel()
    n_mask = round(ratio * N)
    F = (pi > 0.5) & valid                  # vùng xe
    B = ~F                                  # phần còn lại, gồm cả vị trí không xác định
    k_fg = min(round(phi * n_mask), int(q_max * F.sum()))
    k_bg = min(n_mask - k_fg, int(B.sum()))
    fg_idx = sample_without_replacement(F, k_fg, weights=pi, gen=gen)   # ưu tiên patch chắc là xe
    bg_idx = sample_without_replacement(B, k_bg, weights=None, gen=gen) # đều
    mask = zeros(N, bool); mask[fg_idx] = True; mask[bg_idx] = True
    return mask
```

Lấy mẫu không hoàn lại có trọng số bằng Gumbel top-k (key = log w + Gumbel(0, 1), lấy k lớn nhất). Nếu ảnh không có xe (F rỗng), AGM trở về che ngẫu nhiên — đúng hành vi mong muốn cho ảnh đêm vắng.

**Ánh xạ theo crop.** π được tính trên khung hình đầy đủ (lưới 16 × 28). Với mỗi global crop, áp **cùng tham số** RandomResizedCrop và lật lên bản đồ π (nội suy bilinear), rồi average-pool về lưới 14 × 14. Lỗi dễ mắc: dùng π của ảnh gốc cho crop mà không biến đổi theo.

**Bốn chính sách so sánh trong G-AB4** (cùng tỷ lệ che tổng):

| Chính sách | φ | q\_max | Giả thuyết |
| --- | --- | --- | --- |
| Ngẫu nhiên | tỷ lệ diện tích xe tự nhiên | 1.0 | Mốc so sánh |
| Che nhiều xe (kiểu H1) | 0.7 | 1.0 | Dồn công học vào xe |
| Phân tầng cân bằng (mặc định) | 0.5 | 0.6 | Học xe nhiều hơn nhưng giữ ngữ cảnh xe |
| Giữ xe (kiểu V-JEPA4A) | 0.5 × tỷ lệ tự nhiên | 1.0 | Xe làm ngữ cảnh, che chủ yếu nền |

Thêm lưới quét φ ∈ {0.2, 0.35, 0.5, 0.7} × q\_max ∈ {0.4, 0.6, 0.8, 1.0} trên một seed để vẽ **bề mặt hiệu năng** — một hình đáng đưa vào paper vì nó trả lời câu hỏi "che bao nhiêu xe" một cách có hệ thống.

**Lịch φ (tùy chọn).** Tăng φ tuyến tính từ tỷ lệ tự nhiên lên giá trị đích trong 10% đầu quá trình huấn luyện.

**Loss iBOT có trọng số (ablation).** Nhân loss của patch bị che với (1 + β·π), β ∈ {0, 1, 2} — kiểm tra xem tăng trọng số loss có tương đương với tăng tỷ lệ che vùng xe không.

## 7. SRS — hoán đổi vùng tĩnh, và phần mở rộng

**Ý tưởng.** Lấy frame x\_t (camera c, ngày d, 17:20) và frame x' cùng camera nhưng khác ngày d' (17:05). Ở những vị trí mà **cả hai** frame đều tĩnh, thay patch của x\_t bằng patch của x'. Kết quả x̃\_t giữ đúng các xe của x\_t, còn mặt đường, bóng đổ, độ ướt, ánh sáng lấy từ ngày khác. Teacher nhìn x\_t, student nhìn x̃\_t, loss DINO buộc hai biểu diễn giống nhau → mô hình phải **bỏ qua** khác biệt ở vùng tĩnh và dựa vào xe.

So với Background Erasing (CVPR 2021) — trộn mờ một frame tĩnh lên toàn video — SRS không tạo "xe ma" bán trong suốt, vì chỉ hoán đổi đúng những vị trí tĩnh ở cả hai frame. Phép này chỉ làm được nhờ camera cố định: cùng vị trí trên ảnh là cùng một điểm trong cảnh.

**Thuật toán:**

1. **Chọn cặp:** x' cùng camera, cùng chế độ (ngày/IR), khác ngày, slot trong khoảng ±2 giờ (ablation: cùng slot / bất kỳ slot trong chế độ). Cả hai frame đã căn chỉnh (mục 9) và `align_ok`.
2. **Tập hoán đổi:** P\_swap = {p : π\_t(p) < 0.2 và π'(p) < 0.2 và cả hai xác định}, rồi **loại thêm các patch kề** vùng tiền cảnh của cả hai frame (giãn F\_t ∪ F' một patch) để không chép nửa chiếc xe.
3. **Chọn tỷ lệ:** lấy ngẫu nhiên một tập con của P\_swap, tỷ lệ ρ \~ U\[0.3, 1.0\].
4. **Ghép ở mức pixel** trên khung 448 × 256, theo khối 16 × 16; làm mềm biên mỗi khối bằng trộn tuyến tính 4 px để không tạo đường nối sắc — tránh mô hình học dấu vết đường nối.
5. Áp SRS với xác suất p\_srs = 0.5 cho mỗi mẫu, **trước** khi cắt crop và trước AGM. Teacher luôn nhận frame gốc.

**Nghiệm thu nhanh:** trên mọi patch thuộc F\_t, x̃\_t phải trùng từng pixel với x\_t (trừ dải làm mềm 4 px ở biên các khối hoán đổi kề bên).

**Phần mở rộng (chỉ giữ nếu thắng trong ablation):**

| Mã | Cơ chế | Cách làm | Rủi ro |
| --- | --- | --- | --- |
| RIC | Tương phản vùng xe trong cùng camera | Batch 8 camera × 8 frame (khác ngày, slot ±2 giờ). Gộp token patch của student theo trọng số π → vector vùng xe z\_i; cặp dương = z của hai view cùng frame; cặp âm = z của frame khác **cùng camera**. InfoNCE, nhiệt độ 0.2, λ = 0.1 | Âm tính giả khi hai frame có cảnh xe tương tự. Giảm bằng trọng số âm mềm w\_ij = 1 − cos(ẑ\_i, ẑ\_j) tính từ teacher |
| GRL | Đối kháng nhận dạng camera | Bộ phân loại camera ID trên \[CLS\] qua lớp đảo gradient (kiểu DANN), λ = 0.05 | Có thể xóa luôn thông tin phối cảnh hữu ích cho đếm xe |

**Baseline đối nghịch cần có:** chọn cặp **dương** là frame cùng camera, gần thời điểm (kiểu Focus on the Positives). Giả thuyết: cách này tăng phụ thuộc vào nền và **giảm** chất lượng cho bài toán xe — nếu đúng, đây là một kết quả có giá trị riêng.

## 8. Loss, siêu tham số, lịch huấn luyện

```latex
\mathcal{L} = \mathcal{L}_{\text{DINO}}^{[CLS]} + \lambda_{\text{iBOT}} \sum_{p \in \mathcal{M}_{\text{AGM}}} \big(1 + \beta\, \pi(p)\big)\, \ell_{\text{iBOT}}(p) + \lambda_{\text{KoLeo}} \mathcal{L}_{\text{KoLeo}} \;\big[+\, \lambda_{\text{RIC}} \mathcal{L}_{\text{RIC}}\big]
```

𝓜\_AGM là tập patch bị che do AGM chọn; student nhận ảnh đã qua SRS. Mặc định β = 0, λ\_iBOT = 1.0, λ\_KoLeo = 0.1, RIC tắt.

**Siêu tham số khởi điểm** (theo cấu hình kiểu DINOv2, điều chỉnh cho huấn luyện tiếp trên dữ liệu miền — kiểm tra lại với repo DINOv3 chính thức):

| Nhóm | Tham số | Giá trị |
| --- | --- | --- |
| Mô hình | Backbone | ViT-B/16 khởi tạo DINOv3 (ablation và thử nhanh: ViT-S/16) |
|  | Đầu DINO, đầu iBOT | Riêng biệt, mỗi đầu 16.384 prototype, khởi tạo mới |
| Crop | Global | 2 crop 224, tỷ lệ diện tích \[0.32, 1.0\] |
|  | Local | 8 crop 96, tỷ lệ \[0.05, 0.32\] (giảm còn 4 nếu thiếu compute) |
| Che | Tỷ lệ | U\[0.1, 0.5\] mỗi ảnh; che 50% số ảnh; chỉ global view của student |
|  | AGM | φ = 0.5, q\_max = 0.6 |
| SRS | Xác suất, cặp | p\_srs = 0.5; cùng camera, khác ngày, slot ±2 giờ |
| TAM | Thống kê | K = 4, d = 64, η = 0.02, η\_w = 0.005, w\_min = 0.15 |
| Tối ưu | AdamW | lr đỉnh 5e-5 (batch 256), layer-wise decay 0.9, weight decay 0.04 → 0.2 (cosine), warmup 5% số bước, grad clip 3.0, drop path 0.1, bf16 |
| Teacher | EMA, nhiệt độ | momentum 0.994 → 1.0 (cosine); nhiệt độ teacher 0.04 → 0.07 trong 10% đầu; student 0.1 |
| Batch | Cấu trúc | 256 = 32 camera × 8 frame (lấy mẫu theo nhóm camera để có cặp SRS và RIC) |
| Độ dài | Số bước | ViT-B: 60.000 bước; ViT-S cho ablation: 20.000 bước. Mọi biến thể trong cùng bảng **đúng cùng số bước** |

**Lịch:**

1. **Trước khi train:** khởi tạo TAM offline; fit PCA; fit GMM ban đầu.
2. **0–10% số bước:** p\_srs tăng tuyến tính 0 → 0.5; φ tăng từ tỷ lệ tự nhiên lên 0.5.
3. **Toàn bộ quá trình:** thống kê TAM cập nhật trực tuyến từ feature đóng băng của mọi frame trong batch; GMM cập nhật mỗi 1.000 bước.

**Chi phí phụ.** Mỗi bước thêm một lượt forward không gradient của DINOv3 ViT-B trên khung 448 × 256 cho mọi frame (và frame cặp SRS). Ước tính tăng 25–40% thời gian mỗi bước — phải đo và báo cáo (G-E1). Cách giảm: TAM bằng ViT-S đóng băng (ablation G-AB2), hoặc cache feature PCA-64 xuống đĩa (khoảng 57 KB/frame ở fp16).

## 9. Nền móng dữ liệu và đánh giá

Phần này tóm tắt những gì G cần từ hạ tầng chung (chi tiết đầy đủ ở mục IV của tài liệu đề xuất chung). Agent cài các mục này **trước** code của G.

**Chỉ mục dữ liệu** (`data/index/frames.parquet`, một dòng mỗi frame): `frame_id`, `camera_id`, `ts_utc`, `day` và `slot` tính theo giờ **Asia/Ho\_Chi\_Minh**, `is_corrupt`, `is_ir`, `dx`, `dy`, `align_ok`, `cam_epoch`, `gap_prev_s`. Ảnh trong bộ nhớ là RGB uint8; tensor float trong \[0, 1\]; độ phân giải làm việc 448 × 256.

**Tiền xử lý dùng chung:**

| Bước | Cách làm | Ngưỡng khởi điểm |
| --- | --- | --- |
| Frame hỏng | std ảnh xám < 5; > 90% pixel bão hòa; dHash trùng 3 frame liền trước | Kiểm tra trên 200 frame người gán, precision/recall ≥ 0.95 |
| Ảnh IR | Trung bình kênh S (HSV) < 12 và độ lệch giữa các kênh màu rất nhỏ | Như trên |
| Vùng tĩnh | **Không** lấy phần bù của road mask (vỉa hè có xe đậu, người). Lấy 200 frame / 10 ngày, tính độ lệch chuẩn theo thời gian từng pixel, giữ 30% thấp nhất ngoài road mask đã giãn 15 px; `static_textured` = phần có gradient của nền > median | Diện tích < 2% ảnh → `static_ok = False` |
| Căn chỉnh camera | Phase correlation trên `static_textured`; ≤ 4 px bỏ qua; 4–16 px warp; > 16 px hoặc response < 0.1 → `align_ok = False`; median độ lệch theo ngày nhảy > 8 px giữ ≥ 2 ngày → `cam_epoch` mới | Sai số ≤ 0.5 px trên ảnh tự dịch |

**Chia dữ liệu theo cụm địa lý** (sửa lỗi rò rỉ): camera cùng nút giao (hoặc trong bán kính 150 m, DBSCAN haversine) thuộc cùng một cụm; gán nguyên cụm vào train / val / test theo tỷ lệ 70 / 10 / 20% số camera, phân tầng theo loại đường và quận; lưu `configs/splits.json` dùng chung cho mọi hướng. Pretrain G chỉ dùng frame của cụm train.

**Tập nhãn cần cho G:**

| Tập | Quy mô | Dùng cho |
| --- | --- | --- |
| Audit — mask xe | Khoảng 500 frame từ 60 camera phân tầng theo loại đường và nhóm giờ; SAM với box prompt + người duyệt; 10% gán đôi (IoU giữa người gán ≥ 0.8) | G-M1 (đánh giá TAM), probe dày, G-D3 (xóa xe) |
| Gold mức ùn tắc | 800–1.000 frame, 4 mức (thông thoáng / trung bình / chậm / kẹt), 3 người gán, Fleiss' κ ≥ 0.6, đa số + phân xử | Probe mức ùn tắc (G-M2) |
| Nhãn đếm xe | Nhãn đếm hiện có (xe máy, ô tô, xe buýt/tải) | Probe đếm few-shot (G-M2) |

**Metric:** macro-F1 và QWK (quadratic weighted kappa) cho mức ùn tắc; MAE cho đếm; IoU cho mask; AUROC và AP mức patch cho TAM (nhãn patch = ≥ 30% diện tích là xe).

**Quy tắc thống kê:** 3 seed cho thí nghiệm P0; khoảng tin cậy 95% bằng **bootstrap theo cụm camera** (không theo frame); so sánh với baseline mạnh nhất bằng Wilcoxon theo cụm, hiệu chỉnh Holm–Bonferroni trong mỗi bảng; **cùng ngân sách chỉnh siêu tham số** (20 lần thử Optuna trên val) cho G và mọi baseline học được.

**Quy tắc chống rò rỉ cho agent:** tập test (cụm test, audit-test, gold-test) chỉ được đọc trong `evaluate_*.py`; code train, chọn checkpoint, chọn siêu tham số không được import đường dẫn test; mọi lần chạy lưu config, git commit, seed, giờ GPU.

## 10. Spec cho agent

**Thư mục `directionG_camera_ssl/`** — dùng lại `common/` (mục 9) và phần DINO/iBOT đã có trong `direction1_bg_guided_dino/`.

| File | Nội dung chính |
| --- | --- |
| `features.py` | `FrozenExtractor`: DINOv3 ViT-B đóng băng, khung 448 × 256 → token (B, 448, 768) → PCA → (B, 448, 64) chuẩn hóa L2; `fit_pca.py` fit và lưu PCA |
| `tam.py` | `PositionStats`, `GMMCalibrator`, `activity_prior()` |
| `init_tam.py` | Khởi tạo offline bằng k-means theo vị trí |
| `sampler.py` | `CameraGroupedSampler`: mỗi batch 32 camera × 8 frame; `find_srs_partner()` |
| `srs.py` | `static_region_swap()` |
| `masking.py` | `agm_sample()`, `map_to_crop()` |
| `augment.py` | Multi-crop trả về cả tham số crop (để ánh xạ π) |
| `losses.py` | DINO, iBOT, KoLeo (dùng lại H1), `ric_loss()` |
| `train.py` | Vòng huấn luyện; log các đại lượng theo dõi |
| `eval_tam.py` | Đánh giá TAM với mask xe (G-M1) |
| `eval_frozen.py` | Probe mức ùn tắc, đếm, probe dày |
| `probes.py` | Probe camera ID, độ nhất quán khi đổi nền, độ nhạy khi xóa xe |
| `toy.py` | Sinh dữ liệu đồ chơi để test |

**`tam.py` — giao diện:**

```python
class PositionStats:
    """Thống kê trạng thái theo vị trí cho mọi (camera, mode).
    mu:  (C, P, K, d) float16, chuẩn hóa L2      C = số cặp (camera, mode), P = 448
    w:   (C, P, K)    float32, tổng theo K = 1
    s:   (C, P, K)    float32
    """
    def init_kmeans(self, cid: int, U: Tensor):          # U: (n_frames, P, d)
    @torch.no_grad()
    def update(self, cids: LongTensor, U: Tensor):       # cids: (B,), U: (B, P, d)
    @torch.no_grad()
    def atypicality(self, cids, U) -> tuple[Tensor, Tensor]:
        """Trả về a (B, P) và valid (B, P) bool (False khi tập trạng thái tĩnh rỗng)."""
    def static_set(self, cid) -> Tensor:                 # (P, K) bool
    def activity_prior(self, cid) -> Tensor:             # (P,)
    def state_dict(self) / load_state_dict(self, sd)     # lưu cùng checkpoint

class GMMCalibrator:
    """GMM 2 thành phần trên log(a) theo từng cid, buffer 50.000 giá trị gần nhất."""
    def push(self, cids, a, valid)
    def refit(self)                                      # gọi mỗi 1.000 bước
    def posterior(self, cids, a) -> Tensor               # pi (B, P) trong [0, 1]
```

Cài `update` dạng vector hóa: gom mọi (cid, p) của batch, dùng `scatter_add` cho cập nhật tâm và tần suất; không vòng lặp Python theo vị trí.

**`srs.py`:**

```python
def static_region_swap(x, x2, pi, pi2, valid, valid2, ratio, gen, thr=0.2, feather=4):
    """x, x2: (3, 256, 448) cùng camera, đã căn chỉnh. pi, pi2: (16, 28).
    Trả về x_tilde và swap_mask (16, 28) bool.
    1) static = (pi < thr) & (pi2 < thr) & valid & valid2
    2) loại các ô kề tiền cảnh: static &= ~dilate((pi >= 0.5) | (pi2 >= 0.5), 1)
    3) chọn ngẫu nhiên tỷ lệ `ratio` của static
    4) ghép khối 16x16 từ x2 vào x, trộn tuyến tính `feather` px ở biên khối"""
```

**Config mẫu `configs/exp/g_full.yaml`:**

```yaml
model: {arch: vit_base_patch16, init: dinov3_vitb16, drop_path: 0.1}
tam: {backbone: dinov3_vitb16, d: 64, K: 4, eta: 0.02, eta_w: 0.005, w_min: 0.15,
      s_max_mult: 1.5, gmm_refit_every: 1000, source: frozen}   # source: frozen | ema_teacher
agm: {policy: stratified, phi: 0.5, q_max: 0.6, ratio: [0.1, 0.5], masked_img_frac: 0.5,
      phi_warmup_frac: 0.1}
srs: {p: 0.5, p_warmup_frac: 0.1, slot_window_h: 2, thr: 0.2, ratio: [0.3, 1.0], feather_px: 4}
ric: {enabled: false, weight: 0.1, temperature: 0.2, soft_negatives: true}
loss: {ibot: 1.0, koleo: 0.1, ibot_fg_beta: 0.0}
optim: {lr: 5.0e-5, layer_decay: 0.9, wd: [0.04, 0.2], warmup_frac: 0.05, clip: 3.0}
teacher: {momentum: [0.994, 1.0], temp: [0.04, 0.07], temp_warmup_frac: 0.1}
data: {batch_cameras: 32, frames_per_camera: 8, steps: 60000, split: cluster}
```

**Dữ liệu đồ chơi (`toy.py`) để test trước khi chạy thật:** nền là ảnh có kết cấu cố định với 2 chế độ sáng (sáng/tối, chọn ngẫu nhiên mỗi frame); "xe" là hình chữ nhật màu ngẫu nhiên, kích thước ngẫu nhiên, phủ trung bình 20% diện tích vùng "đường"; 50 "ngày" × 40 frame.

**Nghiệm thu:**

1. Trên dữ liệu đồ chơi, AUROC của a\_t với mask hình chữ nhật ≥ 0.95 ở cả hai chế độ sáng.
2. Thêm một hình chữ nhật **cùng màu** ở **cùng vị trí** trong 90% frame → vị trí đó bị coi là tĩnh (đúng giới hạn đã nêu); test xác nhận hành vi này.
3. `PositionStats.update` vector hóa khớp (sai số < 1e-4) với cài đặt vòng lặp tham chiếu trên 1.000 mẫu.
4. `agm_sample`: số patch che đúng bằng `n_mask`; số patch xe bị che ≤ q\_max·|F|; khi F rỗng, phân phối vị trí che không khác che đều (kiểm định chi-bình-phương trên 10.000 lần lấy mẫu).
5. `static_region_swap`: pixel trong F\_t (trừ dải làm mềm) trùng 100% với x; không ô nào thuộc F\_t ∪ F' (đã giãn) bị hoán đổi.
6. `map_to_crop`: với ảnh đồ chơi có một ô sáng, sau crop và lật, vị trí π lớn nhất trùng vị trí ô sáng.
7. Huấn luyện thử 2.000 bước trên dữ liệu đồ chơi với ViT-S: loss giảm, không NaN, IoU attention với hình chữ nhật tăng so với bước 0.

## 11. Kế hoạch thực nghiệm

Mức ưu tiên: **P0** bắt buộc cho paper; **P1** reviewer thường hỏi; **P2** phụ lục. Ablation chạy ViT-S, 20.000 bước, trừ khi ghi khác; kết quả chính chạy ViT-B, 3 seed.

**Baseline tiền huấn luyện** — tất cả ViT-B, cùng dữ liệu, cùng số bước, cùng crop (trừ B0):

| Mã | Baseline | Kiểm tra điều gì |
| --- | --- | --- |
| B0 | DINOv3 gốc, không huấn luyện thêm | Mức sàn |
| B1 | Huấn luyện tiếp DINO + iBOT, che ngẫu nhiên | Lợi ích chỉ do dữ liệu miền |
| B2 | + AttMask | Che theo attention trong một ảnh |
| B3 | + FAM-Δ (H1, ảnh nền median) | Phiên bản dựa trên ảnh nền |
| B4 | + che theo MaskCut | Phát hiện đối tượng không giám sát trên từng ảnh |
| B5 | + Background Erasing (trộn 30% một frame khác cùng camera) | Chống đường tắt nền kiểu trộn toàn ảnh |
| B6 | + cặp dương cùng camera, gần thời điểm (kiểu Focus on the Positives) | Giả thuyết ngược: học cái tĩnh |
| B7 | ADIOS (nếu đủ compute) | Che học được |

**Danh sách thí nghiệm:**

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| G-M1 | Chính | TAM định vị xe tốt đến đâu? (RQ1) | TAM vs Δ pixel (median theo slot), Δ với nền của Hướng 2 mới, MaskCut, attention \[CLS\] DINOv3, khoảng cách tới feature trung bình (một trạng thái), khoảng cách láng giềng gần nhất kiểu PatchCore | AUROC, AP mức patch trên audit-test; tách theo cao điểm, đêm, mưa | P0 |
| G-M2 | Chính | Biểu diễn tốt hơn? (RQ2) | B0–B7 vs G đầy đủ; feature đóng băng | Mức ùn tắc (linear, kNN k = 20): macro-F1, QWK; đếm few-shot 5/10/20% (ridge): MAE; **probe dày**: phân loại tuyến tính từng token patch xe / không xe, train audit-train, IoU audit-test | P0 |
| G-M3 | Chính | Lợi ích có chuyển sang bài toán khác? | Thay DINOv3 bằng backbone G cho đếm xe, mức ùn tắc theo thời gian, phân rã cảnh | Metric của từng bài toán | P1 |
| G-M4 | Chính | Hội tụ nhanh hơn? | Đánh giá kNN mỗi 10% quá trình cho B1, B2, B3, G | Metric theo giờ GPU | P1 |
| G-D1 | Chẩn đoán | Còn dựa vào nền? (RQ3) | Logistic regression đoán camera ID từ \[CLS\], trên frame ngày khác của camera train | Độ chính xác (thấp hơn = ít dựa nền), vẽ cùng metric G-M2 trên một biểu đồ phân tán | P0 |
| G-D2 | Chẩn đoán | Bất biến với nền? | Biến đổi **không dùng lúc train**: hoán đổi vùng tĩnh từ slot lệch ±6 giờ; đổi gamma riêng vùng tĩnh | Cosine \[CLS\] trước/sau; tỷ lệ giữ nguyên dự đoán mức ùn tắc | P0 |
| G-D3 | Chẩn đoán | Có thật sự nhìn xe? (RQ4) | Xóa xe theo **mask người gán** (audit-test): thay pixel xe bằng pixel cùng vị trí của frame vắng nhất cùng camera, cùng chế độ | Mức giảm số xe dự đoán và mức ùn tắc dự đoán (càng giảm mạnh càng tốt) | P0 |
| G-D4 | Chẩn đoán | Attention dồn vào đâu? | Tỷ lệ khối lượng attention \[CLS\] rơi vào mask xe | % | P1 |
| G-AB1 | Ablation | Thành phần nào đóng góp? | B1 / + AGM / + SRS / + AGM + SRS / + RIC / + GRL | G-M2 và G-D1–D3 | P0 |
| G-AB2 | Ablation | Thiết kế TAM | K ∈ {1, 2, 4, 8}; tiêu chí tĩnh: chỉ tần suất / tần suất + độ chặt; khoảng cách: tới trạng thái tĩnh / láng giềng gần nhất / trung bình; d ∈ {32, 64, 128, 768}; nguồn feature: ViT-S đóng băng / ViT-B đóng băng / teacher EMA; thống kê theo chế độ / theo nhóm giờ | G-M1, G-M2 | P0 (K, khoảng cách, nguồn); P1 (còn lại) |
| G-AB3 | Ablation | Ngưỡng tiền cảnh | GMM / cố định top 20% / ngưỡng tối ưu trên audit-val (tham chiếu) | F1 của G-M1; G-M2 | P1 |
| G-AB4 | Ablation | Nên che bao nhiêu xe? | 4 chính sách ở mục 6 + lưới φ × q\_max (1 seed) | G-M2; **bề mặt hiệu năng** | P0 |
| G-AB5 | Ablation | Thiết kế SRS | Tắt / trộn toàn ảnh (B5) / hoán đổi patch **ngẫu nhiên** không theo TAM (control) / hoán đổi vùng tĩnh; cửa sổ cặp: cùng slot / ±2 giờ / bất kỳ; làm mềm biên 0 / 4 px | G-M2, G-D2 | P0 (3 mục đầu) |
| G-AB6 | Ablation | Trọng số loss iBOT theo π | β ∈ {0, 1, 2} | G-M2 | P2 |
| G-R1 | Robustness | Cần bao nhiêu ngày dữ liệu? | Số ngày cho khởi tạo và cập nhật TAM: 1 / 3 / 7 / 14 / tất cả; giữ 1/2, 1/4, 1/8 số frame mỗi camera | G-M1, G-M2 | P0 |
| G-R2 | Robustness | Tổng quát? | Probe với chia theo vùng (giữ 1–2 quận làm test) và ngày → đêm (train ảnh ngày, test ảnh IR); probe đếm trên dữ liệu Đà Nẵng nếu giấy phép cho phép | G-M2 | P1 |
| G-R3 | Robustness | Frame xấu | 8 loại nhiễu kiểu ImageNet-C × 5 mức lên probe đóng băng | Suy giảm tương đối | P1 |
| G-R4 | Robustness | Không căn chỉnh camera | Bỏ bước căn chỉnh | G-M1, G-M2 | P1 |
| G-R5 | Robustness | Camera hay kẹt | Báo cáo riêng 10% camera có tỷ lệ vị trí "không có trạng thái tĩnh" cao nhất | G-M1, G-M2 | P0 |
| G-S1 | Quy mô | Dữ liệu và mô hình | 10 / 25 / 50 / 100% camera train; ViT-S vs ViT-B | G-M2 | P1 |
| G-E1 | Chi phí | Có triển khai được? | Thời gian/bước, giờ GPU tổng, bộ nhớ thống kê; suy luận như ViT thường | — | P0 |
| G-Q1 | Định tính | TAM trông thế nào? | TAM, Δ, MaskCut trên cùng frame: cao điểm, đêm, mưa; ví dụ SRS | Hình | P0 |
| G-Q2 | Định tính | Attention | B1, B3, G cạnh nhau | Hình | P0 |
| G-Q3 | Định tính | Lỗi | Làn luôn kẹt, xe đậu cố định, camera rung | Hình + phân tích | P0 |

**Công bằng cho G-D2:** G được train với SRS nên sẽ "có lợi" nếu đánh giá bằng đúng phép SRS. Vì vậy G-D2 chỉ dùng biến đổi không có lúc train. TAM dùng để tạo biến đổi lấy từ DINOv3 đóng băng — giống nhau cho mọi mô hình được so.

**Bảng/hình cho paper:** Bảng 1 = G-M1; Bảng 2 = G-M2 (mọi baseline); Bảng 3 = G-D1–D3; Bảng 4 = G-AB1; Bảng 5 = G-AB2. Hình: bề mặt φ × q\_max (G-AB4), biểu đồ phân tán camera ID vs metric xe (G-D1), G-Q1, G-Q2.

## 12. Rủi ro, cổng quyết định, lịch, venue

**Ba cổng quyết định** — không qua cổng thì không tiêu thêm compute cho bước sau:

1. **Cổng 1 (trước mọi lần pretrain, khoảng tuần 3–4):** chạy G-M1 trên audit-val. Điều kiện: AUROC của TAM không thấp hơn Δ pixel ở tổng thể, và cao hơn ít nhất 0.05 ở lát cao điểm hoặc đêm. Trượt → thử các biến thể G-AB2; vẫn trượt thì dùng nguồn tiền cảnh khác (nền học từ Hướng 2 mới, hoặc MaskCut) và giữ SRS.
2. **Cổng 2 (ViT-S, 20.000 bước):** G (AGM + SRS) phải vượt B1 và B2 trên kNN mức ùn tắc và probe dày, với mức chênh lớn hơn độ lệch chuẩn giữa 2 seed của B1. Trượt → bề mặt φ × q\_max và các chẩn đoán vẫn có giá trị — gộp thành phần phân tích trong paper khác, không viết paper G riêng.
3. **Cổng 3 (sau kết quả ViT-B):** G-D1–D3 phải cho thấy giảm phụ thuộc nền. Nếu G thắng trên bài toán nhưng chẩn đoán không đổi, câu chuyện "chống đường tắt nền" không đứng — viết lại luận điểm thành "che có hướng dẫn từ chuỗi ảnh".

**Rủi ro:**

| Rủi ro | Dấu hiệu | Cách giảm |
| --- | --- | --- |
| TAM hỏng ở làn luôn kẹt | Tỷ lệ vị trí không có trạng thái tĩnh cao | Báo cáo riêng (G-R5); AGM tự lùi về che ngẫu nhiên ở đó |
| Mô hình học dấu vết đường nối của SRS | G-D2 tốt nhưng G-M2 không tăng | Làm mềm biên; ablation 0 / 4 px; control hoán đổi ngẫu nhiên |
| Lợi ích chỉ do huấn luyện thêm trên dữ liệu miền | G ≈ B1 | B1 đã kiểm soát yếu tố này — nếu G ≈ B1 thì đó là kết luận trung thực |
| Chi phí forward thêm cho TAM | Thời gian/bước tăng > 40% | TAM bằng ViT-S, hoặc cache feature PCA-64 |
| Giấy phép DINOv3 | Muốn công bố trọng số đã huấn luyện tiếp | Đọc điều khoản DINOv3 trước; không được thì công bố code và cách tái tạo |
| Chỉ một thành phố | Reviewer hỏi tổng quát | Probe đếm xe trên dữ liệu Đà Nẵng bằng feature backbone (không cần TAM lúc suy luận) |

**Lịch khoảng 5 tháng** (sau khi nền móng ở mục 9 đã có):

1. **Tháng 1:** `toy.py`, `tam.py`, `init_tam.py`, `eval_tam.py`; chạy G-M1 → **Cổng 1**. Cần mask xe của tập audit.
2. **Tháng 2:** `masking.py`, `srs.py`, `sampler.py`; ablation ViT-S: G-AB1, G-AB2, G-AB4, G-AB5 → **Cổng 2**.
3. **Tháng 3:** ViT-B, cấu hình chốt, 3 seed; baseline B0–B6 cùng compute.
4. **Tháng 4:** chẩn đoán G-D1–D4 (**Cổng 3**), robustness G-R1–R5, chuyển giao G-M3.
5. **Tháng 5:** hình, bảng, viết paper.

**Quan hệ với các hướng khác:** nếu G qua Cổng 2, backbone G thay DINOv3 gốc trong các hướng đếm xe, mức ùn tắc và phân rã cảnh. TAM (mục 5) được Hướng 2 mới dùng ngay sau Cổng 1, không cần chờ G huấn luyện xong.

**Venue:** Pattern Recognition, IEEE TIP, CVIU (nếu nhấn mạnh phương pháp tự học); IEEE T-ITS, EAAI (nếu nhấn mạnh ứng dụng giao thông).

## Nguồn tham khảo

**Đã xác nhận qua trang bài báo hoặc trang chính thức (10/2026):**

- [MGMAE — Huang et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_MGMAE_Motion_Guided_Masking_for_Video_Masked_Autoencoding_ICCV_2023_paper.html)
- [MGM — Fan et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Fan_Motion-Guided_Masking_for_Spatiotemporal_Representation_Learning_ICCV_2023_paper.html)
- [V-JEPA4A / Mask What Matters — Lang, Braun, Valada, arXiv 2608.17178](https://www.alphaxiv.org/abs/2608.17178)
- [ADIOS — Shi et al., ICML 2022](https://proceedings.mlr.press/v162/shi22d.html)
- [Background Erasing — Wang et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Removing_the_Background_by_Adding_the_Background_Towards_Background_Robust_CVPR_2021_paper.html)
- [Jenni & Favaro — Self-Supervised Multi-View Synchronization Learning, arXiv 2010.06218](https://arxiv.org/pdf/2010.06218)
- [Focus on the Positives — Pantazis et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Pantazis_Focus_on_the_Positives_Self-Supervised_Learning_for_Biodiversity_Monitoring_ICCV_2021_paper.html)
- [MoCo với temporal positives cho ảnh vệ tinh — arXiv 2210.11815](https://arxiv.org/pdf/2210.11815)
- [TimeT — Salehi et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Salehi_Time_Does_Tell_Self-Supervised_Time-Tuning_of_Dense_Image_Representations_ICCV_2023_paper.html)
- [CutLER / MaskCut — Wang et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Cut_and_Learn_for_Unsupervised_Object_Detection_and_Instance_Segmentation_CVPR_2023_paper.pdf)
- [Hard Negative Mixing — Kalantidis et al., NeurIPS 2020](https://proceedings.nips.cc/paper_files/paper/2020/file/f7cade80b7cc92b991cf4d2806d6bd78-Paper.pdf)
- [DINOv3 — giấy phép và các biến thể (tài liệu LightlyTrain)](https://docs.lightly.ai/train/stable/pretrain_distill/models/dinov3.html)

**Ghi theo hiểu biết chung, chưa kiểm chứng — tra DOI trước khi đưa vào paper:** DINOv2 (Oquab et al.); iBOT (Zhou et al., ICLR 2022); AttMask (Kakogeorgiou et al., ECCV 2022); PatchCore (Roth et al., CVPR 2022); ImageNet-C (Hendrycks & Dietterich, ICLR 2019); DANN / gradient reversal (Ganin & Lempitsky); KoLeo regularizer trong DINOv2.
