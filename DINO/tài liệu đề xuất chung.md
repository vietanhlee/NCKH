# Đề xuất nghiên cứu: Cặp ảnh nền & chuỗi ảnh camera giao thông TP.HCM — coi background là tín hiệu yếu

Oct 2, 2026 · @Hung Do

## Tóm tắt

Ảnh background hiện có là sản phẩm của median filter theo slot giờ, chưa được kiểm chứng, nên mọi hướng nghiên cứu phải coi nó là **tín hiệu yếu, có nhiễu, chưa biết độ tin cậy** — không phải ground truth. Tài liệu này làm 3 việc: (1) đặt nguyên tắc dùng background và giao thức đo độ tin cậy của nó; (2) sửa 4 hướng cũ (H1–H4) để đứng vững trước reviewer Q1; (3) trình bày chi tiết 6 hướng mới (A–F) ít phụ thuộc vào độ chuẩn của background.

Quy tắc xuyên suốt: background **không** được dùng để lấy ROI phương tiện, không dùng làm nhãn cứng, không dùng ngưỡng trên Δ rồi gọi là "vật lý". Background chỉ được dùng ở dạng mềm (trọng số ngẫu nhiên, thống kê toàn cục, một nguồn nhãn yếu trong nhiều nguồn), và mọi paper đều phải có thí nghiệm chứng minh phương pháp không sụp khi background xấu đi.

## Đánh giá mức sẵn sàng Q1

**Kết luận ngắn:** sau đợt rà soát này, *thiết kế* của các hướng đã đủ chặt về phương pháp để bắt tay triển khai. Còn "đủ để nộp Q1" thì chỉ trả lời được khi có kết quả. Một paper Q1 sẽ đứng hay đổ ở 3 điểm: (1) phương pháp thắng baseline mạnh nhất với khoảng tin cậy không chồng lấn, đúng ở các điều kiện khó (cao điểm, đêm); (2) có bằng chứng định lượng về độ tin cậy của background (tập audit, mục 0.2); (3) không có rò rỉ dữ liệu và mọi so sánh đều công bằng. Phần IV là đặc tả để agent code đúng ngay từ đầu.

### Các lỗi tìm thấy trong đợt rà soát này

| # | Vấn đề | Mức độ | Hậu quả nếu bỏ qua | Đã sửa ở |
| --- | --- | --- | --- | --- |
| 1 | Chia train/test theo camera ID, nhưng nhiều camera cùng nhìn một nút giao từ các góc khác nhau | Nghiêm trọng | Rò rỉ không gian: mô hình đã thấy cùng cảnh, kết quả bị thổi phồng | IV.3 — chia theo cụm địa lý |
| 2 | "Vùng tĩnh ngoài mặt đường" dùng để đo độ tin cậy (I.2, A.6) có vỉa hè — ở Việt Nam vỉa hè đầy xe máy đậu và người | Nghiêm trọng | Ước lượng độ lệch camera và độ tin cậy r\_i bị sai | IV.3 — vùng tĩnh tính từ phương sai theo thời gian |
| 3 | Road mask chưa định nghĩa rõ có gồm vỉa hè không | Trung bình | ρ và chiếm dụng không so sánh được giữa camera | IV.2 — road mask = lòng đường xe chạy |
| 4 | Baseline và phương pháp đề xuất chưa có cùng ngân sách chỉnh siêu tham số | Nghiêm trọng | Reviewer bác bỏ so sánh | IV.5 — cùng số lần thử, cùng tập val |
| 5 | Chưa có giao thức gán nhãn (hướng dẫn, đào tạo, ngưỡng đồng thuận, phân xử) | Nghiêm trọng | Gold set không đáng tin, κ thấp | IV.5 |
| 6 | Kích thước tập đánh giá chọn cảm tính | Trung bình | Khoảng tin cậy quá rộng, không kết luận được | IV.5 — quy tắc kiểm tra độ rộng khoảng tin cậy |
| 7 | So sánh nhiều phương pháp, nhiều điều kiện nhưng chưa hiệu chỉnh đa so sánh | Trung bình | Kết luận "có ý nghĩa" sai | IV.5 — Holm–Bonferroni |
| 8 | Mốc thời gian: dữ liệu lưu UTC hay UTC+7 chưa thống nhất, mà slot giờ tính theo giờ Việt Nam | Trung bình | Background và frame lệch slot | IV.2 |
| 9 | Tuyên bố "đầu tiên" chưa có tra cứu tài liệu có hệ thống | Trung bình | Bị reject vì có công trình trước | IV.13 — giao thức tra cứu |
| 10 | Trích dẫn trong tài liệu ghi theo trí nhớ | Nhỏ | Sai năm, sai hội nghị | IV.13 |
| 11 | Chưa có kế hoạch tái lập (seed, config, phát hành code, báo cáo compute) | Trung bình | Q1 ngày càng yêu cầu | IV.1, IV.13 |
| 12 | Ảnh trong paper có thể lộ biển số, khuôn mặt | Trung bình | Vấn đề đạo đức, bị yêu cầu sửa | IV.13 |
| 13 | H1: ảnh ban đêm hoặc đường vắng có Δ gần như phẳng → FAM suy biến về che ngẫu nhiên | Nhỏ | Không sai, nhưng phải báo cáo tỷ lệ ảnh như vậy | IV.6 |
| 14 | Hướng D: đại lượng dự báo do mô hình khác tạo ra (vòng tròn) | Trung bình | Đánh giá tự xác nhận | II.D đã nêu; IV.11 bắt buộc đánh giá trên nhãn người |

### Những gì vẫn còn là giả định cần kiểm chứng sớm

- Tần suất lấy mẫu thực tế (quyết định tính khả thi của B-LF4, C, D).
- Background có thực sự sai nhiều ở cao điểm không (tập audit sẽ trả lời; nếu không sai nhiều, Hướng A mất động lực chính).
- FAM có hơn AttMask không (quyết định H1).
- Có được phép công bố dữ liệu không (quyết định F).

## 0. Nguyên tắc nền: background là tín hiệu yếu, phải đo trước khi tin

Median filter theo thời gian chỉ đúng khi mỗi pixel thấy mặt đường trống hơn 50% thời gian trong slot. Điều này sai đúng ở những lúc quan trọng nhất: giờ cao điểm, kẹt xe, xe đậu lâu, xe buýt dừng đỗ. Khi đó xe bị "nuốt" vào nền (ghost vehicle), Δ ≈ 0 ngay trên xe. Ngoài ra còn bóng đổ di chuyển theo giờ, mưa làm mặt đường đổi màu, đèn xe ban đêm, camera bị rung hoặc xoay nhẹ, chuyển chế độ hồng ngoại.

### 0.1. Được và không được dùng background thế nào

| Cách dùng | Đánh giá | Lý do |
| --- | --- | --- |
| Lấy ROI/bounding box phương tiện từ Δ | Không được | Không có bằng chứng Δ khớp với xe; sai hệ thống ở vùng đông |
| Ngưỡng Δ > τ rồi gọi là "mật độ vật lý" | Không được | τ chọn tay, sai số không đo được, bị bias ở giờ kẹt |
| Ép mô hình khớp từng pixel với background (L1 cứng) | Hạn chế | Mô hình học luôn cả ghost và nhiễu |
| Trọng số mềm, có ngẫu nhiên (ví dụ masking có xác suất) | Được | Sai ở vài patch chỉ làm lệch xác suất, không làm sai nhãn |
| Thống kê toàn cục (mean/std feature của cả ảnh nền) | Được | Lỗi cục bộ bị trung bình hóa |
| Một nguồn nhãn yếu trong nhiều nguồn | Được | Mô hình gộp nhãn tự học độ tin cậy của từng nguồn |
| Mẫu tham chiếu "bình thường" trong nhiều mẫu | Được | Không phải mẫu duy nhất quyết định |

### 0.2. Background Reliability Audit — bằng chứng bắt buộc cho mọi paper

Thay vì nói "background không chuẩn lắm", hãy **đo** nó. Một tập kiểm định nhỏ do người gán là đủ để trả lời câu hỏi "tin được ở mức nào, trong điều kiện nào". Tập này dùng chung cho mọi hướng và tự nó đã là một phần đóng góp.

**Thiết kế tập audit (ước tính 3–5 ngày công gán nhãn):**

1. Chọn khoảng 60 camera, phân tầng theo loại đường (trục chính, nút giao, đường nhỏ) và góc nhìn.
2. Mỗi camera lấy 6 slot: 2 slot đêm, 2 slot thấp điểm ngày, 2 slot cao điểm → khoảng 360 ảnh background.
3. Với mỗi ảnh background, người gán vẽ polygon các vùng **ghost** (xe còn sót, vệt mờ của xe) và đánh dấu lỗi khác: bóng đổ, ướt, lóa, lệch camera.
4. Với mỗi background, lấy thêm 1–2 ảnh origin cùng slot và gán **mask hoặc box phương tiện** (dùng SAM với box prompt để tăng tốc, người duyệt lại) → khoảng 500 ảnh có mask xe.
5. Gán thêm vùng mặt đường (road mask) cho 60 camera này; nên làm luôn road mask cho cả 608 camera vì hướng H4, B, D đều cần.

**Các chỉ số báo cáo:**

- Ghost rate: tỷ lệ diện tích mặt đường bị ghost, theo slot giờ và loại đường.
- Precision/Recall của "Δ > τ" so với mask xe thật, quét τ từ 0.02 đến 0.3 → đường PR và IoU tốt nhất có thể đạt.
- Phân tích theo điều kiện: ngày/đêm, mưa/khô, mật độ xe (thưa/vừa/đông).

Kết quả kỳ vọng (cần kiểm chứng): Δ khá ổn ở thấp điểm ban ngày, kém rõ rệt ở cao điểm và ban đêm. Một bảng số liệu như vậy cho phép viết trong paper câu "background đáng tin ở điều kiện X với IoU = …, kém ở điều kiện Y" thay vì cảm nhận định tính. Nó cũng định hướng thiết kế: phương pháp nào dựa vào Δ phải được kiểm tra riêng ở các điều kiện background kém.

## I.1. Thí nghiệm chung: Background Degradation Benchmark (BDB)

Mọi phương pháp dùng background phải trả lời câu hỏi của reviewer: "nếu background tệ hơn thì sao?". BDB làm xấu background một cách có kiểm soát ở **lúc test** (và tùy chọn ở lúc train), rồi vẽ đường cong hiệu năng theo mức độ nhiễu.

### Các loại nhiễu và mức độ

| Loại nhiễu | Mô phỏng lỗi thực tế | Mức độ (severity 1→5) |
| --- | --- | --- |
| Sai slot giờ | Ánh sáng/bóng đổ lệch | ±1h, ±2h, ±4h, ±8h, đảo ngày↔đêm |
| Dịch/xoay ảnh | Camera rung, bị chỉnh góc | dịch 2, 4, 8, 16, 32 px; xoay 0.5°–3° |
| Chèn ghost vehicle | Median nuốt xe ở giờ kẹt | dán crop xe thật lên nền phủ 5%, 10%, 20%, 30%, 50% diện tích mặt đường |
| Thay đổi quang học | Mưa, mây, đổi phơi sáng | gamma 0.7–1.5, độ sáng ±10–40% |
| Nhiễu và nén | Camera chất lượng thấp | Gaussian σ = 5–40; JPEG quality 90→10 |
| Background của camera khác | Kiểm tra mô hình có dùng nội dung bg không | ngẫu nhiên camera khác, cùng slot |

Loại "chèn ghost" là quan trọng nhất vì nó mô phỏng đúng lỗi hệ thống của median filter. Crop xe lấy từ mask xe trong tập audit (mục 0.2), dán vào vị trí trên mặt đường bằng Poisson blending.

### Các mốc so sánh bắt buộc (control)

1. **No-bg baseline**: cùng kiến trúc, cùng số tham số, không dùng background (ví dụ kênh Δ = 0, hoặc masking ngẫu nhiên). Đây là đường ngang trên biểu đồ.
2. **Random-bg control**: thay background bằng background camera khác. Nếu kết quả gần như không đổi so với bg thật thì phần cải thiện không đến từ nội dung background mà từ tham số/kênh thêm vào — reviewer sẽ bắt lỗi điểm này.
3. **Oracle-bg** (nếu có): background được người sửa tay trên tập audit, cho biết trần hiệu năng khi bg sạch.

### Cách báo cáo

- Một hình gồm các đường cong: trục x là severity, trục y là metric chính (MAE, mIoU, F1…), mỗi đường là một phương pháp, đường ngang là No-bg baseline.
- Một con số tóm tắt: **diện tích dưới đường cong suy giảm** (giống mCE trong ImageNet-C), và **điểm cắt** — mức nhiễu mà phương pháp tụt xuống dưới No-bg baseline.
- Tuyên bố mạnh nhất có thể đưa vào paper: "phương pháp của chúng tôi không bao giờ tệ hơn No-bg baseline ở mọi mức nhiễu" (graceful degradation). Nếu không đạt được điều này, cần cơ chế tự tắt background (xem bg-dropout và cổng tin cậy ở các mục sau).

### Huấn luyện có nhiễu (tùy chọn nhưng nên có)

Áp dụng ngẫu nhiên các nhiễu trên với xác suất 30–50% lúc train, cộng thêm **bg-dropout** (bỏ hẳn background với xác suất 20–30%). Mô hình học cách dùng background khi nó tốt và bỏ qua khi nó xấu. So sánh "train sạch / test nhiễu" và "train nhiễu / test nhiễu" là một bảng ablation rất thuyết phục.

## I.2. Cải thiện Hướng 1 — BG-Guided DINO

Cập nhật: H1 đã được thay thế bởi Hướng G (II.G), không cần ảnh nền; FAM-Δ dưới đây chỉ còn là baseline B3 của G. Ý tưởng FAM dùng background ở dạng mềm (xác suất che) nên hợp với nguyên tắc ở mục 0. Tuy nhiên có 4 lỗi kỹ thuật cần sửa trước khi viết paper.

### Lỗi 1: Công thức P\_mask không phải phân phối xác suất

Tổng của α·w\_p/max(w) + (1−α)/N trên N patch không bằng 1. Ví dụ ảnh có 196 patch, nhiều patch có w\_p gần max thì tổng có thể vượt 50. Sửa bằng cách chuẩn hóa theo tổng, có nhiệt độ T để điều khiển độ "nhọn":

```latex
P_{\text{mask}}(p) = \alpha \cdot \frac{\exp(\tilde{w}_p / T)}{\sum_{q} \exp(\tilde{w}_q / T)} + (1-\alpha) \cdot \frac{1}{N}
```

Rồi lấy mẫu không hoàn lại (sampling without replacement) đúng số patch cần che theo tỷ lệ che (ví dụ 40%).

### Lỗi 2: Chuẩn hóa theo max nhạy với nhiễu

Một patch nhiễu (lóa đèn, ghost bị lệch) có Δ rất lớn sẽ đè bẹp mọi patch khác. Dùng w̃\_p là **thứ hạng (rank) chuẩn hóa** của w\_p, hoặc cắt ở phân vị 95–99 trước khi chuẩn hóa. Cách dùng rank còn có lợi: không phụ thuộc thang đo Δ giữa ngày và đêm.

### Lỗi 3: Che patch nhưng không có loss trên patch bị che

Hàm loss trong tài liệu chỉ có loss DINO trên token \[CLS\]. Khi đó che patch chỉ là một kiểu augmentation làm mất thông tin, không phải masked image modeling. Muốn FAM có tác dụng thật, cần thêm loss mức patch kiểu iBOT (Zhou et al., ICLR 2022) như DINOv2 đang dùng: student dự đoán token patch của teacher tại vị trí bị che.

```latex
\mathcal{L} = \mathcal{L}_{\text{DINO}}^{[CLS]} + \lambda_{\text{iBOT}} \sum_{p \in \mathcal{M}} -P_t(z_p) \log P_s(\hat{z}_p)
```

Với tập patch bị che 𝓜 lấy mẫu theo FAM, gradient tập trung vào tái tạo ngữ nghĩa vùng có xe — đây mới là luận điểm chính của H1.

### Lỗi 4: Không có cơ chế tự tắt khi background xấu

Thêm **cổng tin cậy theo từng ảnh** α\_i = α\_max · r\_i. Độ tin cậy r\_i ước lượng không cần nhãn bằng cách so origin và background trên **vùng tĩnh không phải mặt đường** (tòa nhà, cột điện, bầu trời — tính từ phương sai theo thời gian như IV.3; không dùng phần bù của road mask vì vỉa hè có xe đậu và người). Ở vùng này không có xe, nên nếu origin và bg vẫn lệch nhiều thì bg đang sai do lệch camera hoặc sai ánh sáng.

```latex
r_i = \exp\left(-\frac{\overline{\Delta}_{\text{static}, i}}{\kappa}\right)
```

Ví dụ: camera bị xoay 2° → Δ trên tường nhà tăng mạnh → r\_i nhỏ → FAM tự lùi về masking ngẫu nhiên. Cổng này dùng được cho cả H3 và H4.

### Thí nghiệm cần có

| Nhóm | Cụ thể |
| --- | --- |
| Baseline cùng compute | DINOv2 gốc; continual DINO + masking ngẫu nhiên; continual DINO + **AttMask** (Kakogeorgiou et al., ECCV 2022 — che theo attention, không cần bg) |
| Downstream | kNN/linear probe đếm xe few-shot 5/10/20%; phân loại mức ùn tắc trên gold set; fine-tune phát hiện xe trên tập nhỏ có box |
| Phân tích | IoU giữa attention map và mask xe thật (tập audit); tỷ lệ diện tích mặt đường thực tế đo từ road mask thay cho con số 70–80% đang ước lượng |
| Độ bền | BDB với ghost injection, có và không có cổng tin cậy |
| Ablation | α ∈ {0, 0.25, 0.5, 0.75, 1}; T; tỷ lệ che; rank vs max-norm |

Rủi ro lớn nhất: **AttMask cho kết quả ngang FAM**. Khi đó background không mang lại gì thêm. Nên chạy so sánh này sớm nhất (khoảng 1–2 tuần compute) trước khi đầu tư viết paper.

## I.3. Hướng 2 — đã nâng cấp và chuyển sang mục II.A

Hướng 2 (Scene Decomposition) và Hướng A là cùng một bài toán: II.A chính là H2 sau khi sửa để không phụ thuộc vào background. Toàn bộ phần sửa cho H2 (ràng buộc nền dùng chung nhiều frame, prior mềm có độ bất định, hai chế độ suy luận, cách đánh giá khi không có ground truth, định vị lại độ mới) được viết chi tiết ở **II.A** để tránh lặp. Thư mục `direction2_scene_decomposition/` giữ nguyên làm điểm xuất phát cho code.

## I.4. Cải thiện Hướng 3 — Foreground-Enhanced Counting

Đưa Δ thẳng vào patch embedding làm mô hình dễ phụ thuộc vào Δ, mà Δ lại sai nhiều nhất ở giờ đông — đúng lúc bài toán đếm khó nhất. Hướng sửa: biến H3 thành một nghiên cứu so sánh **"nên tiêm một prior không đáng tin vào đâu"**, thay vì chỉ đề xuất kênh thứ 4.

### Sửa 1: Khởi tạo kênh Δ bằng 0 thay vì trung bình RGB

Khởi tạo bằng trung bình trọng số RGB làm Δ đóng góp ngay từ epoch 0, trong khi phân phối của Δ (phần lớn gần 0, vài vùng rất lớn) khác hẳn RGB → dễ gây sốc cho các tầng sau. Khởi tạo bằng 0 (giống zero-convolution của ControlNet, Zhang et al., ICCV 2023) giữ nguyên hành vi của mô hình pretrained lúc đầu và để gradient tự "mở" kênh Δ. Báo cáo ablation: mean-init vs zero-init vs random-init.

### Sửa 2: Δ-dropout và kiểm chứng nội dung

- Lúc train, đặt kênh Δ = 0 với xác suất 30% → mô hình không thể chỉ dựa vào Δ.
- Control bắt buộc: kênh thứ 4 = ảnh xám của origin, hoặc Δ tính với background của camera khác. Nếu hai control này cho kết quả gần bằng Δ thật thì cải thiện đến từ tham số thêm vào, không phải từ background.

### Sửa 3: So sánh các vị trí tiêm prior

| Biến thể | Mô tả | Kỳ vọng khi bg nhiễu |
| --- | --- | --- |
| Early fusion (H3 gốc) | Δ là kênh thứ 4 ở patch embedding | Nhạy nhiễu nhất |
| Late fusion | CNN nhỏ trên Δ → vector nối với \[CLS\] | Ít nhạy hơn, mô hình có thể bỏ qua |
| Global conditioning (Hướng E) | Chỉ dùng thống kê toàn cục của bg qua FiLM | Ít nhạy nhất |
| No-bg | DINOv3 + head hồi quy | Mốc so sánh |

Chạy cả 4 biến thể qua BDB (mục I.1). Một hình đường cong suy giảm của 4 biến thể là kết quả trung tâm của paper.

### Sửa 4: Báo cáo theo điều kiện, không chỉ trung bình

- Chia sai số (MAE, RMSE) theo 3 mức mật độ (thưa/vừa/đông) và ngày/đêm. Có khả năng Δ giúp ở mức thưa và trung bình, nhưng không giúp hoặc làm hại ở mức đông — cần trình bày trung thực.
- Few-shot 5/10/20%: lặp 3–5 seed lấy mẫu khác nhau, báo cáo mean ± std; tập train và test tách theo cụm camera (IV.3) thay vì theo từng camera.
- Trọng số LAB 0.5/0.25/0.25 đang chọn tay: hoặc ablation vài bộ trọng số, hoặc đưa |ΔL|, |Δa|, |Δb| thành 3 kênh riêng cho mô hình tự học.

## I.5. Cải thiện Hướng 4 — Spatio-Temporal Density & LoS

H4 có ý tưởng ứng dụng tốt nhất nhưng nhãn đang yếu nhất: ρ\_phys lấy từ ngưỡng Δ, và Δ sai nhiều nhất đúng ở mức Gridlock. Có 6 điểm cần sửa.

### 1. ρ đang chia cho toàn bộ ảnh, không phải mặt đường

Công thức hiện tại chia cho HW, tức tính cả bầu trời, nhà cửa, vỉa hè. Cùng một lượng xe nhưng camera nhìn nhiều trời sẽ có ρ thấp hơn. Phải dùng road mask R của từng camera (608 polygon, khoảng 1–2 ngày công):

```latex
\rho_{\text{proxy}}(t) = \frac{1}{|R|} \sum_{(u,v) \in R} \mathbb{I}\big(\Delta_t(u,v) > \tau\big)
```

### 2. Đổi tên và đổi vai trò của ρ\_phys

Gọi là ρ\_proxy (chiếm dụng ước lượng), bỏ chữ "vật lý". Reviewer sẽ hỏi ngay: nếu mô hình chỉ học bắt chước ρ\_proxy thì sao không tính ρ\_proxy trực tiếp cho xong? Câu trả lời phải là: **mô hình tốt hơn chính nhãn yếu của nó** khi đo trên gold set. Hai cách để đạt điều này:

- **Học từ vùng tin cậy, suy ra vùng không tin cậy**: chỉ dùng ρ\_proxy làm nhãn ở frame có độ tin cậy cao (thấp điểm ban ngày, r\_i cao theo mục I.2), rồi kiểm tra mô hình trên frame cao điểm/ban đêm. Nếu mô hình vượt ρ\_proxy ở vùng khó, đó là đóng góp rõ ràng.
- **Thay ρ\_proxy bằng nhãn gộp đa nguồn** của Hướng B.

### 3. Gold set cho chiếm dụng và mức ùn tắc

Gán tay 800–1000 frame, phân tầng theo mật độ và giờ: mask xe (SAM + duyệt tay) giao với road mask → ρ\_gold; đồng thời 2–3 người gán mức ùn tắc 4 cấp. Báo cáo Cohen's/Fleiss' κ giữa người gán. Gold set dùng để: (a) đo sai số của ρ\_proxy theo điều kiện; (b) đánh giá cuối cùng; (c) hiệu chỉnh ρ\_proxy bằng isotonic regression nếu cần.

### 4. Ngưỡng LoS và tên gọi "chuẩn HCM"

HCM LoS (Highway Capacity Manual) định nghĩa theo mật độ xe/km/làn hoặc tốc độ, không theo tỷ lệ pixel chiếm dụng. Ngưỡng 0.15/0.35/0.60 hiện là tự đặt. Đổi tên thành **"mức ùn tắc dựa trên chiếm dụng" (occupancy-based congestion level)**, và chọn ngưỡng bằng dữ liệu: lấy ngưỡng trên ρ\_gold sao cho khớp tốt nhất với nhãn mức ùn tắc do người gán (tối đa hóa κ).

### 5. BiGRU nhìn thấy tương lai — không dùng được để cảnh báo sớm

BiGRU hai chiều dùng cả frame sau thời điểm t. Với mục tiêu "cảnh báo sớm kẹt xe", đây là rò rỉ thông tin. Cần tách hai chế độ:

- **Ước lượng hiện tại** (nowcasting): được dùng BiGRU trên cửa sổ quá khứ đến t.
- **Dự báo/cảnh báo sớm**: chỉ dùng GRU một chiều hoặc Transformer có causal mask, dự đoán ρ(t+h) với h = 5, 15, 30 phút. Đạo hàm ∂ρ/∂t thay bằng ρ(t+h) − ρ(t) — dễ định nghĩa hơn khi lấy mẫu thưa và nối thẳng sang Hướng D.

### 6. Loss làm mượt L2 xóa mất sự kiện đột ngột

Phạt bình phương ‖ρ̂\_{t+1} − ρ̂\_t‖² làm mờ các thay đổi đột ngột (tai nạn, mưa lớn). Dùng phạt L1 (total variation) hoặc Huber với trọng số nhỏ, và kiểm tra riêng hiệu năng ở các đoạn có thay đổi lớn.

### Baseline cần có

ρ\_proxy trực tiếp; chiếm dụng tính từ box của detector (YOLO/RT-DETR) giao road mask; mô hình đếm mật độ kiểu CSRNet; VLM zero-shot cho mức ùn tắc; DINOv3 + head không có Δ-CNN.

## II.A. Hướng 2 nâng cấp — Phân rã cảnh giao thông từ background prior không hoàn hảo

Tên gợi ý: *Noise-Aware Traffic Scene Decomposition with Imperfect Background Priors*. Mục này thay thế hoàn toàn H2 gốc. Ý chính: background median không còn là "đáp án" mà mô hình phải chép theo, mà chỉ là **một gợi ý có thể sai**. Mô hình tự học chỗ nào gợi ý sai, rồi dùng chính dữ liệu nhiều ngày để tạo ra background tốt hơn.

### A.1. Bối cảnh

Hệ thống camera đô thị có hàng triệu frame nhưng gần như không có nhãn mask phương tiện. Background median theo slot giờ là tín hiệu "miễn phí" duy nhất, nhưng sai có hệ thống:

- **Ghost**: xe đứng yên lâu (kẹt xe, xe buýt dừng, xe đậu) bị median giữ lại trong nền. Lỗi này nặng nhất đúng ở giờ cao điểm.
- **Ánh sáng và thời tiết**: bóng đổ dịch chuyển trong cùng một giờ, mặt đường ướt khi mưa, đèn xe ban đêm.
- **Hình học**: camera bị rung hoặc bị chỉnh góc giữa các ngày, làm background lệch vài pixel so với frame.

Các phương pháp background subtraction học sâu thường cần mask có nhãn để huấn luyện, còn các phương pháp phân rã lớp (Omnimatte, Layered Neural Atlases) tối ưu riêng cho từng video và cần chuyển động liên tục. Cả hai đều không hợp với dữ liệu camera đô thị lấy mẫu thưa, nhiều camera, không nhãn.

### A.2. So sánh với H2 gốc

| Khía cạnh | H2 gốc | H2 nâng cấp (II.A) |
| --- | --- | --- |
| Vai trò của background | Nhãn cứng: L1 giữa Î\_bg và background | Prior mềm, có bản đồ độ bất định σ học được |
| Đầu vào lúc suy luận | Chưa rõ (có thể cần cả background) | Chỉ cần 1 frame origin; chế độ có background là biến thể phụ |
| Chống nghiệm suy biến | Chỉ nhờ L\_bg (nên phụ thuộc hoàn toàn vào background) | Nền dùng chung K frame khác ngày + loss loại trừ + prior mềm |
| Xử lý ghost | Học theo ghost | Phát hiện ghost (σ lớn) và tự làm sạch qua nhiều vòng |
| Đánh giá | Chưa có ground truth | Tập audit, ảnh bán tổng hợp, frame trống thật, ghost cài sẵn |
| Định vị độ mới | "Đầu tiên", 5/5 | Học phân rã từ prior không hoàn hảo, ở quy mô thành phố, suy luận 1 frame |

### A.3. Phát biểu bài toán

Với mỗi camera c, slot giờ s, ngày d, ta có các frame I\_{c,s,d} và một background prior B\_{c,s} (median). Prior này sai trên một tập vị trí E\_{c,s} **không biết trước** (ghost, bóng, lệch). Cần học một mạng f\_θ chỉ từ dữ liệu không nhãn:

```latex
f_\theta(I) = \big(M_\alpha,\; F,\; \hat{B},\; \sigma\big), \qquad I \approx M_\alpha \odot F + (1 - M_\alpha) \odot \hat{B}
```

sao cho M\_α khớp với vùng phương tiện thật, B̂ khớp với mặt đường thật (kể cả dưới xe), và σ lớn đúng ở E\_{c,s}.

### A.4. Câu hỏi nghiên cứu

1. **RQ1 — Phát hiện lỗi prior:** σ học được có tách được vùng background sai mà không cần nhãn không? Đo bằng AUROC của σ so với ghost thật và ghost cài sẵn.
2. **RQ2 — Tự làm sạch background:** gộp B̂ từ nhiều frame có cho background ít ghost hơn median, MOG2 không, và cải thiện qua mấy vòng?
3. **RQ3 — Phân rã từ 1 frame:** chất lượng mask và chất lượng xóa xe so với các baseline hai giai đoạn (detector + inpainting)?
4. **RQ4 — Giá trị ứng dụng và độ bền:** dùng M\_α để ước lượng chiếm dụng mặt đường có tốt hơn ngưỡng Δ không, và có bền dưới BDB (mục I.1) không?

### A.5. Đóng góp dự kiến

1. Công thức phân rã cảnh học từ background prior **không hoàn hảo**, với prior được mô hình hóa có độ bất định.
2. Ràng buộc nền dùng chung **giữa các ngày khác nhau** — giải quyết được xe đứng yên lâu mà ràng buộc cùng ngày không xử lý được.
3. Quy trình tự làm sạch background ở quy mô thành phố và **bản đồ tin cậy** cho từng camera × slot, dùng lại được cho H1, Hướng B và Hướng E.
4. Giao thức đánh giá khi không có ground truth: tập audit, ảnh bán tổng hợp, ghost cài sẵn.

### A.6. Dữ liệu và cách lấy mẫu

**Đơn vị mẫu huấn luyện là một "nhóm", không phải một frame.** Mỗi nhóm gồm K frame của cùng camera c, cùng slot giờ s, nhưng lấy từ **K ngày khác nhau**, kèm prior B\_{c,s}. Mặc định K = 4. Lý do chọn khác ngày: mặt đường gần như không đổi giữa các ngày, còn xe thì khác. Một chiếc xe đứng yên suốt 1 tiếng trong ngày d sẽ không có mặt ở cùng vị trí vào ngày d' — điều mà ràng buộc cùng ngày (H2 gốc) không làm được.

Ví dụ: camera 125, slot 17h, lấy frame 17:20 ngày 03/09, 17:05 ngày 11/09, 17:40 ngày 18/09, 17:15 ngày 25/09. Cả 4 frame cùng thấy một mặt đường, nhưng đám xe kẹt ở mỗi ngày nằm ở vị trí khác nhau.

**Tiền xử lý và lọc (chạy một lần, lưu kết quả):**

1. **Loại frame hỏng:** frame đen, đóng băng (giống hệt frame trước, so bằng hash), nhiễu sọc. Quy tắc gợi ý: độ lệch chuẩn cường độ < 5, hoặc hash trùng với frame trước.
2. **Tách chế độ ngày/đêm hồng ngoại:** frame có độ bão hòa màu trung bình rất thấp là ảnh hồng ngoại. Chỉ ghép frame cùng chế độ vào một nhóm; prior cũng tính riêng theo chế độ.
3. **Phát hiện camera lệch:** ước lượng dịch chuyển giữa frame và prior bằng phase correlation trên **vùng tĩnh ngoài mặt đường** (mask static\_textured, xem IV.3). Lệch > 4 px → căn chỉnh bằng phép dịch/affine; lệch > 16 px hoặc căn chỉnh thất bại → loại frame. Nếu camera bị chỉnh góc vĩnh viễn từ một ngày nào đó, chia dữ liệu thành hai "giai đoạn camera" và tính prior riêng.
4. **Chuẩn hóa màu giữa các ngày:** với mỗi frame, ước lượng hệ số tuyến tính (gain, bias) theo từng kênh màu để khớp vùng tĩnh của frame với vùng tĩnh của prior, bằng bình phương tối thiểu có trọng số robust. Mục đích: chênh lệch ánh sáng giữa các ngày không bị mô hình đẩy sang lớp phương tiện.
5. **Kích thước:** đưa về 448 × 256 (16:9, chia hết cho patch 16 → lưới 28 × 16 token). Thử thêm 640 × 352 trong ablation nếu xe máy ở xa quá nhỏ.

**Chia dữ liệu theo camera:** 70% / 10% / 20% số cụm camera cho train / val / test, trong đó camera cùng một nút giao luôn nằm chung một cụm (IV.3). Tập audit (mục 0.2) cũng chia theo đúng các camera val/test này; **không dùng audit của camera test để chọn siêu tham số hay quyết định dừng vòng tự làm sạch**.

**Lấy mẫu cân bằng:** lấy mẫu đều theo slot giờ (không để giờ thấp điểm chiếm áp đảo) và tăng tỷ trọng slot cao điểm × 2, vì đây là vùng prior sai nhiều và là trọng tâm của paper.

### A.7. Tập bán tổng hợp (có ground truth tuyệt đối)

Đây là tập đánh giá quan trọng nhất vì biết chính xác mặt đường dưới xe và mask xe.

1. **Nền:** chọn frame gần như trống của camera test (detector đếm ≤ 1 xe, thường là 0–5h sáng hoặc thấp điểm trưa); người duyệt nhanh để chắc chắn không còn xe. Mục tiêu khoảng 3–5 frame mỗi camera test.
2. **Phương tiện:** lấy crop xe có mask từ tập audit (SAM + duyệt tay), gom thành thư viện vài nghìn xe máy, ô tô, xe buýt.
3. **Đặt xe đúng phối cảnh:** với mỗi camera, ước lượng quan hệ "tọa độ y → chiều cao xe trung bình" từ các box detector trên frame thật của camera đó (hồi quy tuyến tính). Khi dán xe ở vị trí y, scale theo quan hệ này; chỉ dán trong road mask.
4. **Hòa trộn:** Poisson blending, thêm bóng đổ giả nhẹ (elip mờ dưới xe) và điều chỉnh màu xe theo độ sáng nền. Tránh để ranh giới dán quá dễ nhận ra.
5. **Mật độ:** tạo 3 mức phủ 5–15%, 15–35%, 35–60% diện tích mặt đường, mỗi mức khoảng 700 ảnh → tổng khoảng 2.000 ảnh.

Hạn chế cần nêu trong paper: ảnh dán không giống hệt ảnh thật (thiếu che khuất phức tạp, phản chiếu). Vì vậy tập bán tổng hợp chỉ là **một trong ba** cách đánh giá, cùng với tập audit thật và frame trống thật.

### A.8. Kiến trúc chi tiết

Khung tổng quát: một encoder ViT dùng chung, một decoder đa tỉ lệ kiểu DPT, và 4 đầu ra nhẹ. **Background prior không phải là đầu vào** ở chế độ chính — nó chỉ xuất hiện trong hàm loss. Nhờ vậy lúc suy luận chỉ cần một frame.

| Thành phần | Cấu hình đề xuất | Kích thước tensor (H × W = 256 × 448) |
| --- | --- | --- |
| Đầu vào | Frame origin RGB, chuẩn hóa ImageNet | 3 × 256 × 448 |
| Encoder | DINOv3 ViT-B/16; đóng băng 8 block đầu, fine-tune 4 block cuối (hoặc LoRA rank 16 cho toàn bộ) | token 768 chiều, lưới 16 × 28 |
| Lấy feature | Token từ block 3, 6, 9, 12 (giống DPT) | 4 × (768 × 16 × 28) |
| Reassemble | Chiếu 1×1 về 256 kênh, đưa về tỉ lệ 1/4, 1/8, 1/16, 1/32 của ảnh | 256 × 64 × 112 … 256 × 8 × 14 |
| Fusion | Khối fusion kiểu DPT (residual conv + upsample) từ thô đến mịn | 256 × 64 × 112 |
| Upsample cuối | 2 lần upsample + conv 3×3 | 64 × 256 × 448 |
| Đầu M\_α | Conv 3×3 → 1 kênh → sigmoid | 1 × 256 × 448 |
| Đầu F | Conv 3×3 → 3 kênh → sigmoid | 3 × 256 × 448 |
| Đầu B̂ | Conv 3×3 → 3 kênh → sigmoid | 3 × 256 × 448 |
| Đầu σ | Conv 3×3 → 1 kênh → s = log σ, kẹp trong \[log 0.01, log 0.5\] | 1 × 256 × 448 |

Tổng tham số khoảng 86M (ViT-B) + khoảng 10–15M (decoder). Phương án nhẹ để thử nhanh: ViT-S/16 (khoảng 22M).

**Vài lựa chọn thiết kế và lý do:**

- **Encoder DINO** thay vì CNN từ đầu: B̂ phải "đoán" mặt đường bên dưới xe (inpainting), cần ngữ cảnh toàn cục — đây là điểm mạnh của ViT. Feature DINO cũng đã phân biệt tốt vật thể và nền.
- **Một decoder chung, 4 đầu riêng:** các đầu ra phụ thuộc chặt vào nhau (M\_α quyết định chỗ nào B̂ phải tự đoán), dùng chung decoder giúp nhất quán. Thử biến thể 2 decoder (một cho M\_α, F; một cho B̂, σ) trong ablation.
- **Đầu σ chỉ nhận gradient từ L\_prior:** σ dự đoán "prior sai ở đâu", không nên bị các loss khác kéo lệch. Thực hiện bằng cách stop-gradient σ trong mọi loss trừ L\_prior.
- **σ thuộc về prior, không thuộc về ảnh:** σ trên lý thuyết là thuộc tính của cặp (camera, slot). Vì vậy ngoài σ theo từng frame, ta tính thêm σ̄\_{c,s} = median của σ trên nhiều frame — chính là bản đồ tin cậy của background (xem A.12).

**Hai chế độ suy luận:**

| Chế độ | Đầu vào | Dùng khi | Ghi chú |
| --- | --- | --- | --- |
| (a) Chỉ frame — chế độ chính | 1 frame | Camera mới chưa có background, hoặc background quá tệ | Đóng góp chính của paper |
| (b) Frame + prior | 1 frame + B\_{c,s} đưa qua một CNN nhỏ, cộng vào decoder | Camera đã có background | Train với prior-dropout 50% để một mô hình chạy được cả hai chế độ |

Báo cáo cả hai trong paper: chế độ (b) thường tốt hơn khi prior tốt, nhưng phải chỉ ra nó **không sụp** khi prior xấu (BDB).

&#91;embedded content: Kiến trúc, các loss và vòng tự làm sạch background\]

Prior B chỉ tác động qua L\_prior, nơi σ cho phép mô hình bỏ qua chỗ prior sai; sau mỗi vòng huấn luyện, B̂ và σ được gộp lại thành prior mới cho vòng sau.

### A.9. Hàm loss chi tiết

Với mỗi frame k trong nhóm, ảnh tái tạo là Î^(k) = M^(k) ⊙ F^(k) + (1 − M^(k)) ⊙ B̂^(k). Tổng loss:

```latex
\mathcal{L} = \mathcal{L}_{\text{rec}} + \lambda_p \mathcal{L}_{\text{prior}} + \lambda_{sh} \mathcal{L}_{\text{shared}} + \lambda_{ex} \mathcal{L}_{\text{excl}} + \lambda_{sp} \mathcal{L}_{\text{sparse}} + \lambda_{tv} \mathcal{L}_{\text{tv}} + \lambda_{bin} \mathcal{L}_{\text{bin}}
```

**(1) Tái tạo** — ép ba lớp ghép lại đúng bằng ảnh gốc. Dùng tổ hợp SSIM + L1 quen thuộc (trọng số 0.85 / 0.15 như trong các công trình ước lượng độ sâu tự giám sát):

```latex
\mathcal{L}_{\text{rec}} = \frac{1}{K}\sum_{k} \Big[ 0.85 \cdot \frac{1 - \text{SSIM}(I^{(k)}, \hat{I}^{(k)})}{2} + 0.15 \cdot \| I^{(k)} - \hat{I}^{(k)} \|_1 \Big]
```

**(2) Prior mềm có độ bất định** — log-likelihood Laplace (Kendall & Gal, NeurIPS 2017). Chỗ nào prior sai (ghost), mô hình được phép tăng σ để không phải chép theo, nhưng phải trả giá qua số hạng log σ:

```latex
\mathcal{L}_{\text{prior}} = \frac{1}{K\,HW}\sum_{k}\sum_{u,v} \left[ \frac{\big| \hat{B}^{(k)}(u,v) - B_{c,s}(u,v) \big|_1}{\sigma^{(k)}(u,v)} + \log \sigma^{(k)}(u,v) \right]
```

**(3) Nền dùng chung giữa các ngày** — mặt đường của K frame phải giống nhau. Dùng hàm Charbonnier ρ(x) = √(x² + ε²) và median theo k (stop-gradient trên median để ổn định):

```latex
\mathcal{L}_{\text{shared}} = \frac{1}{K\,HW}\sum_{k}\sum_{u,v} \rho\Big( \hat{B}^{(k)}(u,v) - \operatorname{sg}\big[\operatorname{med}_{j}\, \hat{B}^{(j)}(u,v)\big] \Big)
```

**(4) Loại trừ** — chỗ nào nền dùng chung đã giải thích được pixel thì không được gọi là xe. Loss này thay cho phạt diện tích thuần túy, vốn phạt oan khi đường kẹt thật (xe phủ 50% mặt đường là đúng chứ không phải sai):

```latex
\mathcal{L}_{\text{excl}} = \frac{1}{K\,HW}\sum_{k}\sum_{u,v} M^{(k)}(u,v) \cdot \exp\!\left( - \frac{\big| I^{(k)}(u,v) - \operatorname{sg}[\hat{B}^{(k)}(u,v)] \big|_1}{\tau} \right)
```

Với τ khoảng 0.05 (ảnh trong \[0, 1\]): pixel gần giống nền → trọng số gần 1 → phạt mạnh nếu M > 0.

**(5) Thưa toàn cục (nhỏ)** — L\_sparse = trung bình M, chỉ để tránh M nhiễu lan tràn. **(6) Mượt** — L\_tv = total variation của M, làm mịn biên. **(7) Nhị phân hóa** — L\_bin = trung bình M(1 − M), đẩy M về 0 hoặc 1, chỉ bật sau warmup.

### A.10. Chống nghiệm suy biến

| Nghiệm suy biến | Hiện tượng | Thứ chặn nó |
| --- | --- | --- |
| M = 0, B̂ = I | Xe bị coi là nền, tái tạo hoàn hảo | L\_shared (xe khác nhau giữa các ngày nên B̂ chứa xe sẽ không khớp nhau) + L\_prior ở vùng prior đúng |
| M = 1, F = I | Mọi thứ là xe | L\_excl + L\_sparse |
| σ lớn ở mọi nơi | Bỏ qua prior hoàn toàn | Số hạng log σ + kẹp σ ≤ 0.5; theo dõi trung bình σ trong lúc train |
| B̂ chứa xe đứng yên ở cả K ngày | Xe đậu cố định (ví dụ bãi giữ xe) | Không phân biệt được — nêu là giới hạn; thực tế đó gần như là "nền" |
| Đổi màu xe sang nền | F hấp thụ chênh lệch ánh sáng | Chuẩn hóa màu ở A.6 + L\_excl |

**Theo dõi trong lúc train (ghi log mỗi epoch):** trung bình M trên slot thấp điểm và cao điểm (cao điểm phải lớn hơn rõ rệt); trung bình σ; IoU của M với mask audit-val; ghost rate của median B̂ trên audit-val.

### A.11. Siêu tham số khởi điểm và lịch huấn luyện

| Tham số | Giá trị khởi điểm | Ghi chú |
| --- | --- | --- |
| λ\_p, λ\_sh | 1.0, 1.0 | Chỉnh trên audit-val |
| λ\_ex, λ\_sp, λ\_tv | 0.5, 0.01, 0.01 |  |
| λ\_bin | 0 → 0.05 | Bật từ epoch 10 |
| τ (loss loại trừ) | 0.05 |  |
| K, batch | K = 4, 8 nhóm/batch (32 frame) | Gradient checkpointing nếu thiếu VRAM |
| Optimizer | AdamW, lr 1e-4 decoder, 2e-5 encoder, weight decay 0.05 | Cosine, warmup 2 epoch |
| Độ dài | 30 epoch hoặc khoảng 100k bước | bf16 AMP |
| Phần cứng | 1–2 GPU 24 GB | ViT-S để thử nhanh trên 1 GPU |

**Lịch 3 pha:**

1. **Epoch 0–4 (khởi động):** cố định σ = 0.1 ở mọi nơi (chưa học σ), λ\_p = 2.0. Mô hình học khung phân rã từ prior trước.
2. **Epoch 5–9:** mở khóa đầu σ, giảm λ\_p về 1.0. Mô hình bắt đầu "nghi ngờ" prior ở chỗ L\_shared và L\_rec mâu thuẫn với nó.
3. **Epoch 10 trở đi:** bật L\_bin, giữ nguyên các loss còn lại.

Các giá trị trên là điểm xuất phát hợp lý, không phải kết quả đã kiểm chứng — cần quét trên audit-val.

### A.12. Vòng tự làm sạch background

Sau mỗi lần huấn luyện, dùng mô hình để tạo background mới cho từng (camera, slot), rồi huấn luyện lại với prior mới. Background tốt dần qua 2–3 vòng. Điểm quan trọng: khi gộp, **chỉ tin B̂ ở chỗ mô hình chắc chắn và không có xe**, vì B̂ dưới xe là do mô hình tự đoán.

```python
B = load_median_backgrounds()          # B[c][s]: prior vòng 0
model = init_model()

for r in range(R_MAX):                 # R_MAX = 3
    model = train(model, groups, prior=B)            # A.9–A.11

    B_new, Sigma = {}, {}
    for c, s in camera_slots:
        frames = sample_frames(c, s, n=50, distinct_days=True)
        M, _, B_hat, sigma = model(frames)           # n x H x W (x 3)

        # tin B_hat khi: không có xe (1 - M) và mô hình chắc chắn (sigma nhỏ)
        w = (1 - M) / (sigma ** 2 + 1e-4)            # n x H x W
        B_new[c][s] = weighted_median(B_hat, w, axis=0)
        Sigma[c][s] = median(sigma, axis=0)          # bản đồ tin cậy

    g_old = ghost_rate(B,     audit_val)
    g_new = ghost_rate(B_new, audit_val)             # chỉ camera val!
    if g_old - g_new < 0.01:                         # cải thiện < 1 điểm %
        break
    B = B_new
```

**Các chi tiết cần chú ý:**

- **Không rò rỉ:** tiêu chí dừng chỉ dùng audit của camera **val**. Ghost rate trên camera test chỉ báo cáo ở cuối.
- **Khởi tạo lại hay huấn luyện tiếp:** thử cả hai trong ablation. Huấn luyện tiếp (warm-start) nhanh hơn nhưng có thể giữ lại lỗi của vòng trước; khởi tạo lại sạch hơn nhưng tốn compute.
- **Nguy cơ tự khuếch đại lỗi:** nếu mô hình vòng r tin nhầm một ghost, vòng r+1 sẽ học theo nó mạnh hơn. Theo dõi ghost rate theo từng vòng và theo từng nhóm điều kiện (cao điểm, đêm); nếu một nhóm xấu đi thì dừng sớm cho nhóm đó.
- **Chế độ ngày/đêm và giai đoạn camera:** tính B\_new riêng cho từng chế độ và từng giai đoạn camera (A.6).

### A.13. Bản đồ tin cậy — sản phẩm phụ dùng cho các hướng khác

Sigma\[c\]\[s\] là bản đồ cho biết **chỗ nào trên background của camera c, slot s là không đáng tin**. Đây chính là thứ bạn đang thiếu: bằng chứng định lượng, theo từng pixel, về độ tin cậy của background. Nó dùng lại được ngay:

| Dùng ở đâu | Cách dùng |
| --- | --- |
| H1 (FAM) | Giảm trọng số w\_p ở patch có σ̄ lớn — patch "nghi ngờ" không được ưu tiên che |
| H3/H4 | Nhân Δ với (1 − σ̄ chuẩn hóa) trước khi dùng; tính ρ\_proxy chỉ trên pixel đáng tin |
| Hướng B | LF2 (nguồn background) tự bỏ qua khi tỷ lệ pixel không đáng tin trong road mask vượt ngưỡng |
| Hướng E | Dùng σ̄ làm trọng số khi tính trung bình feature của background |
| Hướng F | Công bố kèm mỗi ảnh background như một lớp metadata |

Trong paper, nên có một hình: cột 1 là background median có ghost, cột 2 là σ̄ (sáng ở đúng chỗ ghost), cột 3 là background sau vòng tự làm sạch, cột 4 là ghost do người gán. Một hình như vậy thuyết phục reviewer hơn mọi bảng số.

### A.14. Thiết kế thí nghiệm

**Năm tập đánh giá** (tất cả chỉ trên camera test, trừ khi ghi khác):

| Tập | Quy mô ước tính | Có ground truth gì | Trả lời câu hỏi |
| --- | --- | --- | --- |
| Audit thật (mục 0.2) | khoảng 70 background + 100 frame có mask xe (phần camera test) | Polygon ghost trên background; mask xe trên frame | RQ1, RQ2, RQ3 |
| Ghost cài sẵn | Mọi camera test, 5–30% diện tích mặt đường | Vị trí chính xác của ghost do ta chèn | RQ1 (có kiểm soát) |
| Bán tổng hợp (A.7) | khoảng 2.000 ảnh | Mask xe + mặt đường dưới xe | RQ3 |
| Frame trống thật | khoảng 300–500 frame | Mặt đường thật, không xe | RQ2, RQ3 (xóa xe) |
| Gold chiếm dụng (H4) | 800–1.000 frame | Chiếm dụng mặt đường do người gán | RQ4 |

**Thí nghiệm ghost cài sẵn — cách làm cụ thể:** với mỗi camera test, chèn crop xe vào prior B\_{c,s} ở vị trí ngẫu nhiên trong road mask (ghi lại mask chèn G). Huấn luyện bình thường với prior đã bị làm hỏng. Sau đó tính AUROC của σ̄\_{c,s} trong road mask với nhãn G. Vì G biết chính xác, đây là bằng chứng sạch nhất cho RQ1. Lặp lại ở 3 mức 5%, 15%, 30%.

**Metric theo câu hỏi:**

| RQ | Metric chính | Metric phụ |
| --- | --- | --- |
| RQ1 | AUROC, AP của σ̄ so với ghost (pixel, trong road mask) | Tương quan giữa σ̄ trung bình và ghost rate theo slot |
| RQ2 | Ghost rate (% diện tích mặt đường) qua từng vòng | PSNR, LPIPS giữa background và frame trống thật cùng slot |
| RQ3 — mask | IoU, F1 của M\_α (ngưỡng 0.5) | Boundary F-score; IoU theo loại xe (xe máy, ô tô) |
| RQ3 — xóa xe | PSNR, LPIPS **chỉ trong vùng bị xe che** (ảnh bán tổng hợp) | FID giữa B̂ và tập frame trống thật |
| RQ4 | MAE chiếm dụng so với gold | Đường cong BDB (mục I.1) |

**Mọi kết quả chia theo điều kiện:** ngày / đêm; mật độ thưa / vừa / đông; khô / mưa. Đây là chỗ phương pháp phải thắng rõ nhất: cao điểm và đêm, nơi median sai nhiều nhất.

**Baseline:**

| Nhóm | Phương pháp | Ghi chú |
| --- | --- | --- |
| Cổ điển | Ngưỡng Δ trên background median (τ tốt nhất chọn trên val); MOG2, KNN của OpenCV chạy trên chuỗi frame | Cho thấy vì sao cần học |
| Hai giai đoạn có giám sát | Detector có mask (YOLOv8-seg hoặc Mask R-CNN huấn luyện trên COCO) + inpainting LaMa (Suvorov et al., WACV 2022) | Baseline mạnh nhất cho RQ3, vì dùng nhãn COCO |
| Phân rã lớp | Một phương pháp kiểu Omnimatte trên khoảng 10 camera | Tối ưu theo từng video, chậm — chỉ chạy tập con |
| Tự giám sát | H2 gốc (L1 cứng với background) | Ablation quan trọng nhất về mặt câu chuyện |

Nếu baseline hai giai đoạn (detector + LaMa) thắng về chất lượng mask ban ngày, điều đó chấp nhận được; luận điểm của paper là phương pháp không cần nhãn, tốt hơn ở cao điểm/ban đêm (nơi detector COCO yếu), và tạo ra bản đồ tin cậy mà baseline không có.

**Ablation:**

| Biến thể | Kiểm tra điều gì |
| --- | --- |
| Bỏ σ (L1 cứng với prior) | Giá trị của prior mềm |
| Bỏ L\_shared | Giá trị của ràng buộc nhiều frame |
| K frame cùng ngày vs khác ngày | Luận điểm "khác ngày" xử lý xe đứng yên |
| K = 2, 4, 8 | Độ nhạy theo K |
| L\_excl vs chỉ L\_sparse | Loss loại trừ có tránh phạt oan khi kẹt xe không |
| Số vòng tự làm sạch R = 0, 1, 2, 3; warm-start vs khởi tạo lại | Đóng góp của vòng tự làm sạch |
| ViT-S vs ViT-B; đóng băng vs fine-tune | Ảnh hưởng của backbone |
| Chế độ (a) vs (b) | Giá trị khi có thêm prior lúc suy luận |

**Thống kê:** chạy 3 seed, báo cáo mean ± std; khoảng tin cậy 95% bằng bootstrap **theo camera** (không theo frame, vì frame cùng camera tương quan mạnh). So sánh cặp với baseline chính bằng kiểm định Wilcoxon trên các camera.

### A.15. Rủi ro và cách giảm

| Rủi ro | Dấu hiệu sớm | Cách giảm |
| --- | --- | --- |
| Camera xoay giữa các ngày làm L\_shared sai | Δ trên vùng tĩnh tăng đột ngột từ một ngày | Căn chỉnh + chia giai đoạn camera (A.6) |
| Ánh sáng khác giữa các ngày bị đẩy sang M\_α | M\_α cao trên mặt đường trống | Chuẩn hóa màu (A.6); thử L\_shared trên feature DINO thay vì pixel |
| σ "lười", lớn ở mọi nơi | Trung bình σ tăng dần, AUROC trên val không tăng | Kẹp σ; tăng λ\_p; khóa σ lâu hơn ở pha 1 |
| Vòng tự làm sạch khuếch đại lỗi | Ghost rate trên val tăng ở một nhóm điều kiện | Dừng sớm theo nhóm; khởi tạo lại thay vì warm-start |
| Lấy mẫu quá thưa nên ít frame mỗi slot | Ít hơn K ngày có frame hợp lệ cho một (camera, slot) | Gộp slot liền kề (±1h) cho các camera thiếu dữ liệu |
| Detector + LaMa thắng rõ ở mọi điều kiện | Kết quả RQ3 trên val | Đổi trọng tâm paper sang RQ1, RQ2 và bản đồ tin cậy; dùng detector + LaMa như nguồn bổ sung |

### A.16. Dàn ý paper

1. **Introduction:** camera đô thị nhiều nhưng không nhãn; background median miễn phí nhưng sai đúng lúc cần; hình minh họa ghost ở giờ cao điểm; 3–4 đóng góp.
2. **Related work:** background subtraction cổ điển và học sâu; phân rã lớp (Omnimatte, Layered Neural Atlases); học từ nhãn nhiễu và độ bất định heteroscedastic; thị giác giao thông hỗn hợp, xe máy dày đặc.
3. **Problem formulation:** mục A.3.
4. **Method:** kiến trúc (A.8), loss (A.9), chống suy biến (A.10), vòng tự làm sạch (A.12).
5. **Data and evaluation protocol:** dữ liệu và lọc (A.6), tập audit, tập bán tổng hợp (A.7), ghost cài sẵn.
6. **Experiments:** RQ1 → RQ4, ablation, BDB, phân tích theo điều kiện.
7. **Discussion and limitations:** xe đứng yên cố định, ảnh bán tổng hợp chưa giống thật, phụ thuộc tần suất lấy mẫu.

### A.17. Lịch 6 tháng

1. **Tháng 1:** pipeline dữ liệu (lọc, căn chỉnh, chuẩn hóa màu, nhóm K frame); hoàn thành tập audit và road mask; chạy lại H2 gốc làm baseline.
2. **Tháng 2:** xây tập bán tổng hợp và tập frame trống; chạy baseline cổ điển và detector + LaMa.
3. **Tháng 3:** thêm L\_shared và L\_excl; thêm đầu σ; chạy thí nghiệm ghost cài sẵn trên val để kiểm tra RQ1 sớm.
4. **Tháng 4:** vòng tự làm sạch; chế độ (b); bản đồ tin cậy.
5. **Tháng 5:** toàn bộ ablation, BDB, RQ4 với gold set; Omnimatte trên tập con.
6. **Tháng 6:** viết paper, làm hình, gửi tạp chí.

**Điểm quyết định cuối tháng 3:** nếu AUROC của σ với ghost cài sẵn không vượt rõ ngưỡng ngẫu nhiên (ví dụ < 0.7), cần xem lại thiết kế trước khi đầu tư tiếp.

**Venue:** IEEE TIP, Pattern Recognition, IEEE TCSVT, IEEE T-ITS.

## II.A2. Hướng 2 phiên bản không cần ảnh nền — phân rã cảnh tự học từ chuỗi ảnh

Tên gợi ý: *Background-Free Traffic Scene Decomposition from Sparse Fixed-Camera Image Streams*. II.A vẫn cần ảnh nền median (dù chỉ làm prior mềm). II.A2 bỏ hẳn: nền của mỗi camera được **học ra** từ chính chuỗi ảnh nhiều ngày, coi phương tiện là phần "ngoại lai" không lặp lại. Nếu A2 thắng A ở cổng quyết định (A2.9), A2 trở thành phương pháp chính của paper Hướng 2, còn A là biến thể dùng khi có sẵn prior.

### A2.1. Bối cảnh và rà soát code hiện tại

**Vì sao bỏ được ảnh nền.** Với camera cố định, nền của frame t là cùng một cảnh tĩnh, chỉ khác ánh sáng, bóng đổ, độ ướt. Các biến đổi này có **ít bậc tự do** — có thể mô tả bằng vài hệ số cho mỗi frame. Phương tiện thì khác nhau ở mọi frame. Vì vậy nếu buộc nền phải nằm trong một không gian ít chiều dành riêng cho camera, xe sẽ không thể "chui" vào nền. Đây chính là giả định nền tảng của background subtraction dạng tái tạo: các frame nền nằm trên một đa tạp ít chiều — và cũng là cách NeRF-W xử lý ảnh du lịch, với embedding ngoại hình cho ánh sáng và embedding riêng cho vật tạm thời.

**Rà soát README `direction2_scene_decomposition/` hiện tại:**

| # | Điểm trong README | Vấn đề | Sửa trong A2 |
| --- | --- | --- | --- |
| 1 | "Dùng ảnh background thật làm **ground-truth** giám sát nhánh nền" (λ\_bg = 1.5) | Median sai đúng ở giờ kẹt; mô hình học luôn cả ghost. Gọi là "ground-truth" sẽ bị reviewer bác | Bỏ hẳn L\_bg; nền học từ chuỗi ảnh (A2.5) |
| 2 | Gọi là "Unsupervised Road Inpainting" | Đang giám sát bằng ảnh nền tính sẵn → đúng ra là giám sát yếu bằng nhãn giả nhiễu | A2 mới đúng nghĩa không giám sát |
| 3 | `--img_size 256` (ảnh vuông) | Frame 16:9 bị méo; xe máy ở xa vốn đã nhỏ | 448 × 256, giữ tỷ lệ |
| 4 | `--freeze_backbone True` | Nhánh nền phải "đoán" mặt đường dưới xe — cần biểu diễn tốt hơn; đóng băng toàn bộ hạn chế điều này | Fine-tune 4 block cuối hoặc LoRA (ablation) |
| 5 | `--match_strategy route_hourly` ghép 1 frame với 1 ảnh nền | Không khai thác chuỗi nhiều ngày | Sampler theo nhóm: cùng camera, nhiều ngày (A2.7) |
| 6 | DataParallel, "best model" theo loss huấn luyện | DataParallel chậm; chọn checkpoint theo loss không phản ánh chất lượng tách | DDP; chọn checkpoint theo IoU trên audit-val |
| 7 | `density_mask.png` | Đây là mask alpha, không phải mật độ | Đổi tên `alpha_mask.png` |
| 8 | Độ mới ⭐⭐⭐⭐⭐, nhắm CVPR/ECCV/NeurIPS | Đó là hội nghị, không phải tạp chí Q1; đã có Omnimatte, AE-NE và các phương pháp robust PCA học sâu | Định vị lại (A2.2) |

### A2.2. Công trình liên quan và khoảng trống

Tra cứu tháng 10/2026; mục có link đã xác nhận qua trang bài báo hoặc trang chính thức.

| Nhánh | Công trình | Ý chính | Ta học gì / khác gì |
| --- | --- | --- | --- |
| Cảnh tĩnh + vật tạm thời + ánh sáng | NeRF-W (Martin-Brualla et al., CVPR 2021) — mô tả qua [khảo sát NeRF](https://arxiv.org/pdf/2501.13104) | Embedding ngoại hình theo từng ảnh cho ánh sáng; embedding riêng cho vật tạm thời | Mượn ý "mã ánh sáng theo frame" — nhưng camera cố định nên không cần 3D |
|  | [RobustNeRF (Sabour et al., CVPR 2023)](https://arxiv.org/pdf/2302.00833) | Coi vật tạm thời là ngoại lai; huấn luyện như IRLS với trimmed least squares và giả định ngoại lai liền khối trong không gian | Dùng trực tiếp làm loss giai đoạn 1; chính tác giả nêu giới hạn kém hiệu quả thống kê trên dữ liệu sạch |
|  | [NeRF On-the-go (Ren et al., CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.pdf) | Dự đoán độ bất định theo pixel để lọc vật tạm thời; ghi nhận NeRF-W, RobustNeRF giảm mạnh khi tỷ lệ che khuất cao | Đúng tình huống giờ kẹt — lý do cần thêm TAM và lấy mẫu nhiều ngày |
| Nền tái tạo, không giám sát | [AE-NE (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf) | Autoencoder học nền như đa tạp ít chiều cho **từng video**; ước lượng nhiễu nền; trọng số bootstrapping | Baseline chính; ta học **một mô hình cho cả thành phố** + chế độ một frame |
| Robust PCA học sâu | [Deep-unfolded RPCA (Luong et al., EUSIPCO 2020)](https://arxiv.org/pdf/2010.00929); [masked RPCA unfolding (Joukovsky et al.)](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/Interpretable_Neural_Networks_for_Video_Separation_Deep_Unfolding_RPCA_With_Foreground_Masking.pdf); CORONA | Mở vòng lặp RPCA thành mạng; bản masked RPCA nhân mask thưa với thành phần hạng thấp thay vì cộng | Cùng giả định hạng thấp + thưa; chủ yếu thử trên video ngắn, dữ liệu tổng hợp |
| Phân rã lớp từ tập ảnh | [DTI-Sprites (Monnier et al., ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Monnier_Unsupervised_Layered_Image_Decomposition_Into_Object_Prototypes_ICCV_2021_paper.pdf) | Học nguyên mẫu đối tượng và nền như ảnh học được, ghép lớp có che khuất | Ý "nền là tham số học được" — ta học nền riêng cho mỗi camera |
| Phân rã video thành lớp | Omnimatte (CVPR 2021); [Generative Omnimatte (CVPR 2025)](https://cvpr.thecvf.com/virtual/2025/poster/34367); OmnimatteZero (SIGGRAPH Asia 2025) | Phân rã video thành lớp RGBA; bản 2025 dùng video diffusion và **cần mask đối tượng đầu vào** | Cần video liên tục và mask — không hợp dữ liệu thưa, không nhãn |
| Phát hiện rồi học | [CutLER (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Cut_and_Learn_for_Unsupervised_Object_Detection_and_Instance_Segmentation_CVPR_2023_paper.pdf) | Tạo mask thô không giám sát rồi huấn luyện detector với loss bỏ qua vùng bị sót, tự huấn luyện nhiều vòng | Mượn khuôn "khớp rồi chưng cất" và DropLoss cho giai đoạn 2 |

**Khoảng trống:** chưa thấy công trình nào (trong phạm vi tra cứu) phân rã cảnh giao thông **không cần ảnh nền, không cần video liên tục, không cần mask**, học từ **ảnh chụp thưa nhiều ngày** của hàng trăm camera cố định, đồng thời cho ra mô hình **một frame** dùng được cho camera mới. Tình huống che khuất nặng (giờ kẹt) — điểm yếu đã được ghi nhận của các phương pháp robust — là trọng tâm đánh giá.

### A2.3. Mô hình hóa bài toán và điều kiện định danh

**Mô hình tạo ảnh.** Frame t của camera c:

```latex
I_t = M_t \odot F_t + (1 - M_t) \odot B_t, \qquad
B_t = \operatorname{clip}_{[0,1]}\Big( E_{c,0} + \sum_{j=1}^{J} \ell_{t,j}\, E_{c,j} \Big)
```

- E\_{c,0}: "cảnh tĩnh trung bình" của camera c — một ảnh học được, độ phân giải đầy đủ 256 × 448.
- E\_{c,1..J}: J ảnh cơ sở mô tả biến đổi của nền (nắng/râm, ướt/khô, bóng cây) — học được, độ phân giải một nửa (128 × 224) rồi nội suy lên, ràng buộc trơn.
- ℓ\_t ∈ ℝ^J: **mã ánh sáng** của frame t, vài con số (mặc định J = 4).
- M\_t, F\_t: mask và lớp phương tiện.

Ví dụ: camera 125 có E\_0 là mặt đường lúc trưa nắng; E\_1 làm mọi thứ tối đi (chiều tà); E\_2 làm mặt đường sẫm và bóng (ướt mưa); E\_3 là vệt bóng cây di chuyển theo giờ. Frame 17:20 ngày mưa ≈ E\_0 + 0.6·E\_1 + 0.8·E\_2 + 0.3·E\_3. Một chiếc xe máy đỏ ở góc trái **không thể** biểu diễn bằng tổ hợp này, vì nếu E\_j chứa chiếc xe đó thì mọi frame khác dùng E\_j cũng bị "dính" xe.

Dung lượng tham số mỗi (camera, chế độ ngày/IR): khoảng 0.69 triệu số thực (2.8 MB fp32) với J = 4 → toàn hệ thống khoảng 3.3 GB, lưu trên đĩa, nạp theo lô camera.

**Điều kiện để bài toán định danh được** — mỗi điều kiện có một thí nghiệm kiểm chứng:

| # | Điều kiện | Ý nghĩa | Kiểm chứng |
| --- | --- | --- | --- |
| C1 | Nền có hạng thấp: J nhỏ, các ảnh cơ sở trơn | Xe (chi tiết cao tần, vị trí thay đổi) không biểu diễn được bằng nền | A2-AB1: quét J ∈ {0, 1, 2, 4, 8}, đo tỷ lệ "xe dính vào nền" |
| C2 | Tỷ lệ chiếm dụng mỗi pixel o\_c(p) — phần frame mà pixel p bị xe che — nhỏ hơn **điểm gãy** κ của bộ ước lượng robust | Giống median: nếu pixel bị che hơn một nửa thời gian thì median hỏng | A2-M2: đường cong sai số nền theo o\_c(p) trên dữ liệu bán tổng hợp |
| C3 | Mã ánh sáng ℓ\_t không mang thông tin xe | Nếu ℓ\_t tự do, nó có thể "vẽ" một chiếc xe buýt to vào nền | Ở giai đoạn 2, ℓ\_t tính chỉ từ feature của vùng tĩnh; A2-AB4 kiểm tra rò rỉ |
| C4 | Camera không dịch chuyển trong một cam\_epoch | Cùng vị trí ảnh = cùng điểm cảnh | Căn chỉnh IV.3; A2-R1 bỏ căn chỉnh |

**Điều kiện C2 là giới hạn cơ bản — phải nói thẳng trong paper.** Ta không chữa được nó, nhưng làm giảm o\_c(p) và đo được nó:

- Lấy mẫu frame trên **mọi giờ của nhiều ngày** (kể cả đêm, sáng sớm) thay vì một slot giờ như median theo slot → o\_c(p) giảm mạnh ở hầu hết pixel.
- Ước lượng o\_c(p) bằng bản đồ hoạt động A\_c(p) của TAM (G.4) → sinh **bản đồ định danh** cho mỗi camera: pixel có A\_c(p) > κ được đánh dấu "không định danh được", báo cáo riêng, không đưa vào tuyên bố chính.
- TAM bổ sung cho phần dư màu: một chiếc ô tô **xám** trên mặt đường xám có phần dư pixel nhỏ nên loss robust không loại được, nhưng feature DINO vẫn khác mặt đường → TAM bắt được. Đây là lý do kết hợp hai tín hiệu.

**Hai chế độ sử dụng:**

| Chế độ | Cần gì | Dùng khi |
| --- | --- | --- |
| T — khớp theo camera | Lịch sử ảnh **không nhãn** của camera (vài ngày) | Camera đã lắp — đúng thực tế vận hành, vì camera cố định luôn có lịch sử |
| S — một frame | Chỉ một frame | Camera mới, chưa có lịch sử; hoặc cần chạy nhanh |

### A2.4. Kiến trúc

&#91;embedded content: Quy trình hai giai đoạn của A2 · khớp theo camera rồi chưng cất\]

**Giai đoạn 1 — khớp nền theo camera (không phải mạng nơ-ron sâu).** Tham số của mỗi camera là E\_{c,0..J} và mã ánh sáng ℓ\_t tự do cho từng frame huấn luyện (kiểu GLO, như embedding ngoại hình của NeRF-W). Tối ưu bằng Adam trên GPU, nhiều camera cùng lúc (mỗi camera độc lập, gom thành tensor theo lô). Mask giai đoạn 1 không do mạng dự đoán mà sinh ra từ trọng số robust (A2.5).

**Giai đoạn 2 — mạng một frame** (nâng cấp `TrafficDecompositionNet` hiện có):

| Thành phần | Cấu hình |
| --- | --- |
| Encoder | DINOv3 ViT-S/16 hoặc ViT-B/16 (hoặc backbone G nếu đã qua cổng), fine-tune 4 block cuối |
| Decoder | DPT đa tỉ lệ như A.8, đầu vào 448 × 256 |
| Đầu M | 1 kênh, sigmoid |
| Đầu F | 3 kênh, sigmoid |
| Đầu B̂ | 3 kênh, sigmoid — nền đầy đủ kể cả dưới xe |
| Đầu σ | 1 kênh — độ bất định của B̂ (đặc biệt vùng bị xe che) |
| Đầu ℓ̂ | Vector J chiều, tính từ **trung bình token patch tĩnh** (π < 0.2 theo TAM) qua MLP 2 tầng — không nhìn vùng xe, để thỏa C3 |

**Suy luận chế độ T** (camera đã có E\_c): với frame mới, giải ℓ\_t bằng bình phương tối thiểu có trọng số trên pixel tĩnh (J = 4 ẩn, 3–5 vòng IRLS, dưới 5 ms), dựng B\_t, rồi mask = hợp nhất của phần dư robust và đầu M của mạng. **Suy luận chế độ S:** chỉ chạy mạng.

### A2.5. Hàm loss — không có ảnh nền ở bất kỳ đâu

#### Giai đoạn 1: khớp nền robust theo camera

Phần dư theo pixel r\_t(p) = trung bình 3 kênh của |I\_t(p) − B\_t(p)|. Trọng số **inlier** cập nhật lại mỗi bước (IRLS), không qua gradient, theo kiểu RobustNeRF — cắt bỏ phần dư lớn nhất và giả định ngoại lai liền khối:

```latex
\bar{r}_t = \operatorname{AvgPool}_{8\times 8}(r_t), \qquad
w^{\text{res}}_t(p) = \operatorname{sigmoid}\!\Big(\frac{Q_t(1-\kappa_t) - \bar{r}_t(p)}{s}\Big), \qquad
w_t(p) = w^{\text{res}}_t(p)\,\big(1 - \pi_t(p)\big)^{\gamma}
```

- Q\_t(1 − κ\_t): phân vị của r̄\_t trong frame t. **Tỷ lệ cắt κ\_t thích ứng theo frame**: κ\_t = clip(trung bình π\_t + 0.05, 0.05, 0.6) — frame kẹt xe cắt nhiều hơn frame vắng. RobustNeRF dùng phân vị cố định; thích ứng theo TAM là một điểm mới cần ablation (A2-AB2).
- π\_t: xác suất tiền cảnh từ TAM (G.4); γ = 1. TAM bắt được xe **cùng màu nền** mà phần dư bỏ sót.
- s = 0.02 (độ mềm).

```latex
\mathcal{L}_1 = \frac{\sum_{t,p} w_t(p)\, \rho\big(I_t(p) - B_t(p)\big)}{\sum_{t,p} w_t(p)} + \lambda_{TV}\sum_{j\ge 1}\operatorname{TV}(E_{c,j}) + \lambda_{\ell}\sum_t \lVert \ell_t \rVert^2 + \lambda_{\perp}\sum_{1\le i<j}\cos^2(E_{c,i}, E_{c,j})
```

ρ là Charbonnier. Mặc định λ\_TV = 0.01, λ\_ℓ = 1e-3, λ\_⊥ = 0.1.

**Khởi tạo không dùng thuật toán trích nền:** E\_{c,0} = trung bình có trọng số (1 − π\_t) của các frame (TAM-weighted mean), E\_{c,j≥1} = nhiễu nhỏ, ℓ\_t = 0. Ablation A2-AB3 so với khởi tạo bằng median và khởi tạo ngẫu nhiên để chứng minh kết quả không phụ thuộc khởi tạo.

**Đầu ra giai đoạn 1** cho mỗi frame huấn luyện: nền B\_t^(1) (đầy đủ cả dưới xe, vì nền là tổ hợp của các ảnh cơ sở), mask M\_t^(1) = 1 − w\_t, độ tin cậy c\_t(p) = |w\_t^res(p) − 0.5| × 2 (gần 0 khi mơ hồ), và bản đồ định danh U\_c(p) = 1 nếu A\_c(p) > 0.5.

#### Giai đoạn 2: chưng cất sang mạng một frame

Î = M ⊙ F + (1 − M) ⊙ B̂.

| Loss | Công thức / cách tính | Vai trò | Trọng số |
| --- | --- | --- | --- |
| L\_rec | 0.85·(1 − SSIM)/2 + 0.15·L1 giữa Î và I | Ghép lại đúng ảnh gốc | 1.0 |
| L\_bgPL | NLL Laplace giữa B̂ và B^(1) với σ học được, chỉ trên pixel có U\_c = 0 | Học nền đầy đủ, kể cả "đoán" mặt đường dưới xe; σ lớn ở chỗ nhãn giả không chắc | 1.0 |
| L\_maskPL | BCE giữa M và M^(1), nhân c\_t; **DropLoss**: bỏ phạt ở pixel mạng dự đoán là xe trong khi M^(1) = 0 nhưng π\_t > 0.5 | Học mask; cho phép mạng tìm ra xe mà giai đoạn 1 bỏ sót (mượn ý CutLER) | 0.5 |
| L\_ℓ | ‖ℓ̂ − ℓ\_t‖² | Chưng cất mã ánh sáng | 0.1 |
| L\_excl, L\_tv, L\_bin | Như A.9 | Mask sạch, không suy biến | 0.5 / 0.01 / 0 → 0.05 |

Không còn L\_shared riêng như ở A: tính nhất quán giữa các ngày đã nằm sẵn trong B^(1), vì mọi frame của camera dùng chung E\_c.

#### Vòng tự cải thiện

Sau giai đoạn 2, thay (1 − π\_t)^γ trong trọng số giai đoạn 1 bằng (1 − M̂\_t) do mạng dự đoán, khớp lại E\_c, sinh nhãn giả mới, huấn luyện tiếp giai đoạn 2. Mặc định 2 vòng; dừng khi IoU trên audit-val tăng dưới 0.5 điểm.

### A2.6. Quy trình hai giai đoạn

```python
# ---------- Giai đoạn 1: khớp nền theo camera (song song nhiều camera trên GPU) ----------
for batch_of_cams in chunk(train_and_eval_cameras, size=32):       # mỗi camera độc lập
    for c in batch_of_cams:
        frames[c] = sample_frames(c, n=600, hours='all', min_days=10, mode=m)
        pi[c]     = tam.posterior(c, frames[c])                         # G.4, không gradient
        E[c]      = init_E(frames[c], pi[c])                           # trung bình có trọng số (1 - pi)
        ell[c]    = zeros(len(frames[c]), J)
    for it in range(2000):                                             # Adam, lr 1e-2
        B = clip(E0 + einsum('tj,jchw->tchw', ell, Ej_up))             # dựng nền mọi frame
        w = robust_weights(frames, B, pi, kappa_from_pi, pool=8).detach()
        loss = weighted_charbonnier(frames, B, w) + reg_TV(Ej) + reg_ell(ell) + reg_orth(Ej)
        loss.backward(); step()
    save_stage1(c, E[c], ell[c], B, M1=1-w, conf, U=activity_prior(c) > 0.5)

# ---------- Giai đoạn 2: chưng cất sang mạng một frame ----------
net = DecompNetA2(backbone='dinov3_vitb16', heads=['M','F','B','sigma','ell'])
for step in range(60_000):
    x, pl = next(loader)                     # frame + nhãn giả giai đoạn 1 (chỉ camera train)
    out = net(x)
    loss = L_rec(out, x) + L_bgPL(out, pl) + 0.5*L_maskPL_dropLoss(out, pl) + 0.1*L_ell(out, pl) + L_mask_regs(out)
    ...

# ---------- Vòng tự cải thiện (R = 2) ----------
for r in range(R):
    M_hat = net.predict_masks(frames)        # thay prior TAM trong trọng số giai đoạn 1
    rerun_stage1(prior=M_hat); retrain_stage2(warm_start=True)
    if delta_iou(audit_val) < 0.005: break
```

**Chi tiết quan trọng:**

| Mục | Quy định | Lý do |
| --- | --- | --- |
| Camera nào được khớp giai đoạn 1 | **Mọi camera**, kể cả val/test — vì giai đoạn 1 không dùng nhãn | Chế độ T đánh giá trên camera test là hợp lệ: chỉ dùng ảnh không nhãn của chính camera đó, giống vận hành thực tế |
| Camera nào dùng nhãn giả để train giai đoạn 2 | **Chỉ camera train** | Chế độ S phải được đánh giá trên camera chưa thấy |
| Lấy mẫu frame giai đoạn 1 | 600 frame mỗi (camera, mode), rải đều mọi giờ của mode, ít nhất 10 ngày | Giảm tỷ lệ chiếm dụng o\_c(p) (điều kiện C2) |
| Ngày IR | Khớp riêng (mode `ir`), E riêng | Ảnh hồng ngoại khác hẳn về màu |
| Camera đổi góc (cam\_epoch mới) | Khớp lại từ đầu cho epoch mới | Điều kiện C4 |
| Tham số giai đoạn 2 | Như A.11 (AdamW, lr 1e-4 decoder / 2e-5 encoder, bf16, 448 × 256) | Thống nhất với II.A |

**Chi phí ước tính (cần đo, G-E1 tương tự):** giai đoạn 1 khoảng vài phút GPU cho mỗi (camera, mode) với 600 frame × 2.000 vòng ở 256 × 448; khoảng 1.200 cặp (camera, mode) → khoảng vài chục giờ GPU cho toàn hệ thống, chạy một lần mỗi vòng. Giai đoạn 2 tương đương huấn luyện II.A.

### A2.7. Spec cho agent — thay đổi so với code hiện tại

**Sửa các file đang có trong `direction2_scene_decomposition/`:**

| File | Hiện tại | Sửa thành |
| --- | --- | --- |
| `dataset.py` | `DecompositionDataset` ghép origin với background theo `route_hourly` | `FrameDataset` đọc `frames.parquet` (IV.2), trả frame 448 × 256 đã căn chỉnh, `cid` (camera, mode, epoch), và nhãn giả giai đoạn 1 nếu có. Giữ class cũ làm baseline H2 gốc, gọi bằng cờ `--legacy_bg` |
| `models.py` | `TrafficDecompositionNet`: DINOv3 đóng băng, 3 nhánh | Thêm decoder DPT, đầu `sigma` và đầu `ell` (gộp token patch tĩnh); tùy chọn `--unfreeze_last 4` |
| `losses.py` | Recon + λ\_bg·L1(nền, ảnh nền) + sparsity + TV | Bỏ hạng λ\_bg; thêm `L_bgPL` (Laplace NLL), `L_maskPL` có DropLoss, `L_ell`, `L_excl`, `L_bin`; recon dùng SSIM + L1 |
| `train.py` | DataParallel, chọn "best" theo loss | DDP (`torchrun`); chọn checkpoint theo IoU trên audit-val; giữ ảnh tiến trình mỗi epoch |
| `infer.py` | Xuất `clean_road`, `vehicles_only`, `density_mask` | Thêm `--mode S` hoặc `--mode T --stage1_dir`; đổi tên `density_mask` → `alpha_mask`; thêm `uncertainty.png` |

**File mới:**

| File | Nội dung |
| --- | --- |
| `scene_fit.py` | Giai đoạn 1: class `SceneBasis`, hàm `robust_weights`, CLI khớp theo lô camera |
| `stage1_store.py` | Lưu/đọc E\_c (npz) và nhãn giả từng frame (fp16, định dạng zarr hoặc npz theo camera) |
| `solve_ell.py` | Suy luận chế độ T: giải ℓ\_t bằng bình phương tối thiểu có trọng số trên pixel tĩnh |
| `selftrain.py` | Điều phối vòng tự cải thiện |
| `synth_occupancy.py` | Dữ liệu bán tổng hợp với **tỷ lệ chiếm dụng kiểm soát được** cho A2-M2 |
| `eval_a2.py` | Mọi metric của A2.8 |

**Giao diện chính:**

```python
class SceneBasis(nn.Module):
    """Tham số nền cho một lô camera.
    E0:  (C, 3, 256, 448)        Ej: (C, J, 3, 128, 224)      ell: (C, N, J)"""
    def forward(self, cam_idx: LongTensor, frame_idx: LongTensor) -> Tensor:   # (n, 3, 256, 448)
        Ej_up = F.interpolate(self.Ej[cam_idx].flatten(1, 2), size=(256, 448), mode='bilinear')
        B = self.E0[cam_idx] + einsum('nj,njchw->nchw', self.ell[cam_idx, frame_idx], Ej_up.view(...))
        return B.clamp(0, 1)

def robust_weights(I, B, pi, pool=8, s=0.02, gamma=1.0) -> Tensor:
    """r = |I - B| trung bình kênh; r_bar = avg_pool(r, pool, stride=1, padding='same');
    kappa_t = clamp(pi.mean((1,2)) + 0.05, 0.05, 0.6); q = quantile(r_bar, 1 - kappa_t) theo frame;
    w = sigmoid((q - r_bar) / s) * (1 - pi) ** gamma. Trả về w (n, H, W), đã detach."""

def solve_ell(I, E0, Ej, w_static, iters=5) -> Tensor:
    """Chế độ T: min_ell sum_p w(p) |I - E0 - sum_j ell_j Ej|^2, IRLS cập nhật w. Trả về (J,)."""
```

**Lệnh chạy:**

```bash
# Giai đoạn 1 cho mọi camera (không nhãn, không ảnh nền)
python DINO/direction2_scene_decomposition/scene_fit.py \
    --cameras all --frames_per_cam 600 --J 4 --iters 2000 \
    --tam_ckpt checkpoints/G/tam_stats.pt --out data/stage1/r0

# Giai đoạn 2 — chỉ dùng nhãn giả của camera train
torchrun --nproc_per_node=2 DINO/direction2_scene_decomposition/train.py \
    --config configs/exp/a2_stage2.yaml --stage1_dir data/stage1/r0

# Suy luận
python DINO/direction2_scene_decomposition/infer.py --mode S --weights ... --input_path output/
python DINO/direction2_scene_decomposition/infer.py --mode T --stage1_dir data/stage1/r0 --weights ... --input_path output/
```

**Nghiệm thu (dùng dữ liệu đồ chơi của G.8, thêm nền biết trước):**

1. Nền kết cấu cố định × 3 yếu tố ánh sáng biết trước + hình chữ nhật phủ 20%: giai đoạn 1 khôi phục nền **dưới các hình chữ nhật** với PSNR > 35 dB và mask IoU > 0.9.
2. Một vị trí bị hình chữ nhật (màu ngẫu nhiên) che trong x% frame, x = 10 → 80: sai số nền thấp khi x nhỏ, tăng vọt quanh x ≈ κ — xác nhận điểm gãy như phân tích C2.
3. Hình chữ nhật cùng màu nền (lệch ±2%): chỉ dùng phần dư thì bỏ sót; thêm prior TAM thì bắt được.
4. `solve_ell` khôi phục ℓ biết trước với sai số < 1e-3.
5. Đầu `ell` không đổi khi thay đổi các patch có π > 0.5 (kiểm tra C3 bằng test).
6. Dataloader giai đoạn 2 không đọc nhãn giả của camera val/test — assert theo `splits.json`.

### A2.8. Kế hoạch thực nghiệm

Dùng chung tập đánh giá với A.14 (audit-test, bán tổng hợp, frame trống thật, gold chiếm dụng) và quy tắc thống kê IV.5.

**Baseline:** Δ trên median theo slot; Δ trên median mọi giờ; MOG2, KNN (OpenCV); SuBSENSE, PAWCS (BGSLibrary); AE-NE (khớp riêng mỗi camera test); RobustNeRF-style (= giai đoạn 1 với κ cố định, không TAM); detector VN + LaMa; H2 gốc (L1 với ảnh nền, cờ `--legacy_bg`); A (prior median có σ, II.A); deep-unfolded RPCA nếu có code công khai (P2).

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| A2-M1 | M | Tách xe và xóa xe tốt đến đâu? | Mọi baseline; A2 chế độ T và S | IoU, F1, boundary F; PSNR/LPIPS vùng che; KID; tách theo cao điểm, đêm, mưa, mật độ | P0 |
| A2-M2 | M | Điểm gãy khi pixel bị che nhiều | `synth_occupancy.py`: một vùng bị xe (màu, loại ngẫu nhiên) che trong x% frame, x = 10, 20, … 80 | Sai số nền tại vùng đó theo x, cho: median theo slot, median mọi giờ, AE-NE, giai đoạn 1 không TAM, giai đoạn 1 đầy đủ — **hình chính** | P0 |
| A2-M3 | M | Chế độ T cần bao nhiêu lịch sử? | Số ngày 1 / 3 / 7 / 14 / 30; số frame 100 / 300 / 600 / 1.200 | IoU, PSNR nền; thời gian khớp | P0 |
| A2-M4 | M | Một frame có bằng khớp theo camera? | Chế độ S trên camera chưa thấy vs chế độ T trên cùng camera | IoU, PSNR, tốc độ | P0 |
| A2-M5 | M | Có ích cho ứng dụng? | Chiếm dụng từ M (trong road mask) | MAE so với gold (H4) | P1 |
| A2-X1 | X | Benchmark chuẩn | CDnet 2014, chế độ T **theo từng video** — cùng thiết lập theo video như AE-NE nên so sánh công bằng hơn A-X1; TAM tính từ DINOv3 đóng băng trên chính video | F-measure theo hạng mục, đặt cạnh số công bố của AE-NE | P0 nếu nhắm TIP/PR, P1 nếu nhắm T-ITS |
| A2-AB1 | AB | Số ảnh cơ sở | J ∈ {0, 1, 2, 4, 8}; độ phân giải E\_j: đầy đủ / nửa / một phần tư | PSNR nền, tỷ lệ "xe dính vào nền" (phần diện tích xe audit mà B^(1) giống frame hơn giống mặt đường) | P0 |
| A2-AB2 | AB | Trọng số robust | Charbonnier thuần / κ cố định (RobustNeRF) / κ\_t theo TAM / + hệ số (1 − π) / kích thước pool 1, 4, 8, 16 | IoU, A2-M2 | P0 |
| A2-AB3 | AB | Khởi tạo | Trung bình có trọng số TAM / median mọi giờ / ngẫu nhiên | IoU, số vòng hội tụ | P1 |
| A2-AB4 | AB | Rò rỉ xe vào mã ánh sáng | Đầu ℓ̂ gộp từ patch tĩnh / từ mọi patch | Tỷ lệ "xe dính vào nền" trong B̂ | P1 |
| A2-AB5 | AB | Loss giai đoạn 2 | Bỏ DropLoss / bỏ trọng số tin cậy / bỏ L\_bgPL / bỏ σ | IoU, PSNR | P0 (DropLoss, tin cậy) |
| A2-AB6 | AB | Vòng tự cải thiện | R = 0, 1, 2, 3 | IoU theo vòng | P1 |
| A2-AB7 | AB | Cách lấy mẫu giai đoạn 1 | Một slot giờ (như median theo slot) / mọi giờ một ngày / mọi giờ nhiều ngày | IoU, A2-M2 trên dữ liệu thật (sai số theo A\_c(p)) | P0 |
| A2-AB8 | AB | Backbone giai đoạn 2 | Đóng băng / mở 4 block / LoRA; ViT-S / ViT-B / backbone G | IoU | P1 |
| A2-AB9 | AB | Kích thước đầu vào | 256 × 256 (README cũ) / 448 × 256 | IoU, IoU riêng xe máy | P1 |
| A2-R1 | R | Không căn chỉnh camera | Bỏ IV.3 | IoU, PSNR | P1 |
| A2-R2 | R | Cảnh thay đổi lâu dài | Bán tổng hợp: một vật cố định (rào chắn) xuất hiện từ ngày k; khớp với cửa sổ trượt 7 / 14 ngày | Thời gian để nền cập nhật | P2 |
| A2-R3 | R | Frame xấu | FCS lên chế độ S | Suy giảm tương đối | P1 |
| A2-E1 | E | Chi phí | Phút GPU khớp mỗi camera; dung lượng E\_c; ms suy luận chế độ T và S; so với thời gian AE-NE phải huấn luyện mỗi camera | — | P0 |
| A2-Q1 | Q | Ảnh cơ sở học được là gì? | Hiển thị E\_0 … E\_4 của vài camera — kỳ vọng thấy ánh sáng, mặt đường ướt, bóng cây | Hình — dễ thuyết phục reviewer về tính diễn giải | P0 |
| A2-Q2 | Q | Kết quả phân rã | Lưới: frame, B̂, M, F ở cao điểm, đêm, mưa; so với median và AE-NE | Hình | P0 |
| A2-Q3 | Q | Bản đồ định danh và lỗi | U\_c(p) cạnh các trường hợp thất bại (làn luôn kẹt, xe đậu cố định) | Hình | P0 |

**Bảng/hình cho paper:** Bảng 1 = A2-M1; Bảng 2 = A2-X1; Bảng 3 = A2-AB1, AB2, AB5, AB7; Hình chính = đường cong điểm gãy A2-M2; Hình = ảnh cơ sở A2-Q1; Hình = A2-M3 (lịch sử cần thiết).

### A2.9. Rủi ro, cổng quyết định, lịch và venue

**Phụ thuộc:** A2 chỉ cần **thành phần TAM** của Hướng G (G.4, dùng DINOv3 đóng băng) — không cần chờ G huấn luyện xong. TAM qua Cổng 1 của G là đủ để bắt đầu A2.

**Hai cổng quyết định:**

1. **Cổng 1 (khoảng 3 tuần: dữ liệu đồ chơi + 20 camera thật):** giai đoạn 1 trên camera của audit-val phải (a) cho mask IoU ở lát cao điểm không thấp hơn II.A (prior median có σ), và (b) có điểm gãy trong A2-M2 cao hơn median theo slot. Nếu trượt: giữ II.A làm phương pháp chính, và dùng B^(1) của giai đoạn 1 làm **prior tốt hơn** cho II.A thay median — công sức không mất.
2. **Cổng 2 (sau giai đoạn 2):** chế độ S trên camera chưa thấy kém chế độ T không quá 0.05 IoU. Nếu kém hơn nhiều: paper nhấn mạnh chế độ T (vẫn thực tế vì camera cố định luôn có lịch sử), chế độ S đưa vào phần thảo luận.

**Rủi ro:**

| Rủi ro | Dấu hiệu | Cách giảm |
| --- | --- | --- |
| Ánh sáng không "ít chiều": đèn pha ban đêm, bóng tòa nhà quét qua theo giờ | Mask dương giả theo vệt bóng | Khớp E riêng cho các nhóm giờ (ví dụ 4 nhóm); tăng J; ablation A2-AB1 |
| Pixel bị che quá nhiều (vi phạm C2) | U\_c(p) = 1 trên diện rộng ở vài camera | Báo cáo riêng, không giấu; lấy mẫu thêm giờ đêm |
| Tự huấn luyện khuếch đại lỗi nhãn giả | IoU trên audit-val giảm qua vòng | Giới hạn R ≤ 3; dừng sớm theo audit-val; giữ DropLoss |
| Xe cùng màu mặt đường | Mask thiếu xe xám/đen trên nhựa | Hệ số TAM trong trọng số; A2-AB2 chứng minh |
| Chi phí giai đoạn 1 cho 1.200 cặp (camera, mode) | Vài chục giờ GPU mỗi vòng | Khớp theo lô 32 camera; giảm frame còn 300 nếu A2-M3 cho thấy đủ |

**Lịch khoảng 5 tháng:**

1. **Tháng 1:** `scene_fit.py`, `robust_weights`, `synth_occupancy.py`; nghiệm thu đồ chơi; chạy 20 camera → **Cổng 1**.
2. **Tháng 2:** giai đoạn 1 cho mọi camera; sửa `dataset.py`, `models.py`, `losses.py`, `train.py` theo A2.7; huấn luyện giai đoạn 2.
3. **Tháng 3:** A2-M1–M4, **Cổng 2**; baseline (AE-NE theo camera, SuBSENSE, PAWCS, detector VN + LaMa).
4. **Tháng 4:** ablation P0, CDnet 2014 (A2-X1), vòng tự cải thiện.
5. **Tháng 5:** hình, bảng, viết.

**Gộp paper:** đề xuất **một paper Hướng 2** với A2 là phương pháp chính (không cần ảnh nền) và II.A là biến thể "khi có prior" — câu chuyện liền mạch: từ prior nhiễu đến không cần prior, kèm phân tích điểm gãy. **Venue:** IEEE TIP, Pattern Recognition, IEEE TCSVT; IEEE T-ITS nếu nhấn mạnh ứng dụng giao thông.

## II.B. Gộp nhãn yếu đa nguồn cho mức độ ùn tắc

Tên gợi ý: *Context-Aware Weak Supervision for Congestion Recognition in Motorbike-Dominant Traffic*. Đây là hướng an toàn nhất: background chỉ là một nguồn nhãn yếu, mô hình tự học khi nào nên tin nó.

### Bối cảnh

Gán nhãn mức ùn tắc cho 608 camera theo thời gian là quá tốn công. Các nguồn tự động đều có điểm mù riêng: detector COCO bỏ sót xe máy chồng lấn khi đông; background proxy sai ở giờ kẹt và ban đêm; VLM zero-shot không ổn định và hay "đoán bừa"; tín hiệu thay đổi theo thời gian không phân biệt được đường vắng với đường kẹt cứng (cả hai đều ít thay đổi). Điểm mấu chốt: **các nguồn sai ở các điều kiện khác nhau**, nên gộp lại có thể tốt hơn từng nguồn.

### Vấn đề

Gộp nhiều nguồn nhãn mà độ tin cậy của mỗi nguồn không biết trước và **thay đổi theo điều kiện** (ngày/đêm, mưa, mật độ). Các mô hình gộp nhãn kinh điển (Dawid–Skene 1979, Snorkel — Ratner et al., VLDB 2017) giả định độ tin cậy cố định.

### Mục tiêu và câu hỏi nghiên cứu

1. RQ1: Mô hình gộp nhãn có điều kiện ngữ cảnh và ràng buộc thời gian có tốt hơn bỏ phiếu đa số và Dawid–Skene thường không?
2. RQ2: Mô hình cuối huấn luyện bằng nhãn gộp có vượt mọi nguồn riêng lẻ và tiến gần mô hình học có giám sát bằng gold set không?
3. RQ3: Độ tin cậy học được của từng nguồn theo điều kiện có giải thích được không (ví dụ nguồn background bị giảm trọng số ban đêm)?

### Phương pháp

**Các nguồn nhãn (labeling functions — LF).** Mỗi LF trả về 1 trong 4 mức (thông thoáng / trung bình / chậm / kẹt) hoặc "bỏ qua" (abstain).

| LF | Cách tính | Khi nào bỏ qua |
| --- | --- | --- |
| LF1 Detector | YOLOv8/RT-DETR (COCO) đếm car/motorcycle/bus/truck trong road mask, chia diện tích → chia mức theo ngưỡng chỉnh trên dev set | Độ tin cậy trung bình của box thấp |
| LF2 Background | ρ\_proxy (mục I.5) → chia mức | Độ tin cậy vùng tĩnh r\_i thấp (mục I.2) |
| LF3 VLM | Qwen2-VL hoặc InternVL 2–8B, prompt mô tả 4 mức + lựa chọn "không xác định" | VLM chọn "không xác định" hoặc 2 prompt khác nhau cho kết quả khác nhau |
| LF4 Thời gian | Độ tương đồng giữa frame liên tiếp trên road mask (cosine feature DINO) kết hợp mức có xe: tĩnh + nhiều xe → kẹt | Khoảng cách giữa 2 frame quá lớn |
| LF5 Lịch sử | Mức phổ biến nhất của cùng camera, cùng slot ở các ngày khác | Ít dữ liệu lịch sử |

**Biến ngữ cảnh c\_t:** ngày/đêm (từ giờ và độ sáng), mưa (VLM hoặc độ mờ ảnh), loại đường, mức mật độ thô.

**Mô hình gộp nhãn có ngữ cảnh và thời gian.** Mức ùn tắc thật y\_t là biến ẩn, thay đổi chậm theo thời gian (chuỗi Markov). Mỗi LF j có ma trận nhầm lẫn phụ thuộc ngữ cảnh π\_j^(c):

```latex
P(y_{1:T}, \lambda_{1:T}) = P(y_1)\prod_{t=2}^{T} A(y_t \mid y_{t-1}) \prod_{t=1}^{T}\prod_{j} \pi_j^{(c_t)}\big(\lambda_{j,t} \mid y_t\big)
```

Ước lượng bằng EM (forward–backward cho phần thời gian). Để tránh quá nhiều tham số, π\_j^(c) có thể tham số hóa bằng hồi quy logistic theo c thay vì một ma trận cho mỗi tổ hợp ngữ cảnh. Đầu ra: phân phối mềm P(y\_t | tất cả LF).

**Mô hình cuối.** DINOv3 + đầu thời gian nhân quả (GRU một chiều), huấn luyện bằng nhãn mềm (cross-entropy với nhãn kỳ vọng). Mô hình cuối không cần LF lúc suy luận và có thể vượt LF ở những frame mà mọi LF bỏ qua.

### Thí nghiệm

**Gold set:** dùng chung với H4 (800–1000 frame, 2–3 người gán, báo cáo κ). Chia 200 frame làm dev (chỉnh ngưỡng LF1, LF2), phần còn lại làm test; chia theo cụm camera (IV.3).

**Baseline:** từng LF riêng lẻ; bỏ phiếu đa số; Dawid–Skene không ngữ cảnh; Snorkel label model; học có giám sát trên gold set (cross-validation, cận trên); VLM lớn hơn dùng một mình.

**Metric:** macro-F1, quadratic weighted kappa (vì nhãn có thứ tự), báo cáo theo từng điều kiện (ngày/đêm, mưa, mật độ).

**Ablation:** bỏ từng LF (leave-one-out); bỏ ngữ cảnh; bỏ chuỗi Markov; kích thước VLM.

**Hình phân tích chính:** độ chính xác học được của từng LF theo điều kiện. Nếu LF2 (background) bị giảm mạnh ban đêm và giờ kẹt, đó chính là bằng chứng định lượng về độ tin cậy của background — đúng điều bạn đang thiếu.

### Rủi ro và cách giảm

- Chi phí VLM: lấy mẫu con khoảng 50–100k frame, chạy model 2–8B tại chỗ.
- LF1 và LF2 cùng sai khi đông (tương quan) → thêm tham số phụ thuộc giữa 2 LF hoặc nêu rõ giới hạn.
- LF4 phụ thuộc tần suất lấy mẫu → kiểm tra khoảng cách frame thực tế trước.

**Venue:** Information Fusion, EAAI, Expert Systems with Applications, IEEE T-ITS. **Công sức:** 3–4 tháng; phần lớn thời gian là gán gold set và chạy VLM.

## II.C. Phát hiện sự cố bất thường kéo dài (ngập nước, xe chết máy, vật cản)

Tên gợi ý: *Persistence-Aware, Camera-Conditioned Anomaly Detection for City-Scale Traffic Surveillance under Sparse Sampling*. Không cần background chính xác vì so sánh ở không gian feature và theo phân phối nhiều ngày.

### Bối cảnh

TP.HCM ngập thường xuyên vào mùa mưa (khoảng tháng 5–11) do mưa lớn và triều cường, gây tê liệt giao thông cục bộ. Ngoài ra còn tai nạn, xe chết máy, công trình chắn đường, cây đổ. Không thể có người theo dõi liên tục 608 camera. Sự cố hiếm, không có nhãn, và hình dạng rất đa dạng — đúng bối cảnh của phát hiện bất thường không giám sát.

### Vấn đề

Phân biệt **thay đổi bất thường kéo dài** (ngập, vật cản, xe hỏng nằm yên) với **thay đổi bình thường**: xe đi qua (thoáng qua), ánh sáng theo giờ, và cả kẹt xe giờ cao điểm (lặp lại hằng ngày nên là bình thường). Với dữ liệu lấy mẫu thưa, không dùng được optical flow hay tracking.

### Mục tiêu và câu hỏi nghiên cứu

1. RQ1: Gộp feature theo thời gian (median trong không gian feature) có loại bỏ được xe đi qua mà vẫn giữ sự cố kéo dài không?
2. RQ2: Mô hình "bình thường" theo từng camera và từng khung giờ có giảm báo động giả so với một mô hình chung không?
3. RQ3: Phát hiện sự cố nhanh đến đâu (độ trễ tính bằng phút) với tỷ lệ báo động giả chấp nhận được (ví dụ dưới 1 lần/camera/ngày)?

### Phương pháp

1. **Feature:** token patch của DINOv3 đóng băng, f\_t(p) cho mỗi patch p.
2. **Gộp theo thời gian trong không gian feature:** F̃\_t(p) = median của f(p) trên cửa sổ W frame gần nhất (ví dụ W = 5–10). Xe đi qua bị loại, thay đổi kéo dài được giữ lại. Đây giống như tính "background trực tuyến" nhưng ở mức feature, không phụ thuộc ảnh background có sẵn.
3. **Ngân hàng bình thường theo camera × nhóm giờ:** với mỗi camera và 4 nhóm giờ (đêm, cao điểm sáng, trưa, cao điểm chiều), lưu F̃ từ các ngày bình thường, nén bằng coreset như PatchCore (Roth et al., CVPR 2022). Ảnh background có sẵn có thể **thêm vào** ngân hàng như vài mẫu tham chiếu — không bắt buộc.
4. **Điểm bất thường:** a\_t(p) = khoảng cách đến láng giềng gần nhất trong ngân hàng; điểm cả ảnh = trung bình top-k patch trong road mask.
5. **Lọc tính kéo dài:** chỉ báo sự kiện khi điểm vượt ngưỡng trong ít nhất N cửa sổ liên tiếp. Ngưỡng của từng camera đặt bằng phân vị 99.5% điểm trên các ngày bình thường để giữ tỷ lệ báo động giả đồng đều giữa camera.
6. **Nhánh lỗi camera:** nếu bất thường nằm chủ yếu ở vùng tĩnh ngoài mặt đường (nhà, trời) → lỗi camera (bị xoay, che, mờ, chuyển hồng ngoại), tách thành loại riêng. Vừa có ích vận hành, vừa giảm báo động giả cho loại sự cố giao thông.
7. **Phân loại sự cố (tùy chọn):** chạy VLM trên frame bị gắn cờ để gợi ý loại (ngập / tai nạn / vật cản / lỗi camera); người duyệt lại cho tập test.

### Xây dựng tập đánh giá

| Nguồn | Cách làm | Dùng để đo |
| --- | --- | --- |
| Khai thác ứng viên | Người duyệt top điểm cao của mọi phương pháp (gộp chung để công bằng) | Precision |
| Đối chiếu bên ngoài | Ngày mưa lớn, điểm ngập từ báo chí và thông báo của thành phố → kiểm tra camera gần đó | Recall cho ngập |
| Mẫu ngẫu nhiên | Duyệt các cửa sổ ngẫu nhiên để ước lượng sự cố bị bỏ sót | Recall xấp xỉ |
| Sự kiện tổng hợp | Dán vật cản cố định (xe hỏng, rào chắn) vào chuỗi frame | Độ trễ, có kiểm soát |

**Metric:** precision/recall mức sự kiện, độ trễ phát hiện (phút), số báo động giả mỗi camera mỗi ngày, AUROC mức frame.

**Baseline:** PatchCore mức frame không gộp thời gian; autoencoder tái tạo; ngưỡng trên Δ với background có sẵn; VLM zero-shot trên mọi frame; ngân hàng chung cho mọi camera.

**Ablation:** W, N; ngân hàng theo nhóm giờ vs chung; có/không thêm background có sẵn vào ngân hàng.

### Rủi ro và cách giảm

- Ít sự kiện: cần dữ liệu trải qua mùa mưa; nếu chưa có, tập trung vào vật cản/xe hỏng/lỗi camera trước.
- Kẹt xe bất thường (ngoài giờ cao điểm) có tính là sự cố không? Cần định nghĩa phân loại rõ trong paper.
- Lấy mẫu quá thưa (trên 5 phút/frame) làm độ trễ lớn — báo cáo trung thực theo tần suất lấy mẫu.

**Venue:** IEEE T-ITS, Transportation Research Part C, EAAI, Expert Systems with Applications. **Công sức:** 4–5 tháng, chủ yếu là xây tập đánh giá; phần mô hình nhẹ, không cần huấn luyện backbone.

## II.D. Dự báo ùn tắc trên đồ thị mạng camera

Tên gợi ý: *City-Scale Congestion Forecasting from Surveillance Camera Networks in Motorbike-Dominant Traffic*. Background gần như không cần dùng.

### Bối cảnh

Nghiên cứu dự báo giao thông chủ yếu dùng dữ liệu cảm biến vòng từ trên cao tốc Mỹ (METR-LA, PEMS-BAY). Ở Việt Nam gần như không có mạng cảm biến như vậy, nhưng có hàng trăm camera — trên thực tế đây chính là mạng cảm biến của thành phố. Giao thông xe máy hỗn hợp có động học khác hẳn cao tốc: ùn tắc hình thành nhanh, lan qua nút giao, phụ thuộc mạnh vào giờ tan học/tan tầm và mưa.

### Vấn đề

Dự báo mức ùn tắc tại mỗi camera sau h phút từ lịch sử hình ảnh của toàn mạng. Khó khăn: (1) đại lượng cần dự báo được suy ra từ ảnh nên có nhiễu; (2) camera mất tín hiệu thường xuyên → dữ liệu thiếu; (3) góc nhìn mỗi camera khác nhau.

### Mục tiêu và câu hỏi nghiên cứu

1. RQ1: Mô hình đồ thị dùng thông tin các camera lân cận có dự báo tốt hơn mô hình từng camera riêng lẻ không?
2. RQ2: Embedding hình ảnh giàu thông tin có tốt hơn một chỉ số vô hướng (mức ùn tắc) làm đặc trưng nút không?
3. RQ3: Mô hình có bền khi 10–50% camera mất tín hiệu không?

### Điều kiện dữ liệu cần kiểm tra trước

- Tọa độ (và nếu có, hướng nhìn) của từng camera.
- Tần suất lấy mẫu đều đặn, tốt nhất ≤ 5 phút/frame.
- Ít nhất 2–3 tháng liên tục; có cả mùa mưa thì càng tốt.

### Phương pháp

1. **Chỉ số giao thông hình ảnh tại mỗi nút:** x\_t^i = phân phối mức ùn tắc từ mô hình cuối của Hướng B (hoặc ρ ước lượng từ H4 đã sửa), cộng embedding e\_t^i = token \[CLS\] của DINO nén bằng PCA xuống 32–64 chiều.
2. **Đưa về lưới thời gian đều** Δt = 5 phút; mặt nạ thiếu m\_t^i = 0 khi không có frame.
3. **Đồ thị:** nút là camera; cạnh theo khoảng cách đường đi ngắn nhất trên mạng đường OpenStreetMap (kernel Gaussian, cắt ngưỡng), cộng thêm ma trận kề học được như Graph WaveNet.
4. **Mô hình:** khung Graph WaveNet hoặc STAEformer, với 3 điều chỉnh: (a) đầu vào đa phương thức \[x, e\] có modality dropout; (b) nhận biết dữ liệu thiếu — đưa mặt nạ m vào làm kênh đầu vào, loss chỉ tính trên điểm có quan sát, và lúc train che ngẫu nhiên cả nút để mô phỏng camera hỏng; (c) đầu ra nhiều tầm h = 15, 30, 60 phút.
5. **Ba đầu dự báo:** hồi quy chỉ số liên tục (masked MAE); phân loại mức ùn tắc có thứ tự (ordinal CE); và **đầu cảnh báo khởi phát kẹt** — xác suất mức ≥ "kẹt" trong h phút tới khi hiện tại chưa kẹt (focal loss vì hiếm).

### Thí nghiệm

**Chia dữ liệu theo thời gian:** 70% tuần đầu train, 10% val, 20% tuần cuối test — tuyệt đối không chia ngẫu nhiên. Thêm một thí nghiệm giữ lại một nhóm camera để kiểm tra nút chưa thấy.

**Đánh giá với nhãn người:** vì chỉ số được suy ra từ ảnh, cần gán gold một mẫu frame **trong giai đoạn test** để đo sai số dự báo so với nhãn người, không chỉ so với chỉ số tự sinh.

**Baseline:** trung bình lịch sử (cùng thứ, cùng khung giờ); giá trị gần nhất; ARIMA; LSTM từng nút; DCRNN (Li et al., ICLR 2018); Graph WaveNet (Wu et al., IJCAI 2019); STAEformer (Liu et al., CIKM 2023).

**Metric:** MAE/RMSE theo từng tầm h; macro-F1 và QWK cho mức ùn tắc; precision/recall và thời gian báo trước cho cảnh báo khởi phát.

**Ablation:** loại đồ thị (khoảng cách / học được / không đồ thị); đặc trưng nút (vô hướng / embedding / cả hai); đường cong bền theo tỷ lệ camera mất tín hiệu 10–50%; ảnh hưởng của khoảng lấy mẫu.

### Đóng góp dự kiến

Benchmark dự báo dựa trên camera quy mô thành phố trong giao thông xe máy (cần tra cứu kỹ để khẳng định "đầu tiên"); mô hình đa phương thức nhận biết dữ liệu thiếu; đầu cảnh báo khởi phát kẹt.

### Rủi ro và cách giảm

- Sai số của chỉ số hình ảnh lan sang dự báo → báo cáo cả sai số so với gold set; dùng nhãn mềm thay vì nhãn cứng.
- Thiếu metadata camera → xin từ nguồn IC4SD hoặc gán tay tọa độ 608 camera.
- Tết, ngày lễ làm lệch phân phối → dùng làm bài kiểm tra stress riêng thay vì loại bỏ.

**Venue:** Transportation Research Part C, IEEE T-ITS, IEEE TKDE. **Công sức:** 4–6 tháng; nên làm sau Hướng B vì dùng mô hình của B để tạo chỉ số.

## II.E. Thích ứng camera mới bằng thống kê toàn cục của background

Tên gợi ý: *Background-Conditioned Generalization to Unseen Traffic Cameras with Unreliable Scene Priors*. Đây là hướng nhanh nhất và tận dụng code H3 có sẵn.

### Bối cảnh

Mô hình đếm xe hoặc nhận diện ùn tắc huấn luyện trên một nhóm camera thường giảm mạnh trên camera mới: phối cảnh khác, kích thước xe khác, bố cục đường khác, ánh sáng khác. Mỗi camera mới lại cần gán nhãn — không khả thi cho hàng trăm camera. Trong khi đó, mỗi camera đã có sẵn ảnh background theo từng giờ: một "bản mô tả cảnh" miễn phí, gần như không có xe.

### Vấn đề

Dùng background để mô tả camera mà không bị lỗi cục bộ (ghost, bóng đổ) làm hỏng. Dùng ở mức pixel (như kênh Δ của H3) thì nhạy nhiễu; câu hỏi là dùng ở mức toàn cục có bền hơn mà vẫn hữu ích không.

### Mục tiêu và câu hỏi nghiên cứu

1. RQ1: Điều kiện hóa bằng thống kê toàn cục của background có cải thiện tổng quát hóa sang camera chưa thấy so với không dùng background và so với kênh Δ không?
2. RQ2: Cách dùng toàn cục có bền hơn cách dùng mức pixel khi background có ghost không (BDB)?
3. RQ3: Dạng điều kiện hóa nào tốt nhất: FiLM, prompt token, hay cross-attention?

### Phương pháp

**1. Bản mô tả cảnh z.** Đưa ảnh background qua DINOv3 đóng băng, lấy token patch, tính trung bình và độ lệch chuẩn trên các patch thuộc road mask và trên toàn ảnh. Để bền hơn, dùng **trung bình cắt** (bỏ 10% patch xa median nhất) và gộp mô tả của 2–3 slot giờ liền kề:

```latex
z = \Big[\, \operatorname{TrimMean}_{p \in R}\, f_{\text{bg}}(p),\ \operatorname{TrimStd}_{p \in R}\, f_{\text{bg}}(p),\ \operatorname{TrimMean}_{p}\, f_{\text{bg}}(p) \,\Big]
```

Lý do bền: ghost phủ tỷ lệ q diện tích chỉ làm lệch trung bình cỡ q lần độ chênh feature, và trung bình cắt còn loại bớt phần đó. Điều này kiểm chứng được bằng thí nghiệm chèn ghost.

**2. Ba dạng điều kiện hóa (so sánh trong paper).**

| Dạng | Cách làm | Thông tin từ bg |
| --- | --- | --- |
| FiLM (Perez et al., AAAI 2018) | γ\_l, β\_l = MLP\_l(z); h ← γ\_l ⊙ LN(h) + β\_l ở L block cuối; khởi tạo γ = 1, β = 0 | Chỉ toàn cục |
| Scene prompt | K token = MLP(z) nối vào chuỗi token của ViT | Chỉ toàn cục |
| Cross-attention | Token origin attend vào token patch của bg | Cục bộ — nhạy nhiễu nhất, dùng để so sánh |

**3. Huấn luyện bền.** bg-dropout 20–30% (thay z bằng một token "rỗng" học được); bg-swap: lấy z từ slot lệch ±1–2 giờ; chèn ghost 0–30% vào ảnh bg trước khi tính z.

**4. Hai bài toán để chứng minh tính tổng quát:** đếm xe (nhãn H3) và mức ùn tắc (gold/nhãn mềm từ Hướng B).

**5. Suy luận trên camera mới:** chỉ cần ảnh background của camera đó, không cần nhãn, không cần cập nhật trọng số.

### Thí nghiệm

**Chia dữ liệu:** theo cụm camera (IV.3) và khó hơn: giữ lại cả một quận/khu vực; thêm kịch bản chỉ train ban ngày, test ban đêm.

**Baseline:** không dùng bg; H3 kênh Δ; late fusion Δ; TENT (Wang et al., ICLR 2021) — thích ứng lúc test không cần bg; fine-tune với k frame có nhãn của camera mới (mốc tham chiếu trên).

**Metric:** MAE/RMSE (đếm), macro-F1/QWK (ùn tắc). **Hình chính:** đường cong BDB của FiLM, prompt, cross-attention và kênh Δ trên cùng một hình.

**Phân tích:** t-SNE của z tô màu theo loại đường và giờ; tương quan z với kích thước xe trung bình (đại diện cho phối cảnh) để cho thấy z mã hóa thông tin camera có ý nghĩa.

### Rủi ro và cách giảm

- DINOv3 vốn đã tổng quát tốt nên cải thiện có thể nhỏ → tập trung vào các dịch chuyển khó (đêm, khu vực mới) và câu chuyện độ bền.
- z có thể mã hóa luôn "camera ID" và gây quá khớp → bg-dropout và chia theo khu vực kiểm soát điều này.

**Venue:** EAAI, Pattern Recognition, IEEE T-ITS. **Công sức:** 2–3 tháng.

## II.F. Paper dataset cho Scientific Data

Nếu được phép công bố dữ liệu, một Data Descriptor trên Scientific Data (Q1) gần như tận dụng lại toàn bộ nhãn đã làm cho các hướng khác. Đây thường là loại paper được trích dẫn nhiều nhất.

### Bối cảnh và khoảng trống

Dữ liệu giao thông công khai chủ yếu đến từ cao tốc và đô thị phương Tây, giao thông ô tô là chính. Rất thiếu dữ liệu chuỗi thời gian từ mạng camera quy mô thành phố với giao thông xe máy hỗn hợp, có kèm ảnh nền theo giờ và nhãn đa nhiệm.

### Nội dung bộ dữ liệu

| Thành phần | Mô tả | Lấy từ |
| --- | --- | --- |
| Chuỗi ảnh | Frame theo thời gian của 608 camera (hoặc một tập con), metadata: camera ID, thời điểm, tọa độ | Dữ liệu hiện có |
| Ảnh nền | Background theo camera × slot giờ, **kèm điểm tin cậy** | Dữ liệu hiện có + Hướng A |
| Road mask | Polygon mặt đường cho 608 camera | Mục 0.2 |
| Tập audit | Ghost trên background, mask xe trên origin | Mục 0.2 |
| Gold set | Mức ùn tắc (nhiều người gán, có κ), chiếm dụng mặt đường | H4, Hướng B |
| Nhãn đếm | Số xe máy, ô tô | H3 |
| Sự kiện | Ngập, vật cản, lỗi camera có thời gian bắt đầu/kết thúc | Hướng C |
| Benchmark | Đếm xe, mức ùn tắc, phát hiện sự cố, dự báo; có code và baseline | Các hướng |

### Cấu trúc bài theo yêu cầu Scientific Data

Background & Summary → Methods (thu thập, tính background, quy trình gán nhãn) → Data Records → **Technical Validation** → Usage Notes → Code Availability. Phần Technical Validation chính là Background Reliability Audit (mục 0.2), độ đồng thuận giữa người gán và BDB — tức điểm yếu "background nhiễu" trở thành nội dung được đo đạc minh bạch, đúng điều tạp chí đánh giá cao.

### Việc phải làm trước khi bắt đầu

- [ ] Xác nhận quyền công bố với đơn vị sở hữu dữ liệu (IC4SD / cơ quan quản lý camera) và chọn giấy phép (ví dụ CC BY-NC 4.0)
- [ ] Ẩn danh: làm mờ khuôn mặt và biển số bằng detector tự động, kiểm tra ngẫu nhiên bằng tay
- [ ] Kiểm tra phí xuất bản (APC) của Scientific Data và nguồn tài trợ
- [ ] Chọn nơi lưu trữ có DOI (Zenodo, Figshare, Harvard Dataverse)

Nếu không được phép công bố ảnh gốc, phương án thay thế yếu hơn nhiều (chỉ công bố feature + nhãn); khi đó nên ưu tiên các hướng A–E.

**Venue:** Scientific Data; hoặc bài dataset kèm benchmark gửi IEEE T-ITS. Lưu ý Data in Brief không phải Q1. **Công sức:** thêm 1–2 tháng nếu đã có nhãn từ các hướng khác.

## II.G. Hướng G — Học biểu diễn hướng phương tiện từ chuỗi ảnh thưa của camera cố định (H1 phiên bản 2)

Tên gợi ý: *Learning Vehicle-Centric Representations from Sparse Fixed-Camera Image Streams without Background Images*. Hướng G **thay thế H1**: bỏ hoàn toàn ảnh background median, thay bằng thống kê feature theo từng vị trí học từ chính chuỗi ảnh. FAM-Δ của H1 trở thành một baseline.

### G.1. Bối cảnh, vấn đề và ý tưởng

**Bối cảnh.** Camera giao thông cố định cho ra chuỗi ảnh nhìn cùng một cảnh suốt nhiều tháng. Với một vị trí (i, j) trên ảnh, phần lớn thời gian nó là mặt đường, cột điện hay tòa nhà; thỉnh thoảng có xe đi qua. Cấu trúc này là "giám sát miễn phí" về chỗ nào là phương tiện, nhưng hiện chưa được khai thác đúng cách:

- Các phương pháp tự học hiện đại (DINOv2/v3, iBOT, MAE) coi mỗi ảnh độc lập, che ngẫu nhiên — phần lớn công sức học rơi vào mặt đường.
- Các phương pháp che theo chuyển động (MGMAE, MGM, V-JEPA4A) cần video liên tục để tính optical flow hoặc hiệu frame liền kề. Dữ liệu của ta là ảnh chụp cách nhau vài phút — không dùng được.
- H1 dùng Δ với ảnh nền median — chịu toàn bộ lỗi của median (ghost ở giờ kẹt, ánh sáng, lệch camera).
- Khi tự học trên dữ liệu có nền dùng chung, mạng dễ "đi đường tắt": học nhận ra nền thay vì đối tượng.

**Vấn đề.** Làm sao, chỉ từ chuỗi ảnh thưa của camera cố định, không có nhãn và không có ảnh nền, để mô hình (1) biết vùng nào nhiều khả năng là phương tiện, (2) dồn năng lực học vào đó một cách hợp lý, và (3) không dựa vào nền để giải bài toán tự học?

**Ý tưởng cốt lõi — 3 thành phần:**

1. **Bản đồ độ khác thường theo vị trí (TAM — Temporal Atypicality Map).** Với mỗi camera và mỗi vị trí patch, duy trì vài "trạng thái tĩnh" phổ biến trong không gian feature DINO (ví dụ mặt đường khô, mặt đường ướt, có bóng cây). Patch hiện tại càng xa mọi trạng thái tĩnh thì càng có khả năng là xe. Mặt đường lặp lại nên tạo cụm chặt và thường xuyên; xe thì đa dạng về màu, loại, vị trí nên không tạo được cụm như vậy.
2. **Che phân tầng theo TAM (AGM — Atypicality-Guided Masking).** Kiểm soát rõ ràng hai đại lượng: tỷ lệ ngân sách che dành cho vùng xe, và tỷ lệ tối đa vùng xe bị che (để xe còn lại làm ngữ cảnh). Câu hỏi "che nhiều hay ít vùng xe" trở thành một thí nghiệm, không phải giả định.
3. **Hoán đổi vùng tĩnh giữa các ngày (SRS — Static-Region Swap).** Thay các patch tĩnh của frame ngày d bằng patch cùng vị trí từ frame ngày d' của cùng camera. Hai ảnh có cùng phương tiện nhưng khác "nền" (ánh sáng, độ ướt, bóng) → buộc biểu diễn bất biến với nền và tập trung vào xe.

**Điểm mạnh so với H1:** không cần ảnh nền; không cần frame liền kề (lấy mẫu thưa đến đâu cũng dùng được, chỉ cần nhiều ngày); TAM tự cải thiện khi feature tốt hơn; mọi thành phần đều đo kiểm được trên tập audit.

### Câu hỏi nghiên cứu

1. **RQ1:** TAM có định vị phương tiện tốt hơn Δ pixel với ảnh nền median và tốt hơn các phương pháp phát hiện đối tượng không giám sát trên từng ảnh (MaskCut), đặc biệt ở cao điểm và ban đêm không?
2. **RQ2:** Che theo TAM có cho biểu diễn tốt hơn che ngẫu nhiên, che theo attention và che theo Δ với cùng compute không? Và nên che nhiều hơn hay ít hơn vùng xe?
3. **RQ3:** SRS có làm giảm mức dựa vào nền và tăng tổng quát hóa sang camera, vùng, điều kiện chưa thấy không?
4. **RQ4:** Biểu diễn có thật sự "hiểu phương tiện" hơn không — đo bằng probe dày (phân đoạn xe), probe nhận dạng camera, và các phép thử can thiệp (hoán đổi nền, xóa xe)?

### Đóng góp dự kiến

1. TAM: thống kê feature đa trạng thái theo vị trí cho camera cố định, không cần ảnh nền và không cần chuyển động.
2. AGM: chiến lược che phân tầng với hai tham số diễn giải được, kèm bằng chứng thực nghiệm về hướng che tối ưu.
3. SRS: phép tăng cường dựa trên cấu trúc cùng camera để chống đường tắt nền.
4. Bộ chẩn đoán đường tắt nền cho tự học trên camera cố định: probe camera ID, độ nhất quán khi hoán đổi nền, độ nhạy khi xóa xe.

### G.2. Công trình liên quan và khoảng trống

Đã tra cứu tháng 10/2026. Các mục có link là đã xác nhận qua trang chính thức hoặc trang bài báo.

| Nhóm | Công trình | Làm gì | Khác ta ở đâu |
| --- | --- | --- | --- |
| Che theo chuyển động | [MGMAE (Huang et al., ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Huang_MGMAE_Motion_Guided_Masking_for_Video_Masked_Autoencoding_ICCV_2023_paper.html) | Dùng optical flow trực tuyến để tạo khối che nhất quán theo thời gian cho video MAE | Cần video liên tục |
| Che theo chuyển động | [MGM (Fan et al., ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Fan_Motion-Guided_Masking_for_Spatiotemporal_Representation_Learning_ICCV_2023_paper.html) | Lấy motion vector từ video nén để dẫn vị trí che | Cần video nén liên tục |
| Che theo vùng quan trọng, giao thông | [V-JEPA4A — Mask What Matters (Lang, Braun, Valada, arXiv 08/2026)](https://www.alphaxiv.org/abs/2608.17178) | Che theo độ quan trọng ngữ nghĩa và thời gian cho video lái xe (camera gắn trên xe) | Camera di chuyển, video liên tục; xem ghi chú bên dưới |
| Che học được | [ADIOS (Shi et al., ICML 2022)](https://proceedings.mlr.press/v162/shi22d.html) | Học hàm che bằng mục tiêu đối kháng; cho thấy che theo mask đối tượng thật tốt hơn hẳn các cách che khác | Không dùng cấu trúc camera cố định; là baseline tham khảo |
| Che theo attention | AttMask (Kakogeorgiou et al., ECCV 2022) — chưa kiểm chứng link | Che theo attention của teacher | Chỉ dùng thông tin trong một ảnh |
| Đường tắt nền | [Background Erasing (Wang et al., CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Removing_the_Background_by_Adding_the_Background_Towards_Background_Robust_CVPR_2021_paper.html) | Trộn một frame tĩnh vào mọi frame của video và buộc feature không đổi → chống dựa vào nền | Trộn toàn ảnh ở mức pixel; SRS hoán đổi chọn lọc theo vị trí tĩnh |
| Đường tắt nền | [Jenni & Favaro (arXiv 2010.06218)](https://arxiv.org/pdf/2010.06218) | Ghi nhận nền dùng chung của camera cố định tạo đường tắt; xử lý bằng trừ nền median | Vẫn cần ảnh nền median |
| Ngữ cảnh camera cố định | [Focus on the Positives (Pantazis et al., ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Pantazis_Focus_on_the_Positives_Self-Supervised_Learning_for_Biodiversity_Monitoring_ICCV_2021_paper.html) | Camera bẫy ảnh: dùng ngữ cảnh không gian–thời gian để chọn cặp **dương** | Ta dùng cùng camera để tạo cặp dương có nền khác (SRS) và, trong ablation, cặp **âm** |
| Ảnh cùng vị trí, khác thời điểm | [MoCo với temporal positives cho ảnh vệ tinh (arXiv 2210.11815)](https://arxiv.org/pdf/2210.11815) | Ảnh cùng vị trí khác thời điểm làm cặp dương → học đặc trưng cảnh bền vững | Học cái **tĩnh**; ta muốn học cái **động** |
| Tự học dày theo thời gian | [TimeT (Salehi et al., ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Salehi_Time_Does_Tell_Self-Supervised_Time-Tuning_of_Dense_Image_Representations_ICCV_2023_paper.html) | Fine-tune ViT bằng loss phân cụm căn chỉnh theo thời gian trên video | Cần căn chỉnh qua frame liền kề; camera cố định cho căn chỉnh vị trí miễn phí |
| Phát hiện đối tượng không giám sát | [CutLER/MaskCut (Wang et al., CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Cut_and_Learn_for_Unsupervised_Object_Detection_and_Instance_Segmentation_CVPR_2023_paper.pdf); TokenCut; LOST | Cắt đồ thị trên feature DINO để tìm đối tượng trong từng ảnh | Chỉ dùng một ảnh; khó với hàng trăm xe máy nhỏ chồng nhau — là baseline cho RQ1 |

**Ghi chú quan trọng từ V-JEPA4A:** theo phần tóm tắt trên trang alphaXiv, nhóm tác giả thấy xác suất che vùng tiền cảnh nên **thấp hơn** vùng nền (che vùng xe quá nhiều làm student mất ngữ cảnh). Điều này ngược với giả định "che nhiều vùng xe" của H1 — cần đọc toàn văn để xác nhận. Vì vậy AGM không cố định hướng che mà tham số hóa nó, và thí nghiệm G-AB4 trả lời trực tiếp câu hỏi này trên dữ liệu camera cố định.

**Khoảng trống của Hướng G** (trong phạm vi đợt tra cứu này, chưa thấy công trình làm đủ các điểm sau): tự học biểu diễn từ **ảnh chụp thưa** của **camera cố định**, dùng **thống kê feature dài hạn theo vị trí** thay cho ảnh nền và cho chuyển động, để vừa dẫn hướng che vừa chống đường tắt nền. Trước khi nộp phải tra lại theo giao thức IV.13, đặc biệt các từ khóa "static camera self-supervised", "surveillance pretraining", "background bias masked image modeling".

### G.3. Tổng quan phương pháp

&#91;embedded content: Phương pháp G · TAM dẫn hướng cả việc che (AGM) lẫn việc hoán đổi nền (SRS)\]

Mỗi bước huấn luyện: (1) một mô hình DINOv3 đóng băng chạy trên khung hình đầy đủ để cập nhật thống kê theo vị trí và tính TAM; (2) SRS ghép patch tĩnh từ một frame ngày khác vào ảnh của student; (3) AGM chọn patch để che theo TAM; (4) teacher nhận ảnh gốc, student nhận ảnh đã biến đổi, loss DINO + iBOT như DINOv2. Kết quả cuối cùng chỉ là một backbone ViT, dùng cho mọi hướng khác (H3, H4, A, B, C, E) thay cho DINOv3 gốc.

### G.4. Thành phần 1 — Bản đồ độ khác thường theo vị trí (TAM)

**Feature đầu vào.** Frame đã căn chỉnh (IV.3), resize về 448 × 256 → DINOv3 ViT-B/16 **đóng băng** → 16 × 28 = 448 token patch, 768 chiều → chuẩn hóa L2 → PCA về d = 64 (fit một lần trên 200k token ngẫu nhiên của camera train) → chuẩn hóa L2 lần nữa, ký hiệu u\_t(p). Dùng mô hình đóng băng để TAM ổn định và không phụ thuộc vào quá trình huấn luyện; biến thể dùng teacher EMA là một ablation.

**Thống kê theo vị trí.** Với mỗi camera c, chế độ m (ngày / IR) và vị trí p, lưu K = 4 "trạng thái": tâm μ\_k (vector đơn vị d chiều), tần suất w\_k (tổng bằng 1) và độ phân tán s\_k (khoảng cách cosine trung bình của các mẫu thuộc trạng thái đó). Bộ nhớ: 608 camera × 2 chế độ × 448 vị trí × 4 trạng thái × 64 chiều × fp16 ≈ 280 MB, đặt thẳng trên GPU.

**Vì sao phân biệt được mặt đường và xe.** Ví dụ vị trí (5, 12) trên lòng đường: 70% thời gian là nhựa xám khô, 15% là nhựa ướt, 15% là các loại xe khác nhau. Hai trạng thái mặt đường tạo cụm **chặt** và **thường xuyên** (w lớn, s nhỏ). Xe đỏ, xe đen, xe buýt, người đi bộ phân tán khắp không gian feature, không gom được thành một trạng thái có w lớn. Vì vậy chỉ coi trạng thái là "tĩnh" khi vừa thường xuyên vừa chặt.

**Cập nhật trực tuyến** (mỗi frame trong batch, không qua gradient):

```latex
k^* = \arg\max_k \cos\big(u_t(p), \mu_k\big), \quad
\mu_{k^*} \leftarrow \frac{(1-\eta)\,\mu_{k^*} + \eta\, u_t(p)}{\lVert \cdot \rVert}, \quad
w_k \leftarrow (1-\eta_w)\, w_k + \eta_w\, \mathbb{1}[k = k^*], \quad
s_{k^*} \leftarrow (1-\eta)\, s_{k^*} + \eta \big(1 - \cos(u_t(p), \mu_{k^*})\big)
```

Với η = 0.02, η\_w = 0.005 (tần suất phản ánh khoảng 200 lần quan sát gần nhất). Nếu cos(u, μ\_{k\*}) < 0.6 và có trạng thái với w < 0.02 thì khởi tạo lại trạng thái đó bằng u (w = 0.05) — cho phép xuất hiện trạng thái mới như mặt đường ướt.

**Khởi tạo.** Chạy một lượt offline: mỗi (camera, chế độ) lấy 300 frame rải trên ít nhất 10 ngày, chạy k-means cosine K = 4 cho từng vị trí → μ, w = tỷ lệ cụm, s.

**Tập trạng thái tĩnh và độ khác thường:**

```latex
\mathcal{S}(c,m,p) = \{\, k : w_k \ge w_{\min},\; s_k \le s_{\max} \,\}, \qquad
a_t(p) = \min_{k \in \mathcal{S}} \frac{1 - \cos\big(u_t(p), \mu_k\big)}{s_k + \epsilon}
```

Với w\_min = 0.15, s\_max = 1.5 × median của mọi s\_k trong camera, ε = 0.01. Chia cho s\_k giống chuẩn hóa kiểu Mahalanobis: trạng thái "lỏng" (ví dụ bóng cây lay động) được phép lệch nhiều hơn. Nếu 𝒮 rỗng — vị trí gần như lúc nào cũng có xe hoặc không ổn định — gán a\_t(p) = không xác định, và AGM xử lý vị trí đó như che ngẫu nhiên. **Báo cáo tỷ lệ vị trí rỗng** theo camera: đây là giới hạn thật của phương pháp và phải nêu trong paper.

**Từ điểm sang tập tiền cảnh (cho AGM và SRS).** Với mỗi (camera, chế độ), fit Gaussian Mixture 2 thành phần trên log a\_t(p) của 50.000 giá trị gần nhất (cập nhật mỗi 1.000 bước) → xác suất tiền cảnh π\_t(p) = hậu nghiệm của thành phần có trung bình lớn hơn. Tập tiền cảnh F\_t = {p : π\_t(p) > 0.5}. Không dùng nhãn. Ngưỡng này được **kiểm tra** (không chỉnh) trên audit-val.

**Sản phẩm phụ — bản đồ hoạt động của camera:** A\_c(p) = 1 − Σ\_{k∈𝒮} w\_k, tức tỷ lệ thời gian vị trí p không ở trạng thái tĩnh. Đây là bản đồ "làn xe hay có xe", không cần road mask — dùng được cho Hướng E (bản mô tả camera) và để kiểm tra chéo road mask người vẽ.

**Giới hạn cần nêu:**

- Vị trí bị chiếm hơn 85% thời gian bởi vật trông giống nhau (hàng xe máy đậu cố định ở mép đường) sẽ bị coi là tĩnh — về mặt ngữ nghĩa có thể chấp nhận.
- Thay đổi ánh sáng nhanh trong ngày tạo nhiều trạng thái; K = 4 có thể thiếu — ablation K.
- Camera bị rung: vị trí patch lệch → mọi thứ thành khác thường. Dựa vào căn chỉnh IV.3, và phát hiện bằng tỷ lệ patch khác thường trên vùng tĩnh (nếu > 50% thì bỏ frame).

### G.5. Thành phần 2 — Che phân tầng theo TAM (AGM)

**Vấn đề với FAM của H1:** chỉ có một núm α, trộn giữa "che theo Δ" và "che đều", nên không trả lời được câu hỏi cốt lõi: nên dành bao nhiêu ngân sách che cho vùng xe, và có nên để lại một phần xe làm ngữ cảnh không. V-JEPA4A gợi ý rằng che vùng tiền cảnh quá nhiều có hại. AGM tách thành hai tham số diễn giải được:

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
    B = ~F                                  # còn lại, gồm cả vị trí không xác định
    k_fg = min(round(phi * n_mask), int(q_max * F.sum()))
    k_bg = min(n_mask - k_fg, int(B.sum()))
    fg_idx = sample_without_replacement(F, k_fg, weights=pi, gen=gen)   # ưu tiên patch chắc là xe
    bg_idx = sample_without_replacement(B, k_bg, weights=None, gen=gen) # đều
    mask = zeros(N, bool); mask[fg_idx] = True; mask[bg_idx] = True
    return mask
```

Lấy mẫu không hoàn lại có trọng số bằng Gumbel top-k (như IV.6). Nếu ảnh không có xe (F rỗng) thì AGM trở về che ngẫu nhiên — đúng hành vi mong muốn cho ảnh đêm vắng.

**Ánh xạ theo crop.** π được tính trên khung hình đầy đủ (lưới 16 × 28). Với mỗi global crop, áp cùng tham số RandomResizedCrop và lật lên bản đồ π (nội suy bilinear) rồi average-pool về lưới 14 × 14 — cùng cách đã đặc tả cho Δ ở IV.6.

**Bốn chính sách so sánh trong G-AB4** (cùng tỷ lệ che tổng):

| Chính sách | φ | q\_max | Giả thuyết |
| --- | --- | --- | --- |
| Ngẫu nhiên | tỷ lệ diện tích xe tự nhiên | 1.0 | Mốc so sánh |
| Che nhiều xe (kiểu H1) | 0.7 | 1.0 | Dồn công học vào xe |
| Phân tầng cân bằng (mặc định) | 0.5 | 0.6 | Học xe nhiều hơn nhưng giữ ngữ cảnh xe |
| Giữ xe (kiểu V-JEPA4A) | 0.5 × tỷ lệ tự nhiên | 1.0 | Xe làm ngữ cảnh, che chủ yếu nền |

Thêm lưới quét φ ∈ {0.2, 0.35, 0.5, 0.7} × q\_max ∈ {0.4, 0.6, 0.8, 1.0} trên một seed để vẽ bề mặt hiệu năng — đây là một hình đáng đưa vào paper vì nó trả lời câu hỏi "che bao nhiêu xe" một cách có hệ thống.

**Lịch φ (tùy chọn).** Tăng φ tuyến tính từ tỷ lệ tự nhiên lên giá trị đích trong 30% đầu quá trình huấn luyện, vì TAM và biểu diễn ban đầu chưa ổn định.

**Loss iBOT có trọng số (ablation).** Nhân loss patch bị che với (1 + β·π), β ∈ {0, 1, 2} — kiểm tra xem tăng trọng số loss có tương đương với tăng tỷ lệ che vùng xe không.

### G.6. Thành phần 3 — Hoán đổi vùng tĩnh (SRS) và phần mở rộng tương phản cùng camera

**Ý tưởng SRS.** Lấy frame x\_t (camera c, ngày d, 17:20) và frame x' cùng camera nhưng ngày d' khác (17:05). Ở những vị trí mà **cả hai** frame đều là tĩnh, thay patch của x\_t bằng patch của x'. Kết quả x̃\_t có đúng các xe của x\_t, nhưng mặt đường, bóng đổ, độ ướt, ánh sáng lấy từ ngày khác. Teacher nhìn x\_t, student nhìn x̃\_t, loss DINO buộc hai biểu diễn giống nhau → mô hình phải **bỏ qua** khác biệt ở vùng tĩnh và dựa vào xe.

So với Background Erasing (CVPR 2021) — trộn mờ một frame tĩnh lên toàn bộ video — SRS không tạo "xe ma" bán trong suốt, vì chỉ hoán đổi đúng những vị trí tĩnh ở cả hai frame. Đây là thứ chỉ làm được nhờ camera cố định: cùng vị trí trên ảnh là cùng một điểm trong cảnh.

**Thuật toán:**

1. **Chọn cặp:** x' cùng camera, cùng chế độ (ngày/IR), khác ngày, slot trong khoảng ±2 giờ (ablation: cùng slot / bất kỳ slot trong chế độ). Cả hai frame đã căn chỉnh (IV.3) và `align_ok`.
2. **Tập hoán đổi:** P\_swap = {p : π\_t(p) < 0.2 và π'(p) < 0.2 và cả hai xác định}, sau khi **loại thêm các patch kề** vùng tiền cảnh của cả hai frame (giãn F\_t ∪ F' một patch) để không chép nửa chiếc xe.
3. **Chọn tỷ lệ:** lấy ngẫu nhiên tập con của P\_swap với tỷ lệ ρ \~ U\[0.3, 1.0\].
4. **Ghép ở mức pixel** trên khung 448 × 256, theo khối 16 × 16; làm mềm biên mỗi khối bằng trộn tuyến tính 4 px để không tạo đường nối sắc — tránh mô hình học dấu vết đường nối.
5. Áp SRS với xác suất p\_srs = 0.5 cho mỗi mẫu, **trước** khi cắt crop và trước AGM. Teacher luôn nhận frame gốc.

**Nghiệm thu nhanh:** trên mọi patch thuộc F\_t, x̃\_t phải trùng từng pixel với x\_t (trừ dải làm mềm 4 px ở biên các khối đã hoán đổi kề bên).

**Phần mở rộng (chỉ giữ nếu thắng trong ablation):**

| Mã | Cơ chế | Cách làm | Rủi ro |
| --- | --- | --- | --- |
| RIC | Tương phản vùng xe trong cùng camera | Batch gồm 8 camera × 8 frame (khác ngày, slot ±2 giờ). Gộp token patch của student theo trọng số π → vector vùng xe z\_i; cặp dương là z của hai view cùng frame; cặp âm là z của frame khác **cùng camera**. InfoNCE, nhiệt độ 0.2, λ = 0.1 | Âm tính giả: hai frame có cảnh xe tương tự. Giảm nhẹ bằng trọng số âm mềm w\_ij = 1 − cos(ẑ\_i, ẑ\_j) tính từ teacher |
| GRL | Đối kháng nhận dạng camera | Một bộ phân loại camera ID trên \[CLS\] qua lớp đảo gradient (kiểu DANN), λ = 0.05 | Có thể xóa luôn thông tin phối cảnh hữu ích cho đếm xe |

**Baseline đối nghịch cần có:** chọn cặp **dương** là frame cùng camera, gần thời điểm (kiểu Focus on the Positives). Giả thuyết của ta là cách này tăng phụ thuộc vào nền và **giảm** chất lượng cho bài toán xe — nếu đúng, đây là một kết quả có giá trị riêng.

### G.7. Loss tổng, siêu tham số và lịch huấn luyện

```latex
\mathcal{L} = \mathcal{L}_{\text{DINO}}^{[CLS]} + \lambda_{\text{iBOT}} \sum_{p \in \mathcal{M}_{\text{AGM}}} \big(1 + \beta\, \pi(p)\big)\, \ell_{\text{iBOT}}(p) + \lambda_{\text{KoLeo}} \mathcal{L}_{\text{KoLeo}} \;\big[+\, \lambda_{\text{RIC}} \mathcal{L}_{\text{RIC}}\big]
```

Với 𝓜\_AGM là tập patch bị che do AGM chọn (G.5); student nhận ảnh đã qua SRS (G.6); mặc định β = 0, λ\_iBOT = 1.0, λ\_KoLeo = 0.1, RIC tắt.

**Siêu tham số khởi điểm** (theo cấu hình kiểu DINOv2, điều chỉnh cho huấn luyện tiếp trên dữ liệu miền — cần kiểm tra lại với repo DINOv3 chính thức):

| Nhóm | Tham số | Giá trị |
| --- | --- | --- |
| Mô hình | Backbone | ViT-B/16 khởi tạo DINOv3 (ablation và thử nhanh: ViT-S/16) |
|  | Đầu DINO, đầu iBOT | Riêng biệt, mỗi đầu 16.384 prototype, khởi tạo mới |
| Crop | Global | 2 crop 224, tỷ lệ diện tích \[0.32, 1.0\] |
|  | Local | 8 crop 96, tỷ lệ \[0.05, 0.32\] (giảm còn 4 nếu thiếu compute) |
| Che | Tỷ lệ | U\[0.1, 0.5\] mỗi ảnh, che 50% số ảnh, chỉ global view của student |
|  | AGM | φ = 0.5, q\_max = 0.6 (mặc định) |
| SRS | Xác suất, cặp | p\_srs = 0.5; cùng camera, khác ngày, slot ±2 giờ |
| TAM | Thống kê | K = 4, d = 64, η = 0.02, η\_w = 0.005, w\_min = 0.15 |
| Tối ưu | AdamW | lr đỉnh 5e-5 (batch 256), layer-wise decay 0.9, weight decay 0.04 → 0.2 (cosine), warmup 5% số bước, grad clip 3.0, drop path 0.1, bf16 |
| Teacher | EMA, nhiệt độ | momentum 0.994 → 1.0 (cosine); nhiệt độ teacher 0.04 → 0.07 trong 10% đầu; student 0.1 |
| Batch | Cấu trúc | 256 = 32 camera × 8 frame (lấy mẫu theo nhóm camera để có cặp SRS và RIC) |
| Độ dài | Số bước | ViT-B: 60.000 bước; ViT-S cho ablation: 20.000 bước — mọi biến thể trong cùng bảng **đúng cùng số bước** |

**Lịch:**

1. **Trước khi train:** khởi tạo TAM offline (G.4); fit PCA; fit GMM ban đầu.
2. **0–10% số bước:** p\_srs tăng tuyến tính 0 → 0.5; φ tăng từ tỷ lệ tự nhiên lên 0.5.
3. **Toàn bộ quá trình:** thống kê TAM cập nhật trực tuyến từ feature đóng băng của mọi frame trong batch; GMM cập nhật mỗi 1.000 bước.

**Chi phí phụ.** Mỗi bước thêm một lượt forward không gradient của DINOv3 ViT-B trên khung 448 × 256 cho mọi frame (và cho frame cặp SRS). Ước tính tăng khoảng 25–40% thời gian mỗi bước — phải đo và báo cáo (G-E1). Phương án giảm: dùng ViT-S đóng băng cho TAM (ablation G-AB2), hoặc cache feature PCA-64 của các frame huấn luyện xuống đĩa (khoảng 57 KB/frame ở fp16).

### G.8. Spec cho agent

**Thư mục `directionG_camera_ssl/`** — dùng lại `common/` (IV.2–IV.5) và phần DINO/iBOT của H1.

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
| `eval_frozen.py` | Probe như IV.6 + probe dày |
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

Cài `update` theo dạng vector hóa: gom mọi (cid, p) của batch, dùng `scatter_add` cho cập nhật tâm và tần suất; không vòng lặp Python theo vị trí.

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

**Config mẫu** `configs/exp/g_full.yaml`:

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

**Dữ liệu đồ chơi (`toy.py`) để test trước khi chạy thật:** nền là ảnh có kết cấu cố định với 2 chế độ sáng (sáng/tối, mỗi frame chọn ngẫu nhiên); "xe" là hình chữ nhật màu ngẫu nhiên, kích thước ngẫu nhiên, phủ trung bình 20% diện tích ở vùng "đường"; sinh 50 "ngày" × 40 frame.

**Nghiệm thu:**

1. Trên dữ liệu đồ chơi, AUROC của a\_t với mask hình chữ nhật ≥ 0.95 ở cả hai chế độ sáng.
2. Thêm một hình chữ nhật **cùng màu** ở **cùng vị trí** trong 90% frame → vị trí đó bị coi là tĩnh (đúng giới hạn đã nêu); test xác nhận hành vi này.
3. `PositionStats.update` vector hóa cho kết quả trùng (sai số < 1e-4) với cài đặt vòng lặp tham chiếu trên 1.000 mẫu.
4. `agm_sample`: số patch che đúng bằng `n_mask`; số patch xe bị che ≤ q\_max·|F|; khi F rỗng, phân phối vị trí che không khác che đều (kiểm định chi-bình-phương trên 10.000 lần lấy mẫu).
5. `static_region_swap`: pixel trong F\_t (trừ dải làm mềm) trùng 100% với x; không ô nào thuộc F\_t ∪ F' (đã giãn) bị hoán đổi.
6. `map_to_crop`: với ảnh đồ chơi có một ô sáng, sau crop và lật, vị trí π lớn nhất trùng vị trí ô sáng.
7. Huấn luyện thử 2.000 bước trên dữ liệu đồ chơi với ViT-S: loss giảm, không NaN, IoU attention với hình chữ nhật tăng so với bước 0.

### G.9. Kế hoạch thực nghiệm

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

**Thí nghiệm** (ablation chạy ViT-S, 20.000 bước, trừ khi ghi khác; kết quả chính ViT-B, 3 seed):

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| G-M1 | M | TAM định vị xe tốt đến đâu? (RQ1) | TAM vs Δ pixel (median r0), Δ với nền đã làm sạch r2 (Hướng A), MaskCut, attention \[CLS\] của DINOv3, khoảng cách tới feature trung bình (một trạng thái), khoảng cách láng giềng gần nhất kiểu PatchCore | AUROC, AP mức patch (nhãn = patch có ≥ 30% diện tích là xe, từ mask audit-test); tách theo cao điểm, đêm, mưa | P0 |
| G-M2 | M | Biểu diễn tốt hơn? (RQ2) | B0–B7 vs G đầy đủ; feature đóng băng | Mức ùn tắc (linear, kNN): macro-F1, QWK; đếm few-shot (ridge): MAE; **probe dày**: phân loại tuyến tính từng token patch xe / không xe, train trên audit-train, IoU trên audit-test | P0 |
| G-M3 | M | Lợi ích có chuyển sang bài toán khác? | Thay DINOv3 bằng backbone G trong H3 (đếm), B (mô hình cuối), E | Metric của từng hướng | P1 |
| G-M4 | M | Hội tụ nhanh hơn? | Đánh giá kNN mỗi 10% quá trình cho B1, B2, B3, G | Metric theo giờ GPU | P1 |
| G-D1 | Chẩn đoán | Còn dựa vào nền? (RQ3) | Logistic regression đoán camera ID từ \[CLS\], trên frame ngày khác của camera train | Độ chính xác (thấp hơn = ít dựa nền), vẽ cùng metric G-M2 trên một biểu đồ phân tán | P0 |
| G-D2 | Chẩn đoán | Bất biến với nền? | Biến đổi **không dùng lúc train**: hoán đổi vùng tĩnh từ slot lệch ±6 giờ; đổi gamma riêng vùng tĩnh | Cosine \[CLS\] trước/sau; tỷ lệ giữ nguyên dự đoán mức ùn tắc | P0 |
| G-D3 | Chẩn đoán | Có thật sự nhìn xe? (RQ4) | Xóa xe theo **mask người gán** (audit-test): thay pixel xe bằng pixel cùng vị trí của frame vắng nhất cùng camera, cùng chế độ | Mức giảm của số xe dự đoán và mức ùn tắc dự đoán (càng giảm mạnh càng tốt) | P0 |
| G-D4 | Chẩn đoán | Attention dồn vào đâu? | Tỷ lệ khối lượng attention \[CLS\] rơi vào mask xe | % | P1 |
| G-AB1 | AB | Thành phần nào đóng góp? | B1 / + AGM / + SRS / + AGM + SRS / + RIC / + GRL | G-M2 và G-D1–D3 | P0 |
| G-AB2 | AB | Thiết kế TAM | K ∈ {1, 2, 4, 8}; tiêu chí tĩnh: chỉ tần suất / tần suất + độ chặt; khoảng cách: tới trạng thái tĩnh gần nhất / láng giềng gần nhất / trung bình; d ∈ {32, 64, 128, 768}; nguồn feature: ViT-S đóng băng / ViT-B đóng băng / teacher EMA; thống kê theo chế độ / theo nhóm giờ | G-M1, G-M2 | P0 (K, khoảng cách, nguồn) — P1 (còn lại) |
| G-AB3 | AB | Ngưỡng tiền cảnh | GMM / cố định top 20% / ngưỡng tối ưu trên audit-val (tham chiếu) | G-M1 F1; G-M2 | P1 |
| G-AB4 | AB | Nên che bao nhiêu xe? | 4 chính sách G.5 + lưới φ × q\_max (1 seed) | G-M2; **bề mặt hiệu năng** | P0 |
| G-AB5 | AB | Thiết kế SRS | Tắt / trộn toàn ảnh (B5) / hoán đổi patch **ngẫu nhiên** không theo TAM (control) / hoán đổi vùng tĩnh; cửa sổ cặp: cùng slot / ±2 giờ / bất kỳ; làm mềm biên 0 / 4 px | G-M2, G-D2 | P0 (3 mục đầu) |
| G-AB6 | AB | Trọng số loss iBOT theo π | β ∈ {0, 1, 2} | G-M2 | P2 |
| G-R1 | R | Cần bao nhiêu ngày dữ liệu? | Số ngày dùng cho khởi tạo và cập nhật TAM: 1 / 3 / 7 / 14 / tất cả; RS-3 (thưa mẫu) | G-M1, G-M2 | P0 |
| G-R2 | R | Tổng quát? | Probe với G-region, G-day2night | G-M2 | P1 |
| G-R3 | R | Frame xấu | FCS lên probe đóng băng | Suy giảm tương đối | P1 |
| G-R4 | R | Không căn chỉnh camera | Bỏ bước căn chỉnh IV.3 | G-M1, G-M2 | P1 |
| G-R5 | R | Camera hay kẹt | Báo cáo riêng nhóm camera có tỷ lệ vị trí "không có trạng thái tĩnh" cao nhất (top 10%) | G-M1, G-M2 | P0 |
| G-S1 | S | Quy mô | 10 / 25 / 50 / 100% camera train; ViT-S vs ViT-B | G-M2 | P1 |
| G-E1 | E | Chi phí | Thời gian/bước, giờ GPU tổng, bộ nhớ thống kê; suy luận như ViT thường | — | P0 |
| G-Q1 | Q | TAM trông thế nào? | TAM, Δ, MaskCut trên cùng frame: cao điểm, đêm, mưa; ví dụ SRS | Hình | P0 |
| G-Q2 | Q | Attention | B1, B3, G cạnh nhau | Hình | P0 |
| G-Q3 | Q | Lỗi | Làn luôn kẹt, xe đậu cố định, camera rung | Hình + phân tích | P0 |

**Lưu ý công bằng cho G-D2:** G được train với SRS nên sẽ "có lợi" nếu đánh giá bằng đúng phép SRS. Vì vậy G-D2 chỉ dùng biến đổi không có lúc train. TAM dùng để tạo biến đổi trong G-D2 lấy từ DINOv3 đóng băng — giống nhau cho mọi mô hình được so.

**Bảng/hình cho paper G:** Bảng 1 = G-M1; Bảng 2 = G-M2 (mọi baseline); Bảng 3 = G-D1–D3; Bảng 4 = G-AB1; Bảng 5 = G-AB2; Hình = bề mặt φ × q\_max (G-AB4), biểu đồ phân tán camera ID vs metric xe (G-D1), G-Q1, G-Q2.

### G.10. Rủi ro, cổng quyết định, lịch và venue

**Ba cổng quyết định** — không qua cổng thì không tiêu thêm compute cho bước sau:

1. **Cổng 1 (trước mọi lần pretrain, khoảng tuần 3–4):** chạy G-M1 trên audit-val. Điều kiện: AUROC của TAM không thấp hơn Δ pixel ở tổng thể, và cao hơn ít nhất 0.05 ở lát cao điểm hoặc đêm. Nếu trượt: thử các biến thể G-AB2; vẫn trượt thì dùng nguồn tiền cảnh khác (Δ với nền r2 của Hướng A, hoặc MaskCut) và giữ SRS.
2. **Cổng 2 (ViT-S, 20.000 bước):** G (AGM + SRS) phải vượt B1 và B2 trên kNN mức ùn tắc và probe dày, với mức chênh lớn hơn độ lệch chuẩn giữa 2 seed của B1. Nếu trượt: kết quả bề mặt φ × q\_max và các chẩn đoán vẫn có giá trị — gộp thành phân tích trong paper A hoặc paper P-1, không viết paper G riêng.
3. **Cổng 3 (sau kết quả ViT-B):** G-D1–D3 phải cho thấy giảm phụ thuộc nền. Nếu G thắng trên bài toán nhưng chẩn đoán không đổi, câu chuyện "chống đường tắt nền" không đứng — viết lại luận điểm thành "che có hướng dẫn từ chuỗi ảnh".

**Rủi ro:**

| Rủi ro | Dấu hiệu | Cách giảm |
| --- | --- | --- |
| TAM hỏng ở làn luôn kẹt | Tỷ lệ vị trí không có trạng thái tĩnh cao | Báo cáo riêng (G-R5); AGM tự lùi về che ngẫu nhiên ở đó |
| Mô hình học dấu vết đường nối của SRS | G-D2 tốt nhưng G-M2 không tăng | Làm mềm biên; ablation làm mềm 0 / 4 px; control hoán đổi ngẫu nhiên |
| Lợi ích chỉ do huấn luyện thêm trên dữ liệu miền | G ≈ B1 | B1 đã kiểm soát yếu tố này — nếu G ≈ B1 thì đó là kết luận trung thực |
| Chi phí forward thêm cho TAM | Thời gian/bước tăng > 40% | TAM bằng ViT-S, hoặc cache feature PCA-64 |
| Giấy phép DINOv3 | Muốn công bố trọng số đã huấn luyện tiếp | Đọc điều khoản DINOv3 trước; nếu không được phép công bố trọng số, công bố code và cách tái tạo |
| Chỉ một thành phố | Reviewer hỏi tổng quát | Probe đếm xe trên dữ liệu Đà Nẵng bằng feature của backbone (không cần TAM lúc suy luận) |

**Lịch khoảng 5 tháng** (sau khi nền móng IV.2–IV.5 đã có):

1. **Tháng 1:** `toy.py`, `tam.py`, `init_tam.py`, `eval_tam.py`; chạy G-M1 → **Cổng 1**. Cần mask xe của tập audit.
2. **Tháng 2:** `masking.py`, `srs.py`, `sampler.py`; ablation ViT-S: G-AB1, G-AB2, G-AB4, G-AB5 → **Cổng 2**.
3. **Tháng 3:** ViT-B, cấu hình chốt, 3 seed; baseline B0–B6 cùng compute.
4. **Tháng 4:** chẩn đoán G-D1–D4 (**Cổng 3**), robustness G-R1–R5, chuyển giao G-M3.
5. **Tháng 5:** hình, bảng, viết paper.

**Quan hệ với các hướng khác:** nếu G qua Cổng 2, backbone G thay DINOv3 gốc trong H3, B, E và encoder của Hướng A — mỗi paper khác có thêm một dòng "backbone G" trong bảng ablation. H1 dừng phát triển; FAM-Δ chỉ còn là baseline B3.

**Venue:** Pattern Recognition, IEEE TIP, CVIU (nếu nhấn mạnh phương pháp tự học); IEEE T-ITS, EAAI (nếu nhấn mạnh ứng dụng giao thông).

## IV. Đặc tả kỹ thuật cho coding agent

Phần này viết để giao thẳng cho agent. Mỗi mục nêu đầu vào, đầu ra, hàm cần có, tham số mặc định và **tiêu chí nghiệm thu** (test phải qua). Agent làm theo thứ tự IV.2 → IV.5 trước (nền móng dùng chung), sau đó mới đến từng hướng. Khi phần IV mâu thuẫn với các phần trước, **phần IV được ưu tiên**.

### IV.1. Nguyên tắc, cấu trúc repo và quy ước

**Nguyên tắc bắt buộc cho agent:**

1. Không hardcode đường dẫn, siêu tham số hay ngưỡng trong code — mọi thứ đi qua file config YAML.
2. Mọi bước tiền xử lý tốn thời gian ghi kết quả ra file (parquet, npz, png) và có thể chạy lại từng phần; không tính lại mỗi lần train.
3. Mọi hàm ngẫu nhiên nhận `rng: np.random.Generator` hoặc seed từ config; không dùng trạng thái ngẫu nhiên toàn cục.
4. Tập test (cụm camera test, audit-test, gold-test) chỉ được đọc bởi script `evaluate_*.py` cuối cùng. Code train, chọn siêu tham số, chọn checkpoint và tiêu chí dừng **không được** import đường dẫn test.
5. Mỗi lần chạy lưu: config đầy đủ, git commit hash, seed, phiên bản thư viện, thời gian GPU.

**Cấu trúc repo đề xuất** (mở rộng từ `g:/nckh/DINO/` hiện có):

```text
DINO/
├── configs/
│   ├── data.yaml                 # đường dẫn gốc, độ phân giải, múi giờ
│   ├── splits.yaml               # bán kính cụm, tỷ lệ, seed
│   └── exp/                      # một file cho mỗi thí nghiệm
├── common/
│   ├── io.py                     # đọc ảnh, đọc/ghi parquet
│   ├── index.py                  # build_frame_index, build_background_index
│   ├── quality.py                # frame hỏng, chế độ IR, độ sáng
│   ├── align.py                  # estimate_shift, warp
│   ├── static_mask.py            # vùng tĩnh từ phương sai thời gian
│   ├── photometric.py            # color_align
│   ├── splits.py                 # chia theo cụm địa lý
│   ├── degrade.py                # BDB
│   ├── reliability.py            # r_i, đọc bản đồ σ̄ của Hướng A
│   ├── metrics.py
│   ├── stats.py                  # bootstrap theo cụm, Wilcoxon, Holm
│   └── subtraction.py            # đã có: tính Δ
├── annotation/
│   ├── guidelines/               # hướng dẫn gán nhãn (markdown)
│   ├── export_tasks.py           # xuất task cho CVAT/Label Studio
│   └── import_labels.py          # chuyển về schema IV.2
├── direction1_bg_guided_dino/    # H1
├── direction2_scene_decomposition/  # Hướng A (H2 nâng cấp)
├── direction3_foreground_enhanced_counting/  # H3
├── direction4_temporal_density/  # H4
├── directionB_weak_supervision/
├── directionC_anomaly/
├── directionD_forecasting/
├── directionE_bg_conditioning/
├── scripts/                      # chạy pipeline, sinh bảng LaTeX, vẽ hình
├── tests/                        # pytest cho common/ và từng hướng
└── outputs/{exp_name}/{run_id}/  # checkpoint, log, metric json
```

**Quy ước chung:**

| Mục | Quy ước |
| --- | --- |
| Ngôn ngữ, thư viện | Python 3.10+, PyTorch 2.x, timm hoặc repo DINOv3 chính thức, OpenCV, pandas + pyarrow, scikit-learn, scipy |
| Ảnh trong bộ nhớ | numpy `uint8` H×W×3 thứ tự **RGB** (OpenCV đọc BGR → đổi ngay khi đọc); tensor float32 trong \[0, 1\], C×H×W |
| Độ phân giải làm việc | 448 × 256 (W × H) mặc định; mọi mask lưu ở độ phân giải gốc, resize khi dùng (nearest cho mask) |
| Thời gian | Lưu `ts_utc` (timestamp có múi giờ); `slot` và `day` tính theo **Asia/Ho\_Chi\_Minh** |
| ID | `camera_id` = chuỗi `stt` hiện có; `frame_id` = `{camera_id}_{yyyymmddHHMMSS}` theo giờ Việt Nam |
| Log | Ghi metric ra `metrics.json` + TensorBoard hoặc W&B |
| Bảng kết quả | Script sinh bảng LaTeX trực tiếp từ `metrics.json`, không chép tay số |

### IV.2. Schema dữ liệu

Tất cả bảng lưu parquet trong `data/index/`. Agent viết `common/index.py` để sinh các bảng này từ cấu trúc thư mục hiện có (`output/{stt}_{timestamp}.jpg`, `traffic_backgrounds/route_{stt}/background_slot_{slot}h.jpg`).

**`cameras.parquet`** — một dòng mỗi camera.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| camera\_id | str | stt |
| lat, lon | float | Tọa độ (gán tay nếu nguồn không có) |
| intersection\_id | str, nullable | Nút giao mà camera nhìn vào; camera cùng nút giao có cùng giá trị |
| district | str | Quận/huyện |
| road\_type | str | `arterial` / `intersection` / `local` |
| heading\_deg | float, nullable | Hướng nhìn, nếu có |
| width, height | int | Độ phân giải gốc |
| cluster\_id | int | Do `splits.py` sinh (IV.3) |
| split | str | `train` / `val` / `test`, theo cluster\_id |

**`frames.parquet`** — một dòng mỗi frame.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| frame\_id | str | `{camera_id}_{yyyymmddHHMMSS}` (giờ VN) |
| camera\_id | str |  |
| ts\_utc | timestamp\[tz=UTC\] |  |
| day | date | Ngày theo giờ VN |
| slot | int8 | Giờ 0–23 theo giờ VN |
| weekday | int8 | 0 = thứ Hai |
| path | str | Đường dẫn tương đối |
| is\_corrupt | bool | `quality.py` |
| is\_ir | bool | Ảnh hồng ngoại / đơn sắc |
| brightness | float | Trung bình kênh V (HSV), 0–255 |
| dx, dy, align\_ok | float, float, bool | Độ lệch so với background cùng slot (`align.py`) |
| cam\_epoch | int | Giai đoạn camera, tăng 1 mỗi khi camera bị chỉnh góc vĩnh viễn |
| gap\_prev\_s | float | Số giây đến frame trước của cùng camera |

**`backgrounds.parquet`** — một dòng mỗi ảnh nền.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| camera\_id, slot | str, int8 |  |
| mode | str | `day` / `ir` |
| cam\_epoch | int |  |
| version | str | `r0_median` (hiện có), `r1`, `r2`… do Hướng A sinh |
| path | str |  |
| sigma\_path | str, nullable | Bản đồ σ̄ (Hướng A), npz float16 |

**Mask theo camera** (`data/masks/{camera_id}/`): `road.png` — lòng đường xe chạy, **không gồm vỉa hè**, nhị phân 0/255, độ phân giải gốc; `static.png` — vùng tĩnh tự sinh (IV.3); `sidewalk.png` (tùy chọn) — vỉa hè, dùng cho phân tích.

**Nhãn** (`data/labels/`):

| File | Định dạng | Nội dung |
| --- | --- | --- |
| `audit_vehicle_masks.json` | COCO instance segmentation | Mask xe trên frame audit; category: motorcycle, car, bus, truck, bicycle, pedestrian |
| `audit_bg_errors.json` | COCO, category: ghost, shadow, wet, glare, misalign | Polygon lỗi trên ảnh background |
| `gold_congestion.parquet` | frame\_id, annotator\_id, level (0–3), uncertain (bool), ts\_annotated | Một dòng mỗi lượt gán của mỗi người |
| `gold_congestion_final.parquet` | frame\_id, level, agreement | Sau phân xử (IV.5) |
| `gold_occupancy.parquet` | frame\_id, occupancy (float, trong road mask) | Tính từ mask xe ∩ road |
| `counts.parquet` | frame\_id, n\_motorcycle, n\_car, n\_bus\_truck, n\_total | Nhãn đếm H3 hiện có, chuyển về schema này |
| `incidents.parquet` | event\_id, camera\_id, type, t\_start, t\_end, source, verified | Hướng C |

**Tiêu chí nghiệm thu IV.2:** (1) mọi `frame_id` duy nhất; (2) `slot` tính lại từ `ts_utc` khớp 100%; (3) mọi frame có background tương ứng (camera\_id, slot, mode, cam\_epoch) hoặc được đánh dấu thiếu; (4) script in thống kê: số frame/camera, phân phối `gap_prev_s` (median, p90), tỷ lệ corrupt, tỷ lệ IR theo slot.

### IV.3. Module chung

Các ngưỡng dưới đây là giá trị khởi điểm, đặt trong `configs/data.yaml`; agent phải in ra phân phối thực tế để người duyệt chỉnh lại.

#### `common/quality.py`

```python
def is_corrupt(img: np.ndarray, prev_hashes: list[int]) -> tuple[bool, str]:
    """Trả về (hỏng?, lý do). Hỏng nếu một trong các điều kiện:
    - std của ảnh xám < 5                         -> 'flat'
    - > 90% pixel có giá trị < 5 hoặc > 250        -> 'saturated'
    - dHash 64-bit trùng 3 frame liền trước        -> 'frozen'
    """

def is_ir(img: np.ndarray) -> bool:
    """Ảnh hồng ngoại/đơn sắc: mean kênh S (HSV) < 12 và
    std(R-G) < 3 và std(G-B) < 3."""
```

Nghiệm thu: trên 200 frame người gán (hỏng/không, IR/không), precision và recall ≥ 0.95.

#### `common/static_mask.py` — sửa lỗi số 2 ở phần rà soát

Vùng tĩnh **không** được lấy là "phần bù của road mask", vì vỉa hè có xe máy đậu và người đi lại. Tính tự động từ phương sai theo thời gian:

1. Với mỗi camera và mỗi `mode` (day/ir), lấy 200 frame không hỏng, rải trên ít nhất 10 ngày và mọi slot thuộc mode đó.
2. Chuyển xám, làm mờ Gaussian 5×5, tính độ lệch chuẩn theo thời gian từng pixel → `std_map`.
3. Loại road mask đã giãn 15 px và sidewalk mask (nếu có).
4. `static` = pixel còn lại có `std_map` thuộc 30% thấp nhất; mở hình thái 5×5.
5. `static_textured` = `static` ∩ (độ lớn gradient Sobel của background > median) — dùng cho căn chỉnh, vì bầu trời tĩnh nhưng không có kết cấu.
6. Nếu diện tích `static_textured` < 2% ảnh → `static_ok = False`; camera đó bỏ qua bước căn chỉnh và r\_i (gán r\_i = 1, ghi log).

Đầu ra: `data/masks/{camera_id}/static_{mode}.png`, `static_textured_{mode}.png`. Nghiệm thu: người duyệt xem lưới ảnh của 30 camera ngẫu nhiên, không có camera nào mà vùng tĩnh phủ lên vỉa hè có xe đậu.

#### `common/align.py`

```python
def estimate_shift(frame, bg, mask_textured) -> tuple[float, float, float]:
    """Phase correlation (cv2.phaseCorrelate) trên ảnh xám, nhân với mask
    và cửa sổ Hann trong bounding box của mask. Trả về (dx, dy, response)."""

def align_ok(dx, dy, response, max_shift=16, min_response=0.1) -> bool

def warp(frame, dx, dy) -> np.ndarray     # cv2.warpAffine, viền BORDER_REFLECT

def detect_cam_epochs(frames_df) -> pd.Series:
    """Tính median (dx, dy) theo ngày cho từng camera. Khi median nhảy > 8 px
    và giữ nguyên ≥ 2 ngày liên tiếp -> bắt đầu cam_epoch mới."""
```

Quy tắc dùng: |shift| ≤ 4 px → bỏ qua; 4–16 px → warp frame về background; > 16 px hoặc response < 0.1 → `align_ok = False`, loại khỏi mọi thí nghiệm dùng background. Nghiệm thu: trên ảnh tự dịch đã biết (±1…16 px), sai số ước lượng ≤ 0.5 px trong 95% trường hợp.

#### `common/photometric.py`

```python
def color_align(src, ref, mask, n_iter=5) -> np.ndarray:
    """Với từng kênh RGB (float [0,1]), tìm a, b tối thiểu hóa
    Huber(a*src + b - ref) trên pixel thuộc mask (IRLS, delta=0.05).
    Kẹp a trong [0.5, 2.0], b trong [-0.3, 0.3]. Trả về ảnh đã chỉnh, kẹp [0,1]."""
```

Mask dùng `static` của camera. Nghiệm thu: ảnh bị nhân gain/bias ngẫu nhiên trong khoảng trên được khôi phục với sai số trung bình tuyệt đối < 0.01.

#### `common/reliability.py`

```python
def static_reliability(frame, bg, static_mask, kappa) -> float:
    """Δ (công thức LAB hiện có trong subtraction.py) trung bình trên static_mask,
    sau khi warp và color_align. r = exp(-mean_delta / kappa)."""

def fit_kappa(train_values: np.ndarray) -> float:
    """kappa = p90(mean_delta trên frame train) / ln 2, để r = 0.5 tại phân vị 90."""

def load_sigma_map(camera_id, slot, mode, version) -> np.ndarray | None
    # bản đồ σ̄ của Hướng A, None nếu chưa có
```

#### `common/splits.py` — sửa lỗi số 1 ở phần rà soát

```python
def make_clusters(cameras_df, eps_m=150) -> pd.Series:
    """Nếu có intersection_id: cluster = intersection_id.
    Camera còn lại: DBSCAN (metric haversine, eps = 150 m, min_samples = 1)."""

def assign_splits(cameras_df, ratios=(0.7, 0.1, 0.2), seed=0) -> pd.Series:
    """Gán nguyên cụm vào train/val/test, phân tầng theo road_type và district,
    mục tiêu tỷ lệ theo SỐ CAMERA. Thử 1.000 hoán vị ngẫu nhiên, chọn cái có
    sai lệch tỷ lệ nhỏ nhất. Lưu configs/splits.json (cố định cho mọi hướng)."""
```

Nghiệm thu: (1) không cụm nào nằm ở hai split; (2) khoảng cách nhỏ nhất giữa một camera train và một camera test > 150 m — in giá trị; (3) bảng phân phối road\_type và district theo split, kiểm định chi-bình-phương không bác bỏ ở mức 0.05; (4) `splits.json` được mọi hướng đọc chung, không hướng nào tự chia lại.

Riêng Hướng D chia thêm theo **thời gian** (IV.11). Few-shot ở H3/E lấy mẫu con **bên trong** split train, không đổi split.

### IV.4. Module BDB — `common/degrade.py`

Đặc tả chính xác cho Background Degradation Benchmark (mục I.1), để mọi hướng dùng **cùng một** bộ nhiễu.

```python
@dataclass
class DegradeContext:
    camera_id: str
    slot: int
    mode: str                       # 'day' / 'ir'
    road_mask: np.ndarray           # H x W bool
    bg_lookup: Callable[[str, int, str], np.ndarray | None]  # (camera, slot, mode) -> bg
    vehicle_bank: VehicleBank       # chỉ crop từ audit-TRAIN
    other_cameras: list[str]        # camera thuộc cụm khác, cùng mode
    height_model: tuple[float, float]  # chiều cao xe = a*y + b (pixel), theo camera

def degrade(bg, kind, severity, ctx, seed) -> tuple[np.ndarray, np.ndarray | None]:
    """Trả về (bg_bị_làm_xấu, ghost_mask hoặc None).
    seed = hash((frame_id hoặc camera_id-slot, kind, severity, global_seed)) -> mọi
    phương pháp nhận đúng cùng một ảnh nền bị làm xấu."""
```

**Bảng mức độ (severity 1 → 5):**

| kind | Tham số theo mức 1 / 2 / 3 / 4 / 5 | Cách làm |
| --- | --- | --- |
| `slot_shift` | lệch 1 / 2 / 4 / 8 / 12 giờ | Lấy bg cùng camera, slot (s ± lệch) mod 24, dấu ngẫu nhiên; nếu thiếu thì lấy slot gần nhất có sẵn; mức 5 thường là đảo ngày ↔ đêm |
| `geometric` | dịch 2 / 4 / 8 / 16 / 32 px; xoay 0.25 / 0.5 / 1 / 2 / 3 độ | Hướng dịch và chiều xoay ngẫu nhiên; `warpAffine` quanh tâm ảnh, `BORDER_REFLECT` |
| `ghost` | phủ 5 / 10 / 20 / 30 / 50% diện tích road mask | Dán crop xe (có alpha) từ `vehicle_bank`, scale theo `height_model` tại tọa độ y; đặt ngẫu nhiên trong road mask, chồng nhau tối đa 30%, dừng khi đạt độ phủ; hòa trộn `cv2.seamlessClone(NORMAL_CLONE)`; trả về `ghost_mask` |
| `photometric` | gamma 1.1 / 1.25 / 1.4 / 1.6 / 2.0 (hoặc nghịch đảo); gain ±5 / 10 / 20 / 30 / 40% | Chiều ngẫu nhiên; áp lên ảnh float rồi kẹp \[0,1\] |
| `noise_jpeg` | Gaussian σ = 5 / 10 / 20 / 30 / 40 (thang 0–255) rồi JPEG chất lượng 90 / 70 / 50 / 30 / 10 | Thứ tự: nhiễu trước, nén sau |
| `other_camera` | một mức | bg của camera ngẫu nhiên trong `other_cameras`, cùng slot và mode, resize về kích thước đích |
| `drop` | một mức | Không có background: H3 → kênh Δ = 0; E → token rỗng; A chế độ (b) → prior = 0 |

**Giao thức đánh giá BDB** (script `scripts/run_bdb.py --method X`):

1. Với mỗi `kind` và `severity`, chạy suy luận trên toàn bộ tập test, lưu metric vào `bdb/{method}/{kind}_{severity}.json`.
2. Mức 0 = background gốc. Baseline No-bg chạy một lần (không phụ thuộc kind).
3. Chỉ số tóm tắt cho mỗi `kind`: **suy giảm tương đối trung bình** D = (1/5) Σ\_s (m\_s − m\_0) / m\_0 với metric "càng nhỏ càng tốt" (đổi dấu với metric "càng lớn càng tốt"); và **mức cắt** s\* = mức nhỏ nhất mà phương pháp tệ hơn No-bg baseline với khoảng tin cậy 95% không chồng lấn (bootstrap theo cụm, IV.5). Không có s\* thì ghi "không cắt" — đây là kết quả mong muốn.
4. Hình chuẩn: lưới 2 × 3 ô (mỗi `kind` một ô), trục x = severity 0–5, trục y = metric chính, mỗi đường một phương pháp có dải tin cậy, đường ngang nét đứt = No-bg baseline.

**Dùng lúc train** (config `degrade_train`): với xác suất `p_degrade = 0.4` áp một kind ngẫu nhiên (trừ `other_camera`) ở severity ngẫu nhiên 1–3; với xác suất `p_drop = 0.25` dùng `drop`. Ở lúc train dùng seed ngẫu nhiên, không cố định.

**Nghiệm thu:** (1) cùng seed → ảnh giống hệt từng byte; (2) `ghost` đạt độ phủ trong ±2 điểm % so với mục tiêu và `ghost_mask` khớp vùng dán; (3) `geometric` dịch đúng số pixel (kiểm bằng `estimate_shift`); (4) `vehicle_bank` không chứa crop nào từ camera val/test — assert khi khởi tạo.

### IV.5. Metric, thống kê và giao thức gán nhãn

#### `common/metrics.py` — mọi hướng dùng chung, không tự viết lại

| Nhóm | Hàm | Định nghĩa và lưu ý cài đặt |
| --- | --- | --- |
| Đếm | `mae`, `rmse`, `rel_mae` | rel\_mae = MAE / trung bình nhãn; báo cáo thêm theo từng loại xe |
| Mức ùn tắc (có thứ tự) | `macro_f1`, `qwk`, `confusion` | QWK = `cohen_kappa_score(weights='quadratic')` |
| Mask | `iou`, `f1`, `boundary_f` | Ngưỡng 0.5; tính trong road mask; boundary F với dung sai 2 px ở độ phân giải làm việc |
| Phát hiện lỗi prior | `pixel_auroc`, `pixel_ap` | Gộp pixel trong road mask của từng camera, rồi bootstrap theo cụm |
| Chất lượng ảnh | `masked_psnr`, `masked_lpips`, `kid` | LPIPS (AlexNet) trên bounding box của vùng che, ngoài mask gán bằng ảnh tham chiếu; dùng **KID thay cho FID** khi ít hơn 2.000 ảnh vì FID lệch mạnh với mẫu nhỏ |
| Dự báo | `mae_h`, `rmse_h`, `onset_prf`, `lead_time` | Theo từng tầm h; khởi phát đúng nếu báo trong cửa sổ \[t\_onset − h, t\_onset\] |
| Sự cố | `event_prf`, `delay`, `fa_per_cam_day` | Gộp báo động liên tiếp cách nhau < 2 bước thành một; báo động trúng nếu rơi trong \[t\_start, t\_end\] |

Mọi hàm nhận `group_ids` (cluster\_id) để phục vụ bootstrap. Nghiệm thu: so khớp với sklearn/torchmetrics trên dữ liệu ngẫu nhiên, sai khác < 1e-6.

#### `common/stats.py`

```python
def cluster_bootstrap_ci(metric_fn, y_true, y_pred, clusters, B=2000, alpha=0.05, seed=0):
    """Lấy mẫu lại CỤM (có hoàn lại), tính metric trên dữ liệu gộp, trả về
    (ước lượng, cận dưới, cận trên) theo phân vị."""

def paired_test(per_cluster_a, per_cluster_b) -> float:
    """Wilcoxon signed-rank trên hiệu theo cụm; trả về p-value."""

def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Hiệu chỉnh Holm–Bonferroni trong một 'họ' so sánh (một bảng kết quả)."""
```

**Quy tắc báo cáo:**

1. 3 seed cho mỗi cấu hình. Metric theo cụm lấy trung bình qua 3 seed, rồi mới bootstrap. Báo cáo: ước lượng \[CI 95%\] và độ lệch chuẩn giữa seed.
2. So sánh với baseline mạnh nhất: Wilcoxon theo cụm, hiệu chỉnh Holm trong mỗi bảng. Chỉ dùng chữ "vượt trội" khi p sau hiệu chỉnh < 0.05.
3. **Ngân sách chỉnh siêu tham số như nhau** cho phương pháp đề xuất và mọi baseline học được: 20 lần thử Optuna (TPE) trên tập val, cùng số epoch tối đa; công bố không gian tìm kiếm trong phụ lục.
4. **Kiểm tra cỡ mẫu sau khi gán nhãn:** nếu nửa độ rộng CI 95% của metric chính vượt ngưỡng — IoU ±0.03, macro-F1 ±0.04, AUROC ±0.03, MAE chiếm dụng ±0.02 — thì gán thêm trước khi chạy thí nghiệm cuối.

#### Giao thức gán nhãn (`annotation/guidelines/`)

**Công cụ:** CVAT hoặc Label Studio; `export_tasks.py` xuất task, `import_labels.py` chuyển về schema IV.2.

**Định nghĩa mức ùn tắc** (người gán xem 3 frame liên tiếp t−1, t, t+1 để thấy chuyển động, nếu có):

| Mức | Tên | Mô tả trực quan |
| --- | --- | --- |
| 0 | Thông thoáng | Xe di chuyển tự do, khoảng trống giữa các xe lớn hơn chiều dài một xe |
| 1 | Trung bình | Xe dày nhưng vẫn còn khoảng trống rõ ràng, xe vẫn di chuyển |
| 2 | Chậm | Xe sát nhau, khoảng trống nhỏ, di chuyển chậm hoặc nhích từng đoạn |
| 3 | Kẹt | Lòng đường gần như kín, xe đứng yên qua các frame |

Trường hợp biên phải có trong hướng dẫn, mỗi trường hợp kèm ít nhất 3 ảnh ví dụ: đèn đỏ (đứng yên nhưng không kẹt — gán theo mật độ, đánh dấu `uncertain`), ban đêm không nhìn rõ, mưa, camera bị che một phần.

**Quy trình:**

1. **Đào tạo:** mọi người gán cùng 50 frame, họp thảo luận chỗ bất đồng, cập nhật hướng dẫn.
2. **Thí điểm:** 100 frame mới, đo Fleiss' κ. Yêu cầu κ ≥ 0.6; chưa đạt thì lặp lại bước 1.
3. **Gán chính thức:** mỗi frame gold do 3 người gán độc lập; nhãn cuối = đa số; hòa → người phân xử thứ tư; lưu cột `agreement`.
4. **Mask xe và polygon ghost:** 1 người gán + 1 người duyệt; 10% (tối thiểu 30 ảnh) gán đôi để báo cáo IoU giữa người gán (yêu cầu ≥ 0.8 với mask xe).
5. **Lấy mẫu phân tầng:** theo split, nhóm giờ (đêm / thấp điểm / cao điểm) và mức đông ước lượng bằng số box detector (3 khoảng phân vị) — tránh tập gold toàn ảnh dễ.
6. **Mù:** khi người duyệt kết quả của nhiều phương pháp (ví dụ ứng viên sự cố ở Hướng C), trộn ngẫu nhiên và ẩn nguồn gốc.

### IV.6. Spec H1 — BG-Guided DINO

**File:** `direction1_bg_guided_dino/` gồm `dataset.py`, `masking.py` (mới), `models.py`, `losses.py`, `train.py`, `eval_frozen.py` (mới).

**Dữ liệu:** frame thuộc cụm train, không hỏng, `align_ok = True`; lấy mẫu cân bằng theo camera và nhóm giờ. Background theo `configs/exp/h1_*.yaml: bg_version` (`r0_median` mặc định, `r2` khi Hướng A xong).

**Tính trọng số patch đúng cách với multi-crop** — lỗi dễ mắc: tính Δ trên ảnh gốc rồi dùng cho crop. Làm đúng:

1. Tính Δ một lần ở độ phân giải gốc trên cặp (frame đã warp + color\_align, background).
2. Khi sinh mỗi global crop, áp **cùng tham số** RandomResizedCrop và lật ngang lên Δ (nội suy bilinear).
3. `avg_pool2d(Δ_crop, kernel=16)` → w (14 × 14 cho crop 224).
4. Nếu có σ̄: w ← w · (1 − σ̄\_norm), với σ̄ cũng được crop cùng tham số.

```python
# masking.py
def fam_probs(w: Tensor, r_i: float, alpha_max=0.75, T=0.1) -> Tensor:
    """w: (N,) trọng số patch. Trả về P (N,), tổng = 1."""
    # rank trung bình khi bằng nhau (như scipy.stats.rankdata 'average'):
    # w hằng số -> mọi patch cùng rank -> P đều. KHÔNG dùng argsort().argsort().
    w_rank = average_rank(w) / max(len(w) - 1, 1)
    p_fg = torch.softmax(w_rank / T, dim=0)
    alpha = alpha_max * r_i
    return alpha * p_fg + (1 - alpha) / len(w)

def sample_mask(P: Tensor, ratio: float, gen) -> Tensor:
    """Lấy k = round(ratio*N) patch không hoàn lại bằng Gumbel top-k:
    key = log P + Gumbel(0,1); mask = top-k(key). Trả về bool (N,)."""
```

Tỷ lệ che lấy ngẫu nhiên đều trong \[0.1, 0.5\] cho mỗi ảnh, và chỉ che 50% số ảnh trong batch (theo cách iBOT/DINOv2). Chỉ che global view của student; teacher luôn nhận ảnh không che.

**Mô hình và loss:** student/teacher ViT-B/16 khởi tạo từ DINOv3 (nếu giấy phép trọng số không cho phép, dùng DINOv2 và ghi rõ). Đầu DINO trên \[CLS\] và đầu iBOT trên patch, mỗi đầu 16.384 prototype (khởi tạo mới). Loss = L\_DINO + 1.0 · L\_iBOT (+ 0.1 · KoLeo nếu theo DINOv2). Teacher là EMA với momentum tăng từ 0.994 lên 1.0 theo cosine; nhiệt độ teacher 0.04 → 0.07 trong 10% đầu quá trình train; centering như DINO.

**Huấn luyện liên tục:** lr 5e-5 (thấp vì là tiếp tục huấn luyện), layer-wise lr decay 0.9, batch hiệu dụng 256 (tích lũy gradient), bf16. Số bước **giống hệt** giữa mọi biến thể.

**Các biến thể bắt buộc (cùng compute):**

| ID | Biến thể |
| --- | --- |
| h1-0 | DINOv3 gốc, không huấn luyện thêm |
| h1-rand | Huấn luyện liên tục + che ngẫu nhiên |
| h1-attm | + AttMask: che patch có attention \[CLS\] của teacher cao nhất ở block cuối, kèm 10% patch gợi ý ngẫu nhiên |
| h1-fam | + FAM, không có cổng tin cậy (r\_i = 1) |
| h1-fam-gate | + FAM + cổng r\_i |
| h1-fam-sigma | + FAM + cổng + σ̄ từ Hướng A |

**Đánh giá với feature đóng băng (`eval_frozen.py`):**

1. Mức ùn tắc: logistic regression trên \[CLS\] với gold set (chia theo cụm), báo cáo macro-F1, QWK; thêm kNN (k = 20, cosine).
2. Đếm xe few-shot: ridge regression trên \[CLS\] + mean patch, 5/10/20% nhãn train, 5 lần lấy mẫu, báo cáo MAE.
3. Attention ∩ xe: ngưỡng hóa attention \[CLS\] (giữ các patch chiếm 30% tổng khối lượng attention), IoU với mask xe audit-test.
4. Chạy BDB (IV.4) với h1-fam và h1-fam-gate: làm xấu background **lúc train**, đo metric 1–2.

**Báo cáo thêm:** tỷ lệ ảnh có FAM gần như đều (entropy của P / ln N > 0.95) theo nhóm giờ — giải thích phần nào ảnh đêm không được hưởng lợi.

**Nghiệm thu:** (1) `fam_probs` tổng bằng 1 (sai số 1e-6) và bằng phân phối đều khi w hằng số và r\_i bất kỳ; (2) số patch bị che đúng bằng round(ratio·N); (3) test với ảnh tổng hợp có một ô vuông sáng: sau crop và lật, patch có w lớn nhất nằm đúng vị trí ô vuông; (4) chạy thử 1.000 bước không NaN, loss giảm.

### IV.7. Spec H3 và H4

#### H3 — Foreground-Enhanced Counting

**File:** `direction3_foreground_enhanced_counting/`; thêm `fusion.py` cho các biến thể tiêm prior.

**Kênh Δ:** tính sau warp + color\_align; chuẩn hóa bằng cách **chia cho p99 của Δ trên tập train** (không trừ trung bình), kẹp \[0, 1\]. Nhờ vậy giá trị 0 nghĩa là "không khác nền", và Δ-dropout chỉ cần gán kênh bằng 0. Nếu có σ̄: Δ ← Δ · (1 − σ̄\_norm).

```python
def adapt_patch_embed_to_4ch(conv: nn.Conv2d, init: str = 'zero') -> nn.Conv2d:
    """Tạo conv 4 kênh, copy trọng số RGB. Kênh 4:
    'zero'   -> 0 (mặc định; đầu ra ở bước 0 bằng đúng mô hình 3 kênh)
    'mean'   -> trung bình 3 kênh RGB (như H3 gốc)
    'random' -> trunc_normal(std=0.02)"""
```

**Các biến thể fusion** (`fusion.py`, chọn qua config `fusion:`): `none`; `early` (4 kênh); `late` — Δ-CNN 4 tầng conv 3×3 stride 2 (32, 64, 128, 128 kênh, GELU) → global average pooling → vector 128, nối với \[CLS\] và trung bình token patch → MLP; `global` — xem IV.12 (Hướng E). Control: `early_gray` (kênh 4 = ảnh xám của origin), `early_othercam` (Δ tính với background camera cụm khác).

**Đầu ra và loss:** 3 giá trị đếm (xe máy, ô tô, xe buýt/tải); dự đoán trên thang log1p, loss Smooth L1 trên log1p(count); tổng = tổng 3 lớp. Đánh giá MAE/RMSE trên thang gốc.

**Huấn luyện:** fine-tune 4 block cuối + đầu; lr 2e-5 (backbone), 1e-4 (đầu); 30 epoch; Δ-dropout p = 0.3; BDB lúc train như IV.4. Few-shot: 5 / 10 / 20 / 100% frame có nhãn **trong cụm train**, phân tầng theo camera, 5 lần lấy mẫu với seed cố định.

**Báo cáo:** MAE theo 3 mức mật độ (phân vị 33/66 của tổng số xe trên train) × ngày/đêm; đường cong BDB cho `early`, `late`, `global`, `none`.

**Nghiệm thu:** (1) với `init='zero'`, đầu ra mô hình 4 kênh ở bước 0 khớp mô hình 3 kênh (`allclose`, atol 1e-5); (2) tỷ lệ batch có Δ bị dropout đo được trong khoảng 0.3 ± 0.02; (3) không file nhãn nào của cụm val/test được mở trong `train.py`.

#### H4 — Spatio-Temporal Density

**File:** `direction4_temporal_density/`; thêm `cache_embeddings.py`.

**Bước 1 — cache:** chạy DINOv3 đóng băng một lần cho mọi frame hợp lệ, lưu \[CLS\] + trung bình patch (float16) vào `data/emb/{model}/{camera_id}.npy` kèm danh sách frame\_id. Mọi thí nghiệm H4, B, D đọc cache này.

**Bước 2 — chuỗi:** chuỗi T = 12 bước kết thúc tại t (**chỉ quá khứ**). Cắt chuỗi khi `gap_prev_s` > 1.5 × median khoảng lấy mẫu của camera. Thêm đặc trưng thời gian: sin/cos của giờ trong ngày, one-hot thứ trong tuần.

**Bước 3 — nhãn:**

```python
def proxy_occupancy(delta, road_mask, tau) -> float:
    """Tỷ lệ pixel trong road_mask có delta > tau. tau chọn trên gold-train để
    MAE với gold_occupancy nhỏ nhất. Không chia cho toàn ảnh."""

def fit_los_thresholds(rho_gold, level_gold) -> tuple[float, float, float]:
    """Tìm 3 ngưỡng tăng dần trên rho (lưới bước 0.01) tối đa hóa QWK với
    nhãn người trên gold-train. Thay cho 0.15/0.35/0.60 tự đặt."""
```

Mục tiêu huấn luyện, chọn qua config `target:` — `proxy_all` (ρ\_proxy mọi frame), `proxy_reliable` (chỉ frame có r\_i ≥ 0.7 và không thuộc nhóm cao điểm/đêm), `weak_B` (nhãn mềm từ Hướng B). Đánh giá luôn trên gold.

**Bước 4 — mô hình:** embedding cache (+ Δ-CNN tùy chọn) → Linear 256 → **GRU một chiều** 2 tầng, 256 chiều → các đầu: ρ̂\_t (sigmoid), mức ùn tắc t (4 logit), và với chế độ dự báo: ρ̂ và mức tại t + h, h ∈ {1, 3, 6} bước. Loss: Smooth L1 (ρ) + CE (mức) + 0.01 · TV-L1 theo thời gian.

**Thí nghiệm then chốt:** huấn luyện với `proxy_reliable`, đánh giá trên frame **không** đáng tin (cao điểm, đêm, r\_i thấp). Nếu MAE so với gold thấp hơn chính ρ\_proxy trên các frame đó, mô hình đã vượt nhãn yếu của nó.

**Nghiệm thu:** (1) test nhân quả: thay frame t + 1 bằng nhiễu ngẫu nhiên, mọi đầu ra tại thời điểm ≤ t không đổi; (2) `proxy_occupancy` chỉ dùng pixel trong road mask (test với mask chỉ phủ nửa ảnh); (3) ngưỡng mức ùn tắc đọc từ file sinh bởi `fit_los_thresholds`, không có hằng số trong code.

### IV.8. Spec Hướng A (H2 nâng cấp) — ánh xạ sang code

Thiết kế khoa học nằm ở A.6–A.14; mục này chỉ nêu những gì agent cần để code đúng.

**File trong `direction2_scene_decomposition/`:**

| File | Nội dung |
| --- | --- |
| `groups.py` | `build_groups(frames_df, K, min_days)`: nhóm (camera, slot, mode, cam\_epoch) → danh sách frame; mỗi mẫu lấy K frame từ K ngày khác nhau; nhóm không đủ K ngày thì gộp slot ±1h, vẫn thiếu thì bỏ |
| `dataset.py` | `GroupDataset`: trả về `frames` (K×3×H×W), `prior` (3×H×W), `road` (1×H×W), `meta`; frame đã warp + color\_align về prior |
| `models.py` | `DecompNet(backbone, freeze_blocks, use_prior_input)` → dict `mask, fg, bg, log_sigma` |
| `losses.py` | Mỗi loss một hàm, nhận dict đầu ra; `DecompLoss` gộp theo trọng số trong config và lịch 3 pha |
| `train.py` | Vòng train; ghi log các đại lượng theo dõi ở A.10 |
| `refine.py` | Vòng tự làm sạch A.12; ghi background mới `version = r{n}` và `sigma_path` vào `backgrounds.parquet` |
| `synth.py` | Sinh tập bán tổng hợp A.7; lưu ảnh, mask xe, mặt đường gốc |
| `inject_ghost.py` | Làm hỏng prior bằng `degrade(kind='ghost')` cho thí nghiệm ghost cài sẵn; lưu `ghost_mask` |
| `evaluate.py` | Mọi metric của A.14, đọc tập test, xuất `metrics.json` |
| `baselines/` | `delta_threshold.py`, `mog2.py`, `detector_lama.py`, `omnimatte_subset.py` |

**Chi tiết cài đặt loss dễ sai:**

- SSIM: cửa sổ 11×11, Gaussian σ = 1.5 (`pytorch_msssim` hoặc kornia), trên ảnh \[0, 1\].
- Charbonnier: ρ(x) = sqrt(x² + 1e-6), trung bình qua kênh.
- Median theo K: `torch.median(bg, dim=0).values`, bọc `.detach()`.
- σ = exp(clamp(log\_sigma, log 0.01, log 0.5)). Ở pha 1, thay σ bằng hằng 0.1 (không dùng đầu ra mạng).
- Stop-gradient: đầu `log_sigma` chỉ nhận gradient từ L\_prior — dùng `log_sigma.detach()` ở mọi chỗ khác (nếu có).
- L\_excl dùng `bg.detach()` bên trong hàm mũ.
- Mọi loss pixel tính trên toàn ảnh; báo cáo metric thì giới hạn trong road mask.

**`refine.py` — trung vị có trọng số theo pixel:** với mỗi pixel, sắp xếp n giá trị B̂ (từng kênh riêng), lấy giá trị tại đó tổng trọng số tích lũy vượt 50%. Cài đặt vector hóa bằng `torch.sort` theo trục n. Nếu tổng trọng số tại pixel < 1e-3 (bị xe che ở mọi frame), giữ giá trị prior cũ và đánh dấu pixel đó trong một mask `unresolved`.

**Kiểm tra sanity trước khi chạy thật:**

1. **Overfit 1 nhóm:** train trên đúng 1 nhóm K = 4 trong 500 bước; L\_rec phải giảm về gần 0 và M phải sáng ở vị trí xe.
2. **Thử nghiệm suy biến:** tắt L\_shared và L\_prior → xác nhận mô hình rơi vào M ≈ 0, B̂ ≈ I (chứng minh hai loss này thật sự cần). Ghi lại để đưa vào phần thảo luận.
3. **Ghost cài sẵn trên train:** sau 1 epoch pha 2, AUROC của σ với ghost cài sẵn trên chính camera train phải > 0.6; nếu không, kiểm tra lại pipeline trước khi tiêu thêm compute.

**Nghiệm thu:** (1) `build_groups` không bao giờ đưa 2 frame cùng ngày vào một mẫu; (2) mọi frame trong nhóm cùng mode và cam\_epoch; (3) `refine.py` chỉ đọc audit-val để quyết định dừng — assert đường dẫn; (4) `synth.py` chỉ dùng crop xe từ camera train và nền từ camera test, lưu kèm seed.

### IV.9. Spec Hướng B — gộp nhãn yếu đa nguồn

**File trong `directionB_weak_supervision/`:** `lfs/lf_detector.py`, `lfs/lf_background.py`, `lfs/lf_vlm.py`, `lfs/lf_temporal.py`, `lfs/lf_history.py`, `context.py`, `label_model.py`, `end_model.py`, `evaluate.py`, `baselines.py`.

**Định dạng đầu ra chung của LF:** `data/lf/{lf_name}.parquet` với cột `frame_id`, `label` (−1 = bỏ qua, 0–3), `raw` (giá trị thô, ví dụ số xe). Ngưỡng của mọi LF chỉnh trên **gold-dev** (200 frame thuộc cụm train), không chạm gold-test.

**Đặc tả từng LF:**

| LF | Cài đặt | Bỏ qua khi |
| --- | --- | --- |
| LF1 detector | YOLOv8x hoặc RT-DETR-L (COCO), lớp car, motorcycle, bus, truck, bicycle; conf ≥ 0.25; ảnh vào 1280 px (xe máy xa rất nhỏ). raw = số box có tâm trong road mask / (diện tích road / 10⁴ px). Ngưỡng 3 mức bằng `fit_los_thresholds` (IV.7) | Conf trung bình < 0.35; hoặc ảnh IR và 0 box |
| LF2 background | raw = ρ\_proxy (IV.7), cùng τ | r\_i < 0.5; thiếu background; `align_ok = False` |
| LF3 VLM | Qwen2-VL-7B-Instruct (thử thêm bản 2B), chạy bằng vLLM, temperature 0, 2 prompt khác nhau (prompt B đảo thứ tự mô tả các mức) | Trả "unknown"; 2 prompt không khớp; JSON lỗi |
| LF4 thời gian | Cần frame trước trong 1.5 × median khoảng lấy mẫu. s = cosine trung bình giữa token patch DINO (cache) của hai frame, chỉ trên patch thuộc road mask. Mức 3 nếu s ≥ θ\_still và raw(LF1) ≥ median; mức 0 nếu raw(LF1) ≤ phân vị 20 | Các trường hợp còn lại; không có frame trước |
| LF5 lịch sử | Đa số phiếu của LF1–LF4 cho cùng camera, cùng slot, cùng loại ngày (thường/cuối tuần) ở các ngày **khác** thuộc giai đoạn train | Ít hơn 5 ngày có dữ liệu |

LF4 dùng raw của LF1, nên hai LF này phụ thuộc nhau — ghi rõ trong paper và thử biến thể LF4 không dùng LF1 (chỉ dùng s) trong ablation.

**Prompt LF3** (tiếng Anh vì VLM ổn định hơn; nội dung bám đúng hướng dẫn gán nhãn IV.5):

```text
You are a traffic analyst. Look at this CCTV image of an urban road in Vietnam,
where most vehicles are motorbikes. Classify the congestion on the ROAD SURFACE:
0 = free flow: large gaps between vehicles
1 = moderate: dense but clear gaps, vehicles moving
2 = slow: vehicles close together, small gaps
3 = jammed: road almost fully covered, vehicles not moving
If the image is too dark, blurry or blocked to judge, answer "unknown".
Reply ONLY with JSON: {"level": 0|1|2|3|"unknown"}
```

**Ngữ cảnh c\_t (`context.py`):** biến phân loại ghép từ: `is_ir` (2), nhóm giờ (đêm / thấp điểm / cao điểm: 3), `road_type` (3), mức đông thô theo phân vị của raw(LF1) (3) → tối đa 54 tổ hợp. Mưa: thêm nếu có nguồn đáng tin (câu hỏi VLM riêng "is the road wet?"), nếu không thì bỏ.

**Mô hình gộp nhãn (`label_model.py`):**

- Biến ẩn y\_t ∈ {0,1,2,3}, chuỗi theo (camera, ngày), cắt chuỗi khi khoảng trống > 1.5 × median.
- P(λ\_j = l | y, c) = softmax\_l(W\_j\[y, l\] + U\_j\[y, l, :\] · onehot(c)); LF bỏ qua thì không đóng góp vào likelihood.
- Khởi tạo: P(y) đều; ma trận chuyển A có đường chéo 0.9; W\_j sao cho độ chính xác 0.7; U\_j = 0. Đặt nhãn khởi tạo bằng bỏ phiếu đa số để cố định hoán vị nhãn.
- EM: bước E bằng forward–backward; bước M cập nhật A từ kỳ vọng chuyển trạng thái, W và U bằng logistic regression có trọng số (L-BFGS, phạt L2 = 1e-2). Dừng khi log-likelihood tăng < 1e-4 hoặc sau 100 vòng.
- Đầu ra: `data/weak/labels.parquet` với `frame_id`, `q0..q3`.

**Baseline gộp nhãn:** bỏ phiếu đa số; Dawid–Skene (thư viện `crowd-kit`); Snorkel `LabelModel`; mô hình của ta bỏ ngữ cảnh; bỏ chuỗi Markov.

**Mô hình cuối (`end_model.py`):** cùng kiến trúc H4 bước 4 (embedding cache → GRU một chiều), loss cross-entropy mềm −Σ q(y) log p(y). Tùy chọn (config): bỏ frame có max q < 0.4.

**Nghiệm thu:** (1) **test mô phỏng** — sinh dữ liệu có chuỗi Markov thật, 5 LF với độ chính xác biết trước thay đổi theo ngữ cảnh (ví dụ LF2 đúng 0.85 ban ngày, 0.4 ban đêm); mô hình gộp phải ước lượng lại độ chính xác trong ±0.05 và có accuracy cao hơn bỏ phiếu đa số; (2) không LF nào đọc gold-test; (3) chạy lại cùng seed cho ra đúng cùng nhãn mềm.

### IV.10. Spec Hướng C — phát hiện sự cố

**Chia dữ liệu theo thời gian, không theo cụm camera.** Mô hình "bình thường" xây riêng cho từng camera, nên cái cần tách là thời gian: giai đoạn P1 (xây ngân hàng), P2 (hiệu chỉnh ngưỡng, khoảng 2 tuần), P3 (đánh giá, nên chứa mùa mưa). Ba giai đoạn liên tiếp, không chồng lấn; ghi ranh giới trong `configs/exp/c_*.yaml`.

**File trong `directionC_anomaly/`:** `features.py`, `pooling.py`, `bank.py`, `score.py`, `events.py`, `camera_fault.py`, `synth_events.py`, `mine_candidates.py`, `review_ui.py`, `evaluate.py`, `baselines/`.

**Pipeline:**

1. **Feature (`features.py`):** DINOv3 ViT-B đóng băng, ảnh 448 × 256 → 16 × 28 = 448 token patch, 768 chiều. Giảm về 128 chiều bằng PCA fit trên mẫu ngẫu nhiên 200k token của P1. Đánh dấu patch đường (≥ 50% pixel thuộc road mask) và patch tĩnh (≥ 50% thuộc `static`).
2. **Gộp thời gian (`pooling.py`):** cửa sổ W = 5 frame liên tiếp (không có khoảng trống > 1.5 × median); median theo từng chiều, từng patch → F̃\_t (448 × 128). Bước trượt 1 frame.
3. **Nhóm giờ:** đêm 22–5h, cao điểm sáng 6–9h, trưa 10–15h, cao điểm chiều 16–19h, tối 20–21h (đặt trong config).
4. **Ngân hàng (`bank.py`):** với mỗi (camera, nhóm giờ, mode), gom F̃ trên patch đường từ P1; coreset k-center tham lam còn 10% (tối đa 20.000 vector); index `faiss.IndexFlatL2`. **Làm sạch hai lượt:** xây ngân hàng, chấm điểm chính P1, bỏ 1% cửa sổ điểm cao nhất (có thể chứa sự cố chưa biết), xây lại.
5. **Điểm (`score.py`):** điểm patch = khoảng cách L2 đến láng giềng gần nhất; điểm cửa sổ = trung bình 10 patch đường điểm cao nhất. Tương tự cho patch tĩnh với ngân hàng tĩnh riêng → `static_score`.
6. **Ngưỡng:** τ\_cam = phân vị 99.5% của điểm cửa sổ trong P2, theo (camera, nhóm giờ).
7. **Sự kiện (`events.py`):** bắt đầu khi điểm > τ trong N = 3 cửa sổ liên tiếp; kết thúc khi < τ trong 3 cửa sổ liên tiếp. Lưu `event_id, camera_id, t_start, t_end, peak_score, mean_static_score`.
8. **Lỗi camera (`camera_fault.py`):** sự kiện có `static_score` > τ\_static, hoặc chuỗi `align_ok = False` ≥ 3 frame → gán `camera_fault`, tách khỏi sự cố giao thông.

**Sự kiện tổng hợp (`synth_events.py`):** trên các ngày bình thường của P3, dán cố định một vật cản (crop xe hoặc rào chắn, từ thư viện của camera train) vào cùng vị trí trên road mask trong M frame liên tiếp, M ∈ {5, 10, 20}. Đo tỷ lệ phát hiện và độ trễ (số frame từ lúc xuất hiện đến lúc báo). Thêm đối chứng: vật xuất hiện trong đúng 1 frame (thoáng qua) — không được tạo sự kiện.

**Tập đánh giá thật:**

1. `mine_candidates.py`: hợp của top 200 sự kiện từ mỗi phương pháp + 200 cửa sổ ngẫu nhiên của P3; trộn ngẫu nhiên, ẩn nguồn.
2. `review_ui.py`: sinh trang HTML tĩnh, mỗi ứng viên hiện lưới 5 frame (t−2 … t+2) và nút gán: `flood`, `stalled_or_accident`, `obstruction`, `camera_fault`, `abnormal_congestion`, `normal`, `unsure`. Xuất `incidents.parquet` (IV.2).
3. Đối chiếu ngập: người dùng lập tay `flood_reports.csv` (ngày, quận, tên đường, link nguồn) từ báo chí và thông báo của thành phố; script ghép với camera trong bán kính 500 m và cửa sổ ±2 giờ để đo recall cho loại ngập.

**Baseline (`baselines/`):** PatchCore mức frame (không gộp thời gian); một ngân hàng chung cho mọi camera; autoencoder tích chập tái tạo frame; ngưỡng trên Δ với background có sẵn; VLM zero-shot ("is there flooding, an accident or an obstruction on the road?") trên mọi frame của một tập con.

**Nghiệm thu:** (1) vật cản tổng hợp kéo dài ≥ 5 frame được phát hiện trong ≤ N + 2 cửa sổ ở ≥ 90% trường hợp trên dữ liệu P2; (2) vật thoáng qua 1 frame tạo sự kiện ở ≤ 1% trường hợp; (3) tỷ lệ cửa sổ vượt ngưỡng trên chính P2 nằm trong 0.5% ± 0.1%; (4) không file nào của P3 được đọc trong lúc xây ngân hàng hay chọn ngưỡng.

### IV.11. Spec Hướng D — dự báo trên đồ thị camera

**Điều kiện bắt đầu:** đã có mô hình cuối của Hướng B (hoặc H4) và cache embedding (IV.7). Kiểm tra trước: median khoảng lấy mẫu ≤ 5 phút ở ít nhất 70% camera; nếu không, nâng bước lưới lên 10–15 phút và ghi rõ. Mô hình B (hoặc H4) dùng để tạo đích phải được huấn luyện \*\*chỉ trên giai đoạn train của D\*\*, để không dự đoán nào trong giai đoạn test đến từ một mô hình đã thấy chính các frame đó.

**File trong `directionD_forecasting/`:** `build_series.py`, `graph.py`, `dataset.py`, `models/`, `train.py`, `evaluate.py`, `gold_eval.py`.

**1. Chuỗi thời gian (`build_series.py`):** lưới đều Δt = 5 phút theo giờ Việt Nam. Với mỗi (camera, ô thời gian): trung bình các dự đoán mức frame trong ô → `q0..q3`, `level_exp` = Σ y·q\_y, `rho_hat` (nếu dùng H4), embedding PCA 32 chiều (PCA fit trên giai đoạn train); `mask` = 1 nếu ô có ít nhất 1 frame. Lưu tensor `X[T, N, F]`, `M[T, N]` và danh sách camera, mốc thời gian.

**2. Đồ thị (`graph.py`):**

- Tải mạng đường TP.HCM bằng `osmnx` (loại `drive`), gắn mỗi camera vào nút gần nhất.
- d\_ij = độ dài đường đi ngắn nhất (mét). W\_ij = exp(−d\_ij² / s²) với s = độ lệch chuẩn của các d\_ij hữu hạn; gán 0 khi W\_ij < 0.1 hoặc d\_ij > 2 km. Bản cơ sở dùng đồ thị đối xứng; thử bản có hướng theo chiều đường nếu biết hướng nhìn camera.
- Đồ thị chỉ dựng từ tọa độ và mạng đường, không dùng dữ liệu giao thông.

**3. Mẫu huấn luyện (`dataset.py`):** đầu vào 12 bước (1 giờ), đích tại h = 3, 6, 12 bước (15, 30, 60 phút). Đặc trưng nút mỗi bước: `q0..q3`, `level_exp`, `rho_hat`, embedding 32 chiều, `mask`, sin/cos giờ trong ngày, one-hot thứ. Giá trị thiếu gán 0, kèm kênh `mask`. Lúc train, che ngẫu nhiên 10% số nút trong cả cửa sổ đầu vào (mô phỏng camera hỏng). Chuẩn hóa (z-score) fit **chỉ trên giai đoạn train**.

**Nhãn khởi phát kẹt:** onset\_h(i, t) = 1 nếu mức tại t ≤ 1 và mức lớn nhất trong (t, t + h\] ≥ 3.

**4. Mô hình (`models/`):** nên tích hợp dữ liệu vào một framework có sẵn các baseline (ví dụ BasicTS hoặc LibCity) để không phải tự cài DCRNN, Graph WaveNet, STAEformer — **kiểm tra lại** framework nào đang được bảo trì trước khi chọn. Mô hình đề xuất = Graph WaveNet hoặc STAEformer với 3 thay đổi: kênh mask ở đầu vào, đặc trưng embedding hình ảnh, và 3 đầu ra. Loss = masked MAE (trên `level_exp`, chỉ tính ô có quan sát) + 0.5 · CE (mức tại t + h) + 0.5 · focal BCE (γ = 2) cho onset.

**5. Chia dữ liệu:**

| Chế độ | Cách chia | Mục đích |
| --- | --- | --- |
| Chính (transductive) | Theo thời gian: 70% tuần đầu train, 10% val, 20% tuần cuối test; mọi camera | So sánh với baseline |
| Nút chưa thấy (inductive) | Như trên, nhưng camera thuộc cụm test (splits.json) bị che hoàn toàn khi train; lúc test được dùng lịch sử của chính chúng | Kiểm tra tổng quát hóa sang camera mới |
| Stress test | Các ngày lễ (Tết, 2/9…) nếu có trong giai đoạn test, báo cáo riêng | Độ bền với dịch chuyển phân phối |

**6. Đánh giá với nhãn người (`gold_eval.py`) — bắt buộc vì đích do mô hình khác tạo ra:** lấy 300 frame trong giai đoạn test (phân tầng theo nhóm giờ và mức dự đoán), gán theo IV.5; so dự báo đưa ra tại t − h cho ô chứa frame đó với nhãn người. Báo cáo song song với metric trên đích tự sinh.

**Baseline:** trung bình lịch sử (cùng thứ, cùng ô giờ, tính trên train), giá trị gần nhất, ARIMA (statsmodels, 50 camera ngẫu nhiên), LSTM từng nút, DCRNN, Graph WaveNet, STAEformer — tất cả cùng đầu vào và cùng ngân sách chỉnh siêu tham số.

**Nghiệm thu:** (1) không cửa sổ nào vượt qua ranh giới train/val/test; (2) bộ chuẩn hóa chỉ thấy dữ liệu train (assert theo mốc thời gian); (3) thay đổi giá trị đích tương lai không làm đổi tensor đầu vào; (4) dự báo của baseline "giá trị gần nhất" tính tay khớp với code trên 10 ví dụ.

### IV.12. Spec Hướng E — điều kiện hóa bằng thống kê background

**File trong `directionE_bg_conditioning/`:** `descriptor.py`, `conditioning.py`, `dataset.py` (dùng lại dataset H3 và gold của B), `train.py`, `evaluate.py`, `analysis.py`.

**1. Bản mô tả cảnh (`descriptor.py`)** — cache theo (camera, slot, mode, bg\_version):

```python
def scene_descriptor(bg, road_patch_mask, sigma=None, trim=0.10) -> np.ndarray:
    """bg -> DINOv3 đóng băng (448x256) -> token patch P (448 x 768).
    Với từng tập patch S in {road, all}:
      m = median theo patch (từng chiều); d_p = ||P_p - m||_2
      bỏ trim*|S| patch có d_p lớn nhất -> S'
      nếu có sigma: trọng số w_p = 1 - sigma_norm trung bình trong patch
      mean_S, std_S = trung bình/độ lệch chuẩn có trọng số trên S'
    Trả về z = [mean_road, std_road, mean_all]  (2304,)"""

def slot_aggregate(z_by_slot, s) -> np.ndarray:
    """median từng chiều của z tại slot s-1, s, s+1 (bỏ slot thiếu)."""
```

**2. Ba dạng điều kiện hóa (`conditioning.py`):**

| Dạng | Cài đặt | Khởi tạo để bước 0 = mô hình không dùng bg |
| --- | --- | --- |
| `film` | MLP: LayerNorm(2304) → 512 → GELU → 2 · 768 · L (L = 4 block cuối). Trong mỗi block: h ← γ ⊙ LN(h) + β, áp sau LayerNorm trước attention | Tầng Linear cuối: trọng số 0, bias cho γ = 1, β = 0 |
| `prompt` | MLP: 2304 → 512 → K · 768, K = 4 token nối sau token \[CLS\] ở đầu vào, không có position embedding | Không thể đúng identity — báo cáo đầu ra bước 0 thay đổi bao nhiêu |
| `xattn` | Thêm khối cross-attention (8 head) sau block 9 và 12: query = token frame, key/value = token patch của bg (encoder đóng băng) | Projection đầu ra khởi tạo 0 |

Token rỗng `z_null` (học được) thay cho z khi bg bị bỏ.

**3. Huấn luyện:** siêu tham số như H3 (IV.7). Tăng cường: bg-dropout 0.25; bg-swap (lấy z của slot lệch ±1–2 giờ) 0.3; chèn ghost vào ảnh bg **trước khi** tính z, xác suất 0.3, severity 1–3.

**4. Bài toán:** đếm xe (nhãn H3) và mức ùn tắc (gold, hoặc nhãn mềm B cho train).

**5. Các kiểu chia (file riêng, đều sinh từ `splits.py`):**

| Kiểu | Định nghĩa |
| --- | --- |
| `cluster` | splits.json chuẩn (IV.3) |
| `region` | Giữ nguyên 1–2 quận làm test (khoảng 20% camera), phần còn lại train/val theo cụm — `splits_region.json` |
| `day2night` | Train chỉ ảnh mode `day`, test ảnh mode `ir` của camera cụm test |

**6. Baseline:** `none`; H3 `early`; H3 `late`; TENT — chỉ áp được cho bài toán phân loại mức ùn tắc (cập nhật tham số affine của LayerNorm bằng tối thiểu entropy, batch 32 frame của cùng camera test); fine-tune với k ∈ {10, 50} frame có nhãn của camera mới làm mốc tham chiếu trên.

**7. Phân tích (`analysis.py`):** t-SNE của z tô màu theo `road_type` và nhóm giờ; tương quan Spearman giữa 3 thành phần chính đầu tiên của z và chiều cao box xe trung bình theo camera (đại diện phối cảnh); cosine giữa z sạch và z sau khi chèn ghost 5–50% (đường cong độ bền của chính bản mô tả).

**Nghiệm thu:** (1) với `film` và `xattn`, đầu ra bước 0 khớp mô hình `none` (atol 1e-5); (2) tra cứu z luôn theo đúng camera\_id của frame — test bằng cách hoán đổi camera và kiểm tra assert; (3) z chỉ tính từ ảnh background, không bao giờ từ frame đang dự đoán.

### IV.13. Checklist trước khi nộp Q1

Áp dụng cho mọi paper sinh ra từ tài liệu này. Agent làm được các mục có đánh dấu (A); các mục còn lại cần người làm.

**Khóa thiết kế trước khi chạy thí nghiệm cuối**

- [ ] Ghi vào `configs/exp/final_*.yaml`: metric chính, tập test, baseline, số seed — trước khi mở tập test lần đầu (A)
- [ ] Tập test chỉ được đánh giá với cấu hình đã chọn trên val; số lần mở tập test được ghi log (A)
- [ ] Kiểm tra cỡ mẫu theo quy tắc ở IV.5 sau khi gán nhãn (A)

**Tính mới và trích dẫn**

- [ ] Tra cứu có hệ thống trên Scopus, IEEE Xplore, Web of Science, Google Scholar và arXiv, giai đoạn 2015–nay. Ví dụ chuỗi tìm kiếm: ("background subtraction" OR "scene decomposition") AND ("noisy" OR "imperfect" OR "uncertainty") AND ("traffic" OR "surveillance")
- [ ] Lập bảng công trình liên quan: tên, venue, năm, khác biệt với ta — lặp lại tra cứu 1 tháng trước khi nộp
- [ ] Kiểm tra từng trích dẫn trong tài liệu này bằng DOI hoặc trang chính thức (năm, venue, tác giả)
- [ ] Chỉ dùng chữ "đầu tiên" khi bảng tra cứu ủng hộ; nếu không, dùng "theo hiểu biết của chúng tôi" kèm phạm vi rõ ràng

**Tái lập**

- [ ] `environment.yml` hoặc Dockerfile; ghi phiên bản CUDA, PyTorch (A)
- [ ] Mọi bảng và hình sinh bằng script từ `metrics.json` (A)
- [ ] Báo cáo compute: loại GPU, tổng giờ GPU cho mỗi thí nghiệm chính (A)
- [ ] Phát hành code (GitHub) kèm config cuối và checkpoint nếu được phép

**Đạo đức và dữ liệu**

- [ ] Văn bản cho phép dùng dữ liệu cho nghiên cứu và công bố hình ảnh
- [ ] Làm mờ khuôn mặt và biển số trong mọi hình trong paper (detector tự động + kiểm tra tay) (A)
- [ ] Viết mục Data Availability và Ethics Statement

**Chọn tạp chí**

- [ ] Kiểm tra hạng Q1 hiện tại trên SJR cho đúng phân ngành (hạng thay đổi hằng năm)
- [ ] Đọc Aims & Scope, xem 5–10 bài gần nhất của tạp chí cùng chủ đề và trích dẫn khi phù hợp
- [ ] Kiểm tra phí xuất bản, giới hạn trang, định dạng (LaTeX template)

**Chuẩn bị cho câu hỏi của reviewer**

| Câu hỏi dễ gặp | Câu trả lời cần có sẵn trong paper |
| --- | --- |
| Sao không dùng detector có sẵn? | Bảng so sánh với detector (và detector + LaMa), chia theo cao điểm/đêm |
| Background nhiễu thì kết quả có ý nghĩa không? | Tập audit (mục 0.2) + đường cong BDB |
| Có tổng quát sang thành phố khác không? | Chia theo vùng (`region`); nếu có, thử thêm trên một bộ dữ liệu công khai khác — cần tìm bộ phù hợp |
| Cải thiện có thật hay do tham số thêm? | Control `other_camera`, `early_gray`; cùng ngân sách chỉnh siêu tham số |
| Kết quả có ổn định không? | 3 seed, CI bootstrap theo cụm, Holm |

**Trình bày**

- [ ] Hình dạng vector, bảng màu thân thiện với người mù màu (A)
- [ ] Mục Limitations trung thực (xe đứng yên cố định, phụ thuộc tần suất lấy mẫu, một thành phố)
- [ ] Hiệu đính tiếng Anh trước khi nộp

## V. Kế hoạch thực nghiệm đầy đủ

Phần này liệt kê **mọi thí nghiệm** cần chạy để ra kết quả cho paper, theo từng hướng, để agent có thể biến mỗi dòng thành một file config và chạy. Phần V bổ sung cho các mục thí nghiệm trước đó (A.14, II.B–II.E); khi trùng nhau, phần V chi tiết hơn và được ưu tiên.

### Ba phát hiện từ tra cứu làm thay đổi kế hoạch

1. **Hướng A có công trình rất gần:** AE-NE ([Sauvalle & de La Fortelle, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf)) đã cho autoencoder dự đoán thêm một kênh nhiễu background theo pixel và đánh giá trên CDnet 2014, LASIESTA, BMC 2012. Nhưng AE-NE huấn luyện **riêng cho từng video**, và chính tác giả ghi nhận mô hình thất bại khi vật lớn đứng yên lâu. Vì vậy Hướng A phải: (a) có AE-NE làm baseline, (b) đánh giá thêm trên CDnet 2014, (c) định vị rõ khác biệt — một mô hình chung cho cả thành phố, prior bên ngoài có độ bất định, và ràng buộc nhiều ngày giải quyết đúng điểm yếu xe đứng yên. Công trình về trích tiền cảnh từ background không hoàn hảo ([Chan et al.](https://arxiv.org/pdf/1808.08210)) cũng phải trích dẫn.
2. **Hướng B có baseline chuẩn sẵn:** [WRENCH (NeurIPS 2021 Datasets & Benchmarks)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/1c9ac0159c94d8d0cbedc973445af2da-Abstract-round2.html) cung cấp cài đặt của nhiều label model, trong đó có cả HMM và **Conditional HMM** — rất gần ý tưởng "HMM có điều kiện ngữ cảnh" của Hướng B. Cần dùng chúng làm baseline và định vị lại tính mới (xem V.6).
3. **Có bộ dữ liệu Việt Nam để kiểm tra chéo:** một bộ khoảng 23.000 ảnh CCTV ở Đà Nẵng với hơn 1,1 triệu nhãn ([Vo et al., CMC 2026](https://www.techscience.com/cmc/online/detail/26156/pdf)) và UIT-VinaDeveS22 ([CTU Journal](https://ctujs.ctu.edu.vn/index.php/ctujs/article/download/461/612)). Dùng được theo hai cách: huấn luyện detector "địa phương" làm **baseline mạnh hơn** detector COCO, và làm tập kiểm tra **khác thành phố** cho bài toán đếm. Cần kiểm tra giấy phép từng bộ.

### Phân loại thí nghiệm và mã

Mỗi thí nghiệm có mã `{hướng}-{loại}{số}`, ví dụ `A-AB3`.

| Mã loại | Loại | Trả lời câu hỏi |
| --- | --- | --- |
| M | Kết quả chính | Phương pháp có tốt hơn baseline không? |
| AB | Ablation | Thành phần nào đóng góp bao nhiêu? |
| R | Robustness | Có bền khi dữ liệu xấu đi không? |
| G | Tổng quát hóa | Có chạy được ở camera, vùng, điều kiện chưa thấy không? |
| S | Độ nhạy và quy mô | Kết quả thay đổi thế nào theo siêu tham số, lượng dữ liệu? |
| E | Hiệu năng tính toán | Có triển khai được cho 608 camera không? |
| X | Benchmark bên ngoài | Có so được với kết quả công bố trên dữ liệu chuẩn không? |
| Q | Định tính và lỗi | Thành công và thất bại trông như thế nào? |

**Mức ưu tiên:** **P0** — bắt buộc, thiếu thì không nộp; **P1** — reviewer Q1 thường hỏi, nên có trong bài chính; **P2** — đưa vào phụ lục nếu còn thời gian.

### Registry cho agent

Mỗi thí nghiệm là một file `configs/registry/{id}.yaml`. Agent viết `scripts/run_registry.py` để chạy theo bộ lọc, tự bỏ qua thí nghiệm đã có `metrics.json` khớp hash của config.

```yaml
id: A-AB3
direction: A
type: ablation          # main|ablation|robustness|generalization|scaling|efficiency|external|qualitative
priority: P0
question: "K frame khác ngày có tốt hơn K frame cùng ngày không?"
base_config: configs/exp/a_full.yaml
overrides:
  data.group.same_day: true
seeds: [0, 1, 2]
split: cluster          # cluster|region|day2night|temporal|external
eval_sets: [audit_test, synth_test]
metrics: [iou, f1, ghost_rate]
depends_on: [A-M1]
paper_target: "Paper A — Bảng 4, dòng 3"
```

```bash
python scripts/run_registry.py --direction A --priority P0          # chạy tất cả P0 của A
python scripts/run_registry.py --id A-AB3 --smoke                   # 1% dữ liệu, 1 seed, kiểm tra pipeline
python scripts/make_tables.py --paper A                             # sinh bảng LaTeX từ metrics.json
```

**Quy tắc chung cho mọi bảng ablation:** mỗi dòng chỉ thay **một** yếu tố so với cấu hình đầy đủ; dòng đầu là cấu hình đầy đủ; cùng seed, cùng số bước huấn luyện; cột cuối là chênh lệch so với cấu hình đầy đủ kèm CI 95% (IV.5).

### V.1. Bộ robustness và tổng quát hóa dùng chung

Ngoài BDB (làm xấu **background**, IV.4), cần thêm các trục khác vì reviewer sẽ hỏi "nếu chính **frame** xấu, dữ liệu thưa hơn, nhãn nhiễu hơn thì sao". Các bộ dưới đây đặt trong `common/` và mọi hướng gọi chung.

| Mã | Bộ | Làm gì | Mức độ | Áp dụng cho |
| --- | --- | --- | --- | --- |
| RS-1 | BDB | Làm xấu background (IV.4) | 5 mức × 6 loại | Mọi hướng dùng bg: H1, H3, H4, A, B (LF2), E |
| RS-2 | Frame Corruption Suite (FCS) | Làm xấu frame đầu vào theo kiểu ImageNet-C (Hendrycks & Dietterich, ICLR 2019) nhưng chọn loại hợp với camera giao thông | 5 mức × 8 loại | Mọi hướng |
| RS-3 | Thưa mẫu theo thời gian | Giữ 1/2, 1/4, 1/8 số frame của mỗi camera | 3 mức | H4, A, B (LF4, HMM), C, D |
| RS-4 | Nhiễu nhãn huấn luyện | Đổi nhãn sang mức kề bên (ordinal) 10 / 20 / 40%; với đếm: nhân nhãn với hệ số ngẫu nhiên log-normal σ = 0.1 / 0.2 / 0.4 | 3 mức | H3, H4, B (mô hình cuối), E |
| RS-5 | Quy mô dữ liệu | Dùng 10 / 25 / 50 / 100% số camera train (giữ nguyên cụm) | 4 mức | Mọi hướng có huấn luyện |
| RS-6 | Lát điều kiện tự nhiên | Không làm xấu gì, chỉ chia kết quả theo: ngày / đêm / IR, cao điểm / thấp điểm, mưa / khô, loại đường | — | Mọi hướng |

**RS-2 — chi tiết FCS (`common/corrupt.py`):** dùng thư viện `imagecorruptions` cho các loại có sẵn và tự cài các loại riêng; mỗi loại 5 mức.

| Loại | Mô phỏng | Cách làm |
| --- | --- | --- |
| gaussian\_noise | Cảm biến kém, ban đêm | `imagecorruptions` |
| motion\_blur | Xe nhanh, phơi sáng dài | `imagecorruptions` |
| defocus\_blur | Ống kính bẩn, lấy nét sai | `imagecorruptions` |
| jpeg\_compression | Nén mạnh khi truyền | `imagecorruptions` |
| low\_light | Thiếu sáng | gamma 1.5 → 3.0 + nhiễu Poisson tăng dần (tự cài) |
| fog | Sương, khói bụi | `imagecorruptions` |
| rain | Mưa | vệt mưa ngẫu nhiên kiểu albumentations `RandomRain`, mật độ tăng theo mức |
| glare | Đèn pha ban đêm | chèn 1–5 đốm sáng Gaussian lớn trên road mask (tự cài) |

Chỉ số tóm tắt: **suy giảm tương đối trung bình** như BDB (IV.4), tính riêng cho từng loại và trung bình 8 loại (giống mCE nhưng chuẩn hóa theo chính phương pháp đó ở mức 0, kèm bảng so với baseline mạnh nhất). Corruption chỉ áp **lúc test**, trừ khi thí nghiệm ghi rõ "train có corruption".

**Các kiểu chia để kiểm tra tổng quát hóa** (sinh từ `common/splits.py`, dùng chung):

| Mã | Kiểu chia | Ghi chú |
| --- | --- | --- |
| G-cluster | Theo cụm địa lý (mặc định, IV.3) | Mọi kết quả chính |
| G-region | Giữ nguyên 1–2 quận làm test | Dịch chuyển phân phối mạnh hơn |
| G-day2night | Train chỉ mode `day`, test mode `ir` | Ánh sáng |
| G-temporal | Train các tuần đầu, test các tuần sau (cùng camera) | Thay đổi theo mùa, theo thời gian |
| G-city | Test trên dữ liệu Đà Nẵng hoặc UIT-VinaDeveS22 | Chỉ cho bài toán dựa trên ảnh đơn (đếm, phân loại mức); số đếm suy từ box; cần kiểm tra giấy phép |

**E — hiệu năng tính toán, báo cáo giống nhau cho mọi hướng (`scripts/profile.py`):** số tham số; GFLOPs (fvcore) ở 448 × 256; độ trễ mỗi frame trên GPU (batch 1 và 32) và CPU; bộ nhớ GPU đỉnh; chi phí phụ trợ (đọc background, căn chỉnh, tính Δ). Kèm một dòng "khả thi vận hành": số frame/giây cần để xử lý 608 camera ở tần suất lấy mẫu thực tế, so với thông lượng đo được.

**Seed:** mọi thí nghiệm P0 chạy 3 seed; P1, P2 chạy 1 seed nếu thiếu compute, nhưng ghi rõ trong chú thích bảng.

### V.2. Thực nghiệm H1 — BG-Guided DINO

**Luận điểm cần chứng minh:** (1) che theo tiền cảnh cho biểu diễn tốt hơn che ngẫu nhiên và che theo attention với **cùng compute**; (2) hội tụ nhanh hơn (đây là tuyên bố trong tài liệu gốc mục 2.2 — phải đo, không được chỉ nói); (3) không sụp khi background xấu.

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| H1-M1 | M | FAM có tốt hơn các cách che khác? | 6 biến thể ở IV.6, feature đóng băng | Mức ùn tắc: macro-F1, QWK (linear, kNN); đếm few-shot 5/10/20%: MAE | P0 |
| H1-M2 | M | Lợi ích có giữ khi fine-tune? | Dùng backbone của từng biến thể làm khởi tạo cho H3 (fine-tune 4 block cuối) | MAE đếm | P1 |
| H1-M3 | M | Có hội tụ nhanh hơn? | Lưu checkpoint mỗi 10% quá trình train; đánh giá kNN mức ùn tắc tại mỗi checkpoint | Đường cong metric theo giờ GPU cho h1-rand, h1-attm, h1-fam-gate | P0 |
| H1-AB1 | AB | Tỷ trọng α | α\_max ∈ {0, 0.25, 0.5, 0.75, 1} | Như M1 | P0 |
| H1-AB2 | AB | Độ nhọn T | T ∈ {0.05, 0.1, 0.2, 0.5, ∞ (đều)} | Như M1 | P1 |
| H1-AB3 | AB | Chuẩn hóa w | rank / max-norm (công thức gốc, đã sửa tổng = 1) / giá trị thô qua softmax | Như M1 | P1 |
| H1-AB4 | AB | FAM có cần loss iBOT? | FAM + chỉ loss DINO (như thiết kế gốc) vs FAM + DINO + iBOT | Như M1 | P0 |
| H1-AB5 | AB | Tỷ lệ che | \[0.1, 0.5\] / \[0.3, 0.3\] / \[0.5, 0.7\] | Như M1 | P2 |
| H1-AB6 | AB | Nguồn trọng số | Δ từ bg median r0 / bg đã làm sạch r2 (Hướng A) / σ̄ / chỉ attention teacher (AttMask) | Như M1 | P1 |
| H1-AB7 | AB | Cổng tin cậy | Không cổng / cổng r\_i / cổng r\_i + σ̄ | Như M1, tách theo đêm và cao điểm | P0 |
| H1-R1 | R | Bền với bg xấu? | BDB áp **lúc pretrain** (bg xấu → w sai), 3 mức: sạch, mức 2, mức 4 của `ghost` và `geometric` | Như M1 | P0 |
| H1-R2 | R | Feature có bền với frame xấu? | FCS lúc đánh giá linear probe | Suy giảm tương đối | P1 |
| H1-G1 | G | Tổng quát? | Linear probe với G-region, G-day2night | Như M1 | P1 |
| H1-S1 | S | Lượng dữ liệu pretrain | 10 / 25 / 50 / 100% frame | Như M1 | P1 |
| H1-E1 | E | Chi phí | Thời gian/bước huấn luyện của FAM so với che ngẫu nhiên (chi phí tính Δ, đọc bg) | Giây/bước, giờ GPU tổng | P1 |
| H1-Q1 | Q | Mô hình nhìn vào đâu? | Attention \[CLS\] trên 20 ảnh test, 4 biến thể cạnh nhau | Hình | P0 |
| H1-Q2 | Q | Khi nào FAM không giúp? | Tỷ lệ ảnh có FAM gần đều theo nhóm giờ; kết quả M1 tách theo nhóm giờ | Bảng + hình | P1 |

**Tiêu chí quyết định sớm:** sau H1-M1 với 1 seed, nếu h1-fam-gate không vượt h1-attm với chênh lệch lớn hơn độ lệch chuẩn giữa seed của h1-rand, **dừng H1** và dồn compute cho Hướng A/E. Ghi kết quả này vào phụ lục của paper A như một kết quả âm.

### V.3. Thực nghiệm H3 — đếm xe với prior background

**Luận điểm:** tiêm prior background ở đâu thì vừa có lợi khi bg tốt, vừa không hại khi bg xấu. H3 nên viết chung paper với Hướng E (cùng bài toán, cùng dữ liệu), trong đó H3 là các biến thể "tiêm cục bộ" và E là "tiêm toàn cục".

**Baseline mạnh bắt buộc (ngoài các biến thể fusion):**

| Baseline | Ghi chú |
| --- | --- |
| DINOv3 đóng băng + hồi quy tuyến tính | Mốc thấp |
| DINOv3 fine-tune, không bg (`none`) | Mốc chính |
| Đếm bằng detector COCO (YOLOv8x / RT-DETR) trong road mask | Không cần nhãn đếm |
| Đếm bằng detector **fine-tune trên dữ liệu Việt Nam** (bộ Đà Nẵng hoặc UIT-VinaDeveS22) | Baseline mạnh nhất; ghi rõ dùng thêm nhãn ngoài |

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| H3-M1 | M | Fusion nào tốt nhất? | `none`, `early`, `late`, `global` (E) × few-shot 5/10/20/100% | MAE, RMSE tổng và từng loại xe; CI theo cụm | P0 |
| H3-M2 | M | So với detector? | Như M1 + 2 baseline detector | MAE theo 3 mức mật độ × ngày/đêm | P0 |
| H3-AB1 | AB | Khởi tạo kênh Δ | zero / mean / random | MAE, đường cong loss 2 epoch đầu | P0 |
| H3-AB2 | AB | Δ-dropout | p ∈ {0, 0.1, 0.3, 0.5} | MAE sạch và MAE dưới BDB mức 4 | P0 |
| H3-AB3 | AB | Biểu diễn Δ | 1 kênh LAB có trọng số / 3 kênh \|ΔL\|,\|Δa\|,\|Δb\| / hiệu RGB / Δ × (1 − σ̄) | MAE | P1 |
| H3-AB4 | AB | Cải thiện có đến từ nội dung bg? | Control `early_gray`, `early_othercam` | MAE | P0 |
| H3-AB5 | AB | Phiên bản bg | r0 median / r2 đã làm sạch (Hướng A) | MAE theo nhóm giờ | P1 |
| H3-AB6 | AB | Độ sâu fine-tune | đóng băng / 4 block cuối / toàn bộ / LoRA rank 16 | MAE, thời gian train | P1 |
| H3-AB7 | AB | Hàm loss | Smooth L1 trên log1p / MSE / Poisson NLL | MAE, RMSE | P2 |
| H3-R1 | R | Bền với bg xấu? | BDB đủ 6 loại × 5 mức cho 4 biến thể fusion | Đường cong + D + mức cắt s\* | P0 |
| H3-R2 | R | Bền với frame xấu? | FCS 8 loại × 5 mức | Suy giảm tương đối | P1 |
| H3-R3 | R | Bền với nhãn nhiễu? | RS-4 | MAE | P2 |
| H3-G1 | G | Tổng quát? | G-region, G-day2night | MAE | P0 |
| H3-G2 | G | Khác thành phố? | G-city: chỉ áp được biến thể `none` và detector (dữ liệu ngoài không có bg) — nêu rõ đây là giới hạn của cách tiêm bg | MAE | P2 |
| H3-S1 | S | Đường cong học | RS-5 × 4 biến thể fusion | MAE theo % camera train | P1 |
| H3-E1 | E | Chi phí | V.1 mục E, gồm chi phí căn chỉnh + tính Δ | ms/frame | P1 |
| H3-Q1 | Q | Sai ở đâu? | Biểu đồ phân tán dự đoán vs nhãn, tô màu theo nhóm giờ; 12 ảnh sai nhiều nhất kèm Δ | Hình | P0 |

**Kết quả trung tâm của paper:** một hình đường cong BDB cho 4 biến thể fusion trên cùng trục, cộng bảng M1. Nếu `early` thắng khi bg sạch nhưng cắt dưới `none` ở mức thấp, còn `global` không bao giờ cắt, đó đã là một câu chuyện hoàn chỉnh.

### V.4. Thực nghiệm H4 — mật độ và mức ùn tắc theo thời gian

**Luận điểm:** (1) mô hình theo thời gian ước lượng chiếm dụng và mức ùn tắc tốt hơn ngưỡng Δ và detector, đặc biệt ở cao điểm/đêm; (2) học từ frame đáng tin vẫn tổng quát sang frame không đáng tin — vượt chính nhãn yếu; (3) dự báo ngắn hạn có ích cho cảnh báo. H4 nên viết chung paper với Hướng B.

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| H4-M1 | M | Ước lượng hiện tại tốt đến đâu? | Mục tiêu `proxy_all` / `proxy_reliable` / `weak_B` × mô hình GRU một chiều; baseline: ρ\_proxy, detector COCO, detector VN, VLM zero-shot, DINOv3 + đầu tuyến tính từng frame | MAE chiếm dụng so với gold; macro-F1, QWK mức ùn tắc | P0 |
| H4-M2 | M | Có vượt nhãn yếu của chính nó? | Train `proxy_reliable`, test riêng trên lát không đáng tin (cao điểm, đêm, r\_i < 0.7) | MAE so với gold của mô hình vs của ρ\_proxy trên cùng lát; kiểm định Wilcoxon theo cụm | P0 |
| H4-M3 | M | Dự báo ngắn hạn | h = 1, 3, 6 bước; baseline: giá trị gần nhất, trung bình lịch sử cùng giờ | MAE, macro-F1 theo h | P1 |
| H4-AB1 | AB | Thành phần thời gian | Không thời gian (từng frame) / GRU một chiều / Transformer causal / BiGRU (chỉ cho ước lượng hiện tại, ghi rõ dùng tương lai) | Như M1 | P0 |
| H4-AB2 | AB | Độ dài chuỗi | T ∈ {3, 6, 12, 24} | Như M1 | P1 |
| H4-AB3 | AB | Nhánh Δ-CNN | Có / không | Như M1, sạch và dưới BDB | P0 |
| H4-AB4 | AB | Loss làm mượt | Không / L2 (thiết kế gốc) / TV-L1 | MAE chung và MAE ở các đoạn thay đổi đột ngột (chênh lệch ρ\_gold giữa 2 bước > 0.2) | P1 |
| H4-AB5 | AB | Ngưỡng mức ùn tắc | Fit từ dữ liệu (`fit_los_thresholds`) vs cố định 0.15/0.35/0.60 | macro-F1, QWK | P0 |
| H4-AB6 | AB | Ngưỡng "đáng tin" | r\_i ≥ 0.5 / 0.7 / 0.9 (đánh đổi lượng nhãn và chất lượng nhãn) | Như M2 | P1 |
| H4-AB7 | AB | Đặc trưng thời gian | Có / không sin-cos giờ, thứ | Như M1 | P2 |
| H4-R1 | R | Bền với bg xấu? | BDB lên nhánh Δ-CNN và lên ρ\_proxy làm nhãn | MAE | P0 |
| H4-R2 | R | Bền với lấy mẫu thưa? | RS-3 | MAE, F1 | P1 |
| H4-R3 | R | Bền với frame xấu? | FCS | Suy giảm tương đối | P2 |
| H4-G1 | G | Tổng quát? | G-region, G-day2night, G-temporal | Như M1 | P1 |
| H4-E1 | E | Chi phí | Có cache embedding: ms/chuỗi; không cache: ms/frame | — | P2 |
| H4-Q1 | Q | Bám theo chuỗi thật? | 6 chuỗi điển hình (hình thành kẹt, tan kẹt, mưa, đêm): ρ̂, ρ\_proxy, ρ\_gold trên cùng trục thời gian | Hình | P0 |

**Lưu ý thiết kế:** gold set phải có đủ frame **liên tiếp** để đánh giá theo thời gian (Q1, AB4). Khi gán gold (IV.5), dành 20% ngân sách cho khoảng 15 đoạn liên tục, mỗi đoạn 12–24 frame, chứa sự hình thành hoặc tan kẹt.

### V.5. Thực nghiệm Hướng A — phân rã cảnh từ prior không hoàn hảo

**Định vị lại so với AE-NE (bắt buộc viết trong Related Work):** AE-NE huấn luyện một autoencoder **riêng cho mỗi video**, dự đoán nền và mức nhiễu nền từ chính chuỗi frame, và tác giả nêu giới hạn khi vật lớn đứng yên lâu. Hướng A khác ở 4 điểm, và mỗi điểm phải có một thí nghiệm chứng minh:

| Khác biệt | Thí nghiệm chứng minh |
| --- | --- |
| Một mô hình chung cho mọi camera, suy luận 1 frame, không cần huấn luyện lại cho camera mới | A-M1 trên cụm camera test chưa thấy; A-E1 thời gian triển khai camera mới |
| σ mô hình hóa độ sai của **prior bên ngoài** (median), không phải nhiễu tái tạo | A-M2 ghost cài sẵn: AUROC của σ |
| Ràng buộc nhiều ngày xử lý xe đứng yên lâu | A-AB2 và lát "cao điểm" trong A-M1; so trực tiếp với AE-NE ở lát này |
| Tự làm sạch prior ở quy mô thành phố | A-M3 ghost rate qua các vòng |

**Baseline đầy đủ:** ngưỡng Δ trên median; MOG2 và KNN (OpenCV); SuBSENSE và PAWCS (có trong BGSLibrary); **AE-NE** (code công khai) — huấn luyện riêng cho mỗi camera test, mỗi camera dùng chuỗi frame cùng giai đoạn; detector + LaMa (COCO và detector VN); H2 gốc.

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| A-M1 | M | Mask và xóa xe tốt đến đâu? | Mọi baseline, chế độ (a) và (b), trên audit-test và tập bán tổng hợp | IoU, F1, boundary F; PSNR/LPIPS vùng che; KID; tách theo nhóm giờ và mật độ | P0 |
| A-M2 | M | σ có tìm ra chỗ prior sai? | Ghost cài sẵn 5/15/30% + ghost thật trên audit | AUROC, AP của σ̄; so với hai mốc: Δ giữa prior và B̂, và phương sai của B̂ qua K frame | P0 |
| A-M3 | M | Prior có sạch dần? | Vòng 0–3 | Ghost rate trên audit-test; PSNR/LPIPS so với frame trống thật | P0 |
| A-M4 | M | Có ích cho ứng dụng? | Chiếm dụng = trung bình M\_α trong road mask, so với ρ\_proxy từ median và từ prior r2 | MAE so với gold chiếm dụng | P0 |
| A-X1 | X | So được với benchmark chuẩn? | CDnet 2014, chọn hạng mục liên quan: baseline, badWeather, lowFramerate, nightVideos, intermittentObjectMotion, shadow, cameraJitter. Thích ứng: prior = median của video; K frame lấy ở các đoạn cách xa nhau trong cùng video thay cho "khác ngày". Chạy theo **giao thức không thấy cảnh** (huấn luyện trên dữ liệu TP.HCM, không fine-tune trên video CDnet), báo cáo riêng giao thức fine-tune ngắn | F-measure theo hạng mục, theo đúng cách tính của CDnet; đặt cạnh số công bố của AE-NE, ghi rõ khác giao thức | P1 |
| A-X2 | X | Benchmark thứ hai | LASIESTA hoặc BMC 2012 | F-measure | P2 |
| A-AB1 | AB | Prior mềm | L1 cứng / Laplace NLL / Gaussian NLL / β-NLL (Seitzer et al., ICLR 2022) với β = 0.5 | IoU; AUROC của σ | P0 |
| A-AB2 | AB | Ràng buộc nhiều frame | Không L\_shared / K frame cùng ngày liên tiếp / K frame khác ngày | IoU, ghost rate, riêng lát cao điểm | P0 |
| A-AB3 | AB | K | K ∈ {2, 4, 8} | IoU, bộ nhớ GPU | P1 |
| A-AB4 | AB | Loss trên mask | Chỉ L\_sparse / L\_excl / L\_excl + L\_bin; τ ∈ {0.02, 0.05, 0.1} | IoU, tỷ lệ M trung bình ở lát kẹt | P0 |
| A-AB5 | AB | Trọng số khi làm sạch prior | Median thường của B̂ / chỉ 1/σ² / (1 − M)/σ² | Ghost rate vòng 1 | P1 |
| A-AB6 | AB | Khởi tạo lại hay train tiếp giữa các vòng | Hai lựa chọn | Ghost rate, giờ GPU | P2 |
| A-AB7 | AB | Backbone | DINOv3 ViT-S / ViT-B / backbone H1 (nếu có) / CNN ImageNet (ResNet-50) | IoU | P1 |
| A-AB8 | AB | Lịch huấn luyện | Có pha khóa σ / mở σ từ đầu | AUROC σ, độ ổn định (std giữa seed) | P1 |
| A-AB9 | AB | Tiền xử lý | Bỏ căn chỉnh / bỏ chuẩn hóa màu | IoU, tỷ lệ M sai trên mặt đường trống | P1 |
| A-R1 | R | Prior càng xấu thì sao? | BDB lên prior **lúc train** (ghost, geometric, slot\_shift) mức 1–5 | IoU, AUROC σ | P0 |
| A-R2 | R | Frame xấu? | FCS lúc test | IoU | P1 |
| A-R3 | R | Ít ngày dữ liệu? | Số ngày có sẵn cho mỗi (camera, slot): 2 / 4 / 8 / tất cả | IoU, ghost rate | P1 |
| A-G1 | G | Tổng quát? | G-region, G-day2night | IoU | P1 |
| A-S1 | S | Quy mô | RS-5 số camera train | IoU | P2 |
| A-E1 | E | Triển khai | fps suy luận so với MOG2, SuBSENSE, AE-NE; thời gian để có kết quả trên một camera **mới** (AE-NE phải train lại, A không) | fps, phút | P0 |
| A-Q1 | Q | Hình chính | Prior có ghost, σ̄, prior sau làm sạch, ghost người gán | Hình 4 cột | P0 |
| A-Q2 | Q | Lỗi | Xe đậu cố định, mặt đường ướt, đèn pha ban đêm | Hình + phân tích | P0 |

**Thống kê cho A-M2:** AUROC tính theo pixel nhưng CI bootstrap theo cụm camera, vì pixel trong cùng ảnh tương quan rất mạnh.

### V.6. Thực nghiệm Hướng B — gộp nhãn yếu đa nguồn

**Định vị lại tính mới:** WRENCH đã có HMM và Conditional HMM làm label model cho bài toán gán nhãn chuỗi, nên "HMM + ngữ cảnh" tự nó **không đủ mới** cho Q1. Đóng góp nên chuyển sang: (1) bài toán và các nguồn nhãn đặc thù thị giác giao thông, trong đó có nguồn từ background với độ tin cậy đo được; (2) ngữ cảnh là **điều kiện quan sát** (ngày/đêm, IR, mật độ) làm thay đổi độ chính xác của từng nguồn, kiểm chứng định lượng bằng gold set; (3) phân tích độ tin cậy của từng nguồn theo điều kiện — chính là bằng chứng về giới hạn của background median. Mô hình gộp nhãn của ta phải so trực tiếp với Conditional HMM của WRENCH.

**Baseline gộp nhãn** (dùng cài đặt WRENCH, `pip install ws-benchmark`, thích ứng dữ liệu): Majority Voting, Weighted MV, Dawid–Skene, Data Programming, MeTaL, FlyingSquid, EBCC, HMM, Conditional HMM. Thêm: từng LF riêng lẻ, VLM lớn nhất chạy được dùng một mình, và **cận trên có giám sát** (mô hình cuối học trực tiếp trên gold, cross-validation 5 lần theo cụm).

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| B-M1 | M | Label model nào tốt nhất? | Mọi baseline gộp nhãn vs mô hình của ta, trên gold-test | Accuracy, macro-F1, QWK; **ECE** và reliability diagram của nhãn mềm | P0 |
| B-M2 | M | Mô hình cuối có vượt mọi nguồn? | Mô hình cuối học từ nhãn mềm của từng label model; so với LF riêng, VLM riêng, cận trên có giám sát | macro-F1, QWK; tách theo lát RS-6 | P0 |
| B-M3 | M | Đánh đổi độ phủ và độ chính xác | Ngưỡng tin cậy trên nhãn mềm từ 0 đến 0.9 | Đường cong độ phủ (% frame có nhãn) vs độ chính xác | P1 |
| B-AB1 | AB | Mỗi LF đóng góp gì? | Bỏ từng LF (5 lần) | macro-F1 label model và mô hình cuối | P0 |
| B-AB2 | AB | Biến ngữ cảnh nào quan trọng? | Không ngữ cảnh / chỉ IR / chỉ nhóm giờ / chỉ mật độ / chỉ loại đường / đầy đủ | macro-F1, log-likelihood trên gold-dev | P0 |
| B-AB3 | AB | Chuỗi Markov | Không / có; ma trận chuyển chung / theo nhóm giờ | macro-F1, độ "nhảy" nhãn giữa 2 bước liên tiếp | P0 |
| B-AB4 | AB | Phụ thuộc giữa LF1 và LF4 | LF4 dùng LF1 / LF4 độc lập (chỉ độ tương đồng) | macro-F1 | P1 |
| B-AB5 | AB | VLM | Qwen2-VL 2B / 7B / (lớn hơn nếu đủ GPU); 1 prompt / 2 prompt nhất quán | macro-F1 của LF3, độ phủ, giờ GPU | P1 |
| B-AB6 | AB | Huấn luyện mô hình cuối | Nhãn mềm / nhãn cứng (argmax) / lọc max q ≥ 0.4 | macro-F1, ECE | P1 |
| B-AB7 | AB | Lượng nhãn người để chỉnh ngưỡng LF | gold-dev 0 / 50 / 100 / 200 frame | macro-F1 | P1 |
| B-R1 | R | Label model có tự giảm trọng số nguồn hỏng? | Làm hỏng LF2 bằng BDB mức 1–5 (bg xấu → LF2 sai) | macro-F1 label model; trọng số học được của LF2 theo mức | P0 |
| B-R2 | R | Một nguồn đột ngột hỏng theo ngữ cảnh | Lật ngẫu nhiên 50% nhãn của LF1 chỉ ban đêm | Độ chính xác ước lượng của LF1 ban đêm; macro-F1 | P1 |
| B-R3 | R | Frame xấu | FCS (ảnh hưởng LF1, LF3, mô hình cuối) | macro-F1 | P2 |
| B-R4 | R | Lấy mẫu thưa | RS-3 (ảnh hưởng LF4 và chuỗi Markov) | macro-F1 | P1 |
| B-G1 | G | Tổng quát? | Mô hình cuối với G-region, G-day2night, G-temporal | macro-F1, QWK | P1 |
| B-S1 | S | Chi phí và lợi ích của VLM | Giờ GPU của LF3 vs mức tăng macro-F1 | Bảng | P1 |
| B-Q1 | Q | Nguồn nào đáng tin khi nào? | Heatmap độ chính xác ước lượng của từng LF × ngữ cảnh, đặt cạnh độ chính xác thật đo trên gold | Hình chính của paper | P0 |
| B-Q2 | Q | Lỗi | Frame mà mọi nguồn cùng sai | Hình + phân tích | P1 |

**Kiểm tra tính hợp lệ của B-Q1:** độ chính xác label model tự ước lượng cho từng LF phải được đối chiếu với độ chính xác thật trên gold-test (tương quan Spearman qua các ô ngữ cảnh). Nếu tương quan thấp, hình B-Q1 không được dùng làm bằng chứng.

### V.7. Thực nghiệm Hướng C — phát hiện sự cố

**Luận điểm:** (1) gộp feature theo thời gian + mô hình bình thường theo camera × nhóm giờ phát hiện sự cố kéo dài với báo động giả thấp; (2) phát hiện sớm (độ trễ tính bằng phút); (3) bền khi thời tiết và mùa thay đổi. Vì sự cố thật hiếm, mọi kết quả trên sự cố thật phải báo cáo **số sự kiện** và CI (Wilson cho tỷ lệ), không chỉ phần trăm.

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| C-M1 | M | Phát hiện sự kiện tổng hợp | Vật cản kéo dài M ∈ {5, 10, 20} frame; đối chứng thoáng qua 1 frame | Tỷ lệ phát hiện, độ trễ (frame và phút), tỷ lệ báo nhầm với vật thoáng qua | P0 |
| C-M2 | M | Sự cố thật | Mọi phương pháp, trên tập ứng viên đã duyệt mù | Precision, recall xấp xỉ, độ trễ, FA/camera/ngày; theo loại sự cố | P0 |
| C-M3 | M | Ngập | Đối chiếu với `flood_reports.csv` | Recall theo sự kiện ngập, độ trễ so với thời điểm báo cáo | P1 |
| C-M4 | M | Đường cong vận hành | Quét ngưỡng phân vị 97% → 99.9% | Đường recall (tổng hợp + thật) theo FA/camera/ngày — **hình chính** | P0 |
| C-AB1 | AB | Gộp thời gian | W ∈ {1, 3, 5, 10}; median / mean / không gộp | Như M1, M4 | P0 |
| C-AB2 | AB | Mức chi tiết của mô hình bình thường | Một ngân hàng chung / theo camera / theo camera × nhóm giờ | FA/camera/ngày tại cùng recall | P0 |
| C-AB3 | AB | Kích thước coreset | 1 / 5 / 10 / 100% | Như M4, bộ nhớ, thời gian | P1 |
| C-AB4 | AB | Tính kéo dài | N ∈ {1, 2, 3, 5} | Như M1, M4 | P0 |
| C-AB5 | AB | Backbone | DINOv3 ViT-B / DINOv2 ViT-B / WideResNet-50 ImageNet (mặc định của PatchCore) / backbone H1 | Như M4 | P1 |
| C-AB6 | AB | Giảm chiều | PCA 64 / 128 / 256 / không | Như M4, bộ nhớ | P2 |
| C-AB7 | AB | Làm sạch ngân hàng | 1 lượt / 2 lượt | Như M4 | P1 |
| C-AB8 | AB | Gộp điểm cửa sổ | top-k với k ∈ {1, 5, 10, 50} / trung bình | Như M4 | P2 |
| C-AB9 | AB | Thêm background vào ngân hàng | Có / không | Như M4 | P2 |
| C-AB10 | AB | Nhánh lỗi camera | Có / không | FA của loại sự cố giao thông | P1 |
| C-R1 | R | Lấy mẫu thưa | RS-3 | Như M1 (độ trễ tính bằng phút) | P0 |
| C-R2 | R | Thời tiết làm báo động giả? | FCS loại rain, fog, low\_light, glare áp lên cửa sổ bình thường | FA/camera/ngày tăng thêm | P0 |
| C-R3 | R | Khởi động lạnh | Ngân hàng xây từ 3 / 7 / 14 / 28 ngày | Như M4 | P1 |
| C-R4 | R | Trôi theo mùa | Ngân hàng từ mùa khô, đánh giá mùa mưa; so với ngân hàng cập nhật trượt (thêm ngày bình thường gần nhất, bỏ ngày cũ nhất mỗi tuần) | FA/camera/ngày theo tuần | P1 |
| C-E1 | E | Vận hành | Bộ nhớ ngân hàng mỗi camera, thời gian chấm điểm mỗi cửa sổ, tổng cho 608 camera | MB, ms | P1 |
| C-Q1 | Q | Thư viện sự kiện | Mỗi loại sự cố 3 ví dụ: lưới frame + bản đồ điểm patch | Hình | P0 |
| C-Q2 | Q | Báo động giả | 10 báo động giả điển hình, phân nhóm nguyên nhân | Bảng + hình | P0 |

**Lưu ý thống kê:** recall trên sự cố thật chỉ là **xấp xỉ** (tập ứng viên khai thác từ chính các phương pháp, cộng mẫu ngẫu nhiên). Báo cáo rõ cách xây tập và dùng recall trên sự kiện tổng hợp (C-M1) làm số đo chính cho so sánh có kiểm soát.

### V.8. Thực nghiệm Hướng D — dự báo trên đồ thị camera

**Công cụ:** [BasicTS](https://github.com/zezhishao/BasicTS) có sẵn DCRNN, Graph WaveNet, STID, STAEformer, MegaCRN… với pipeline đánh giá thống nhất — nên tích hợp dữ liệu vào BasicTS thay vì tự cài baseline. Một số bản BasicTS ghi hỗ trợ PyTorch trong khoảng phiên bản giới hạn, nên dựng **môi trường riêng** cho D, tách khỏi môi trường DINOv3 (chỉ trao đổi qua file `X[T, N, F]`).

**Baseline:** trung bình lịch sử (cùng thứ, cùng ô giờ); giá trị gần nhất; ARIMA (50 camera); LSTM từng nút; DCRNN; Graph WaveNet; **STID** (baseline đơn giản nhưng mạnh, không dùng đồ thị); STAEformer.

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| D-M1 | M | Dự báo tốt đến đâu? | Mọi baseline vs mô hình đề xuất, chế độ transductive, h = 15/30/60 phút | MAE, RMSE trên `level_exp`; macro-F1, QWK mức tại t + h | P0 |
| D-M2 | M | Đúng so với nhãn người? | 300 frame gold trong giai đoạn test (IV.11) | macro-F1, QWK theo h | P0 |
| D-M3 | M | Cảnh báo khởi phát kẹt | Đầu onset; baseline: ngưỡng trên dự báo `level_exp`, trung bình lịch sử | Precision, recall, F1, thời gian báo trước | P0 |
| D-M4 | M | Camera chưa thấy | Chế độ inductive (IV.11) | Như M1 | P1 |
| D-AB1 | AB | Đồ thị có thật sự giúp? | Không đồ thị / khoảng cách đường đi / chỉ học được / cả hai / **đồ thị ngẫu nhiên cùng mật độ cạnh** (control) | Như M1 | P0 |
| D-AB2 | AB | Đặc trưng nút | Chỉ vô hướng (`level_exp`) / chỉ embedding / cả hai | Như M1 | P0 |
| D-AB3 | AB | Xử lý dữ liệu thiếu | Không kênh mask, không che nút / chỉ mask / mask + che nút lúc train | Như M1 trên dữ liệu có thiếu thật | P0 |
| D-AB4 | AB | Độ dài đầu vào | 6 / 12 / 24 bước | Như M1 | P1 |
| D-AB5 | AB | Nguồn đích | Mô hình cuối B / H4 / ρ\_proxy | M2 (so nhãn người) | P1 |
| D-AB6 | AB | Đầu onset | Có / không (chỉ ngưỡng trên dự báo) | Như M3 | P1 |
| D-R1 | R | Camera hỏng lúc test | Che ngẫu nhiên 0 / 10 / 20 / 30 / 50% nút trong cửa sổ đầu vào | Đường cong MAE theo tỷ lệ | P0 |
| D-R2 | R | Đầu vào nhiễu | Cộng nhiễu vào `q0..q3` (mô phỏng mô hình B sai) với độ lệch 0.05 / 0.1 / 0.2 | MAE | P1 |
| D-R3 | R | Bước lưới thô hơn | 5 / 10 / 15 phút | MAE theo h tính bằng phút | P1 |
| D-R4 | R | Ngày bất thường | Ngày lễ; ngày mưa lớn (nối dữ liệu mưa lịch sử theo giờ — cần chọn nguồn) | MAE riêng các ngày này | P1 |
| D-S1 | S | Lượng lịch sử | 2 / 4 / 8 / tất cả tuần train | Như M1 | P1 |
| D-S2 | S | Mật độ mạng camera | Giữ 25 / 50 / 100% camera (lấy mẫu theo cụm) | Như M1 | P2 |
| D-E1 | E | Chi phí | Tham số, thời gian train/epoch, độ trễ suy luận cho toàn mạng | — | P1 |
| D-Q1 | Q | Lan truyền ùn tắc | Bản đồ mức dự báo vs thật tại t, t + 15, t + 30 phút cho một sự kiện kẹt | Hình | P0 |
| D-Q2 | Q | Mô hình học được gì? | Ma trận kề học được so với khoảng cách đường đi; trọng số attention theo không gian | Hình | P2 |

**Rò rỉ cần chặn thêm:** đặc trưng nút dùng embedding PCA — PCA fit chỉ trên giai đoạn train; trung bình lịch sử chỉ tính trên giai đoạn train; mô hình B tạo đích chỉ huấn luyện trên giai đoạn train (IV.11).

### V.9. Thực nghiệm Hướng E — điều kiện hóa bằng thống kê background

**Luận điểm:** (1) thống kê toàn cục của background giúp tổng quát sang camera và điều kiện chưa thấy; (2) bền hơn hẳn cách dùng background ở mức pixel; (3) cái giúp là **nội dung cảnh không có xe**, không chỉ là "có thông tin về camera".

**Control quan trọng nhất (E-AB6):** thay bản mô tả từ ảnh background bằng bản mô tả tính từ **trung bình feature của 50 frame thường** của chính camera đó (có xe). Nếu hai cách cho kết quả như nhau, background không đóng góp gì riêng — và luận điểm (3) sụp. Đây là câu reviewer chắc chắn hỏi.

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| E-M1 | M | Có tổng quát tốt hơn? | `none`, H3 `early`, H3 `late`, E `film` / `prompt` / `xattn`; G-cluster, G-region, G-day2night; 2 bài toán | MAE (đếm); macro-F1, QWK (mức ùn tắc) | P0 |
| E-M2 | M | So với thích ứng lúc test | TENT (chỉ bài toán mức ùn tắc); fine-tune với k ∈ {10, 50} frame có nhãn của camera mới | Như M1 | P0 |
| E-AB1 | AB | Dạng điều kiện hóa | `film` / `prompt` / `xattn` | Như M1, sạch và BDB mức 4 | P0 |
| E-AB2 | AB | Thành phần bản mô tả | Chỉ mean road / mean + std road / + mean toàn ảnh; có / không trung bình cắt; 1 slot / 3 slot; có / không trọng số σ̄ | Như M1 | P1 |
| E-AB3 | AB | Độ sâu FiLM | L ∈ {1, 2, 4, 8, 12} | Như M1 | P1 |
| E-AB4 | AB | Tăng cường lúc train | Bỏ từng loại: bg-dropout, bg-swap, chèn ghost | Như M1 sạch và dưới BDB | P0 |
| E-AB5 | AB | Bộ mã hóa bản mô tả | DINOv3 đóng băng / dùng chung encoder đang fine-tune | Như M1 | P2 |
| E-AB6 | AB | Background có đóng góp riêng? | Bản mô tả từ bg / từ trung bình feature 50 frame thường / từ một frame ngẫu nhiên / embedding camera ID học được (chỉ có ở camera train → ở camera test dùng embedding trung bình) | Như M1 trên G-cluster và G-region | P0 |
| E-R1 | R | Bền với bg xấu? | BDB đủ loại cho `film`, `xattn`, H3 `early`, H3 `late` | Đường cong + D + s\* — **hình chính** | P0 |
| E-R2 | R | Bản mô tả có ổn định? | Cosine giữa z sạch và z sau ghost 5–50%, có / không trung bình cắt | Đường cong | P1 |
| E-R3 | R | Frame xấu | FCS | Suy giảm tương đối | P2 |
| E-S1 | S | Ít camera train | RS-5: 10 / 25 / 50 / 100% camera | Mức chênh E − `none` theo % camera | P1 |
| E-E1 | E | Chi phí | Bản mô tả cache theo (camera, slot): chi phí suy luận thêm gần 0; báo cáo thời gian tính cache cho 608 camera × 24 slot | — | P2 |
| E-Q1 | Q | z mã hóa gì? | t-SNE của z; tương quan Spearman giữa thành phần chính của z và chiều cao box xe trung bình | Hình | P1 |
| E-Q2 | Q | Lỗi | Camera test mà E tệ hơn `none` — có điểm gì chung (bg tệ, góc nhìn lạ)? | Bảng + hình | P1 |

### V.10. Hướng F — Technical Validation cho bộ dữ liệu

Scientific Data không đòi phương pháp mới, nhưng đòi chứng minh dữ liệu **đúng, đầy đủ, dùng được**. Mỗi mục dưới đây là một tiểu mục trong phần Technical Validation.

| ID | Kiểm tra | Cách làm | Kết quả trình bày | Ưu tiên |
| --- | --- | --- | --- | --- |
| F-TV1 | Độ phủ dữ liệu | Số frame theo camera × ngày; tỷ lệ frame hỏng, IR; phân phối khoảng lấy mẫu | Heatmap camera × ngày; bảng thống kê | P0 |
| F-TV2 | Độ đúng thời gian | So `slot` tính từ `ts_utc` với tên file; kiểm tra lệch đồng hồ camera bằng độ sáng theo giờ (bình minh/hoàng hôn) | Tỷ lệ khớp; danh sách camera lệch | P0 |
| F-TV3 | Chất lượng background | Tập audit: ghost rate theo slot, loại đường; so median (r0) với bản đã làm sạch (r2) | Bảng + hình — tận dụng A-M3 | P0 |
| F-TV4 | Đồng thuận người gán | Fleiss' κ mức ùn tắc; IoU giữa người gán cho mask xe và ghost | Bảng | P0 |
| F-TV5 | Phân bố nhãn | Phân bố mức ùn tắc theo giờ, loại đường; số xe theo loại | Biểu đồ | P0 |
| F-TV6 | Không trùng lặp giữa split | Perceptual hash: đếm cặp ảnh gần giống nhau nằm ở hai split khác nhau | Số cặp (mục tiêu 0 ngoài các camera cùng cụm) | P0 |
| F-TV7 | Riêng tư | Duyệt tay 1.000 ảnh ngẫu nhiên sau làm mờ: tỷ lệ khuôn mặt, biển số còn đọc được | Tỷ lệ kèm CI Wilson | P0 |
| F-TV8 | Metadata camera | Duyệt tay tọa độ của 50 camera ngẫu nhiên bằng ảnh bản đồ vệ tinh / Street View | Sai số vị trí trung bình | P1 |
| F-TV9 | Benchmark | Bảng tóm tắt baseline cho từng tác vụ (đếm, mức ùn tắc, sự cố, dự báo) lấy từ các hướng | Bảng | P0 |

### V.11. Ánh xạ sang bảng/hình của paper và thứ tự chạy

Điền mã thí nghiệm vào trường `paper_target` trong registry để `make_tables.py` sinh đúng bảng. Đề xuất gom hướng thành paper như sau.

| Paper | Hướng | Bảng chính | Hình chính | Phụ lục |
| --- | --- | --- | --- | --- |
| P-1: tiêm prior background không tin cậy cho đếm và nhận diện ùn tắc | H3 + E | Kết quả chính (H3-M1, H3-M2, E-M1); tổng quát hóa (E-M1 các split, E-M2, H3-G1); ablation (E-AB1–AB6, H3-AB1–AB4) | Đường cong BDB của 4 cách tiêm (H3-R1, E-R1); đường cong học (E-S1); t-SNE (E-Q1) | FCS, nhãn nhiễu, hiệu năng, lỗi |
| P-2: gộp nhãn yếu đa nguồn có điều kiện quan sát | B + H4 | So sánh label model (B-M1); mô hình cuối (B-M2, H4-M1); vượt nhãn yếu (H4-M2); ablation (B-AB1–AB3, H4-AB1, H4-AB5) | Heatmap độ tin cậy nguồn × ngữ cảnh (B-Q1); reliability diagram (B-M1); chuỗi thời gian (H4-Q1); B-R1 | VLM, lấy mẫu thưa, tổng quát hóa |
| P-3: phân rã cảnh từ prior không hoàn hảo | A (+ H1 nếu qua cổng quyết định) | Kết quả chính (A-M1); phát hiện lỗi prior (A-M2); làm sạch prior (A-M3); CDnet (A-X1); ablation (A-AB1–AB5, AB9) | Hình 4 cột (A-Q1); đường cong BDB (A-R1); lỗi (A-Q2) | A-M4, R2, R3, G1, E1, H1 |
| P-4: dự báo ùn tắc từ mạng camera | D | Kết quả chính (D-M1, D-M2); cảnh báo khởi phát (D-M3); ablation (D-AB1–AB3) | Đường cong camera hỏng (D-R1); bản đồ lan truyền (D-Q1) | D-M4, R2–R4, S1 |
| P-5: phát hiện sự cố | C | Tổng hợp (C-M1); thật (C-M2); ablation (C-AB1, AB2, AB4) | Đường cong vận hành (C-M4); thư viện sự kiện (C-Q1) | R1–R4, E1 |
| P-6: bộ dữ liệu | F | F-TV1–TV9 | Heatmap độ phủ; mẫu ảnh | — |

**Thứ tự chạy cho agent:**

1. **Giai đoạn 0 — hạ tầng:** cài IV.2–IV.5 và V.1; viết test nghiệm thu; cache embedding DINOv3; chạy `--smoke` cho mọi registry P0. Song song: người gán làm road mask, tập audit, gold set.
2. **Giai đoạn 1 — thí nghiệm quyết định (1 seed):** thống kê khoảng lấy mẫu (F-TV1); ghost rate trên audit (F-TV3); H1-M1; sanity check của A (IV.8); E-AB6. Kết quả giai đoạn này quyết định giữ hay bỏ H1, A, D.
3. **Giai đoạn 2 — P-1 và P-2:** mọi thí nghiệm P0 của H3, E, B, H4, đủ 3 seed.
4. **Giai đoạn 3 — P-3:** P0 của A; H1 nếu qua cổng quyết định.
5. **Giai đoạn 4 — P-4, P-5:** P0 của D (sau khi mô hình B ổn định), P0 của C (khi có dữ liệu mùa mưa).
6. **Giai đoạn 5:** P1 của mọi hướng; P2 nếu còn compute; F-TV khi chuẩn bị P-6.

**Quy tắc khóa kết quả:** trước giai đoạn 2, ghi `configs/exp/final_*.yaml` cho từng paper (IV.13). Sau khi đã mở tập test cho một paper, mọi thay đổi phương pháp phải ghi lại kèm lý do và báo cáo trung thực trong paper.

### V.12. Nguồn tham khảo

**Đã đọc trực tiếp hoặc xác nhận qua trang chính thức, kết quả tìm kiếm (10/2026):**

| Nguồn | Dùng cho |
| --- | --- |
| [Sauvalle & de La Fortelle — AE-NE, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf) | Baseline gần nhất của Hướng A; giao thức CDnet 2014, LASIESTA, BMC 2012 |
| [Chan et al. — Foreground Extraction from Imperfect Backgrounds (arXiv)](https://arxiv.org/pdf/1808.08210) | Related work Hướng A |
| [WRENCH — NeurIPS 2021 Datasets & Benchmarks](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/1c9ac0159c94d8d0cbedc973445af2da-Abstract-round2.html) | Baseline label model của Hướng B, gồm HMM và Conditional HMM |
| [BasicTS (GitHub)](https://github.com/zezhishao/BasicTS) | Baseline Hướng D: DCRNN, Graph WaveNet, STID, STAEformer |
| [Hướng dẫn DINOv3 (Roboflow)](https://blog.roboflow.com/train-dinov3/) và [tài liệu LightlyTrain](https://docs.lightly.ai/train/stable/pretrain_distill/models/dinov3.html) | Các biến thể ViT-S/B/L, giấy phép DINOv3 riêng (không phải MIT/Apache) — cần đọc điều khoản trước khi công bố trọng số dẫn xuất |
| [Vo et al. — bộ dữ liệu xe Đà Nẵng, CMC 2026](https://www.techscience.com/cmc/online/detail/26156/pdf) | Detector VN, kiểm tra khác thành phố |
| [UIT-VinaDeveS22 — CTU Journal](https://ctujs.ctu.edu.vn/index.php/ctujs/article/download/461/612) | Detector VN, kiểm tra khác thành phố |

**Xác nhận qua danh mục tài liệu tham khảo của bài AE-NE:** CDnet 2014 (Wang et al., CVPRW 2014); LASIESTA (Cuevas et al., CVIU 2016); BMC 2012 (Vacavant et al.); SuBSENSE (St-Charles et al., IEEE TIP 2015); PAWCS (St-Charles et al., IEEE TIP 2016); BGSLibrary (Sobral, 2013); Kendall & Gal (NeurIPS 2017); Seitzer et al. về pitfalls của heteroscedastic NLL (2022); Mandal & Vipparthi về đánh giá không phụ thuộc cảnh (IEEE T-ITS 2020).

**Ghi theo hiểu biết chung, chưa kiểm chứng trong đợt này — tra DOI trước khi đưa vào paper:** DINOv2 (Oquab et al.); iBOT (Zhou et al., ICLR 2022); AttMask (Kakogeorgiou et al., ECCV 2022); MAE (He et al., CVPR 2022); ImageNet-C (Hendrycks & Dietterich, ICLR 2019) và thư viện `imagecorruptions`; PatchCore (Roth et al., CVPR 2022); Dawid & Skene (1979); Snorkel (Ratner et al., VLDB 2017); Conditional HMM cho gán nhãn yếu (Li et al., ACL 2021); DCRNN (Li et al., ICLR 2018); FiLM (Perez et al., AAAI 2018); TENT (Wang et al., ICLR 2021); LaMa (Suvorov et al., WACV 2022); Omnimatte (Lu et al., CVPR 2021); Layered Neural Atlases (Kasten et al., 2021); ControlNet (Zhang et al., ICCV 2023); Qwen2-VL; Highway Capacity Manual.

## III. Lộ trình và thứ tự ưu tiên

Nên đầu tư 1–2 tháng đầu vào nền móng dùng chung (road mask, tập audit, gold set), vì mọi hướng đều cần và nó trả lời luôn câu hỏi "background tin được đến đâu". Sau đó đi từ hướng nhanh, an toàn (E, B) đến hướng độ mới cao (A) rồi hướng impact lớn (D, C, F).

&#91;embedded content: Lộ trình 4 giai đoạn · nền móng dùng chung → 3 paper\]

H1 đặt ở giai đoạn 2 nhưng chỉ viết thành paper nếu thí nghiệm FAM so với AttMask cho kết quả rõ ràng; D chỉ bắt đầu khi đã có mô hình của B.

### Việc nên làm ngay

- [ ] Đo tần suất lấy mẫu thực tế, độ dài chuỗi thời gian và tỷ lệ frame bị thiếu của từng camera
- [ ] Gán road mask cho 608 camera
- [ ] Chạy thử FAM so với AttMask và masking ngẫu nhiên trên một tập nhỏ (quyết định số phận H1)
- [ ] Hỏi đơn vị sở hữu dữ liệu về quyền công bố (quyết định có làm F hay không)

### Lưu ý chung khi nhắm Q1

- CVPR, ECCV, NeurIPS là hội nghị, không phải tạp chí Q1; nếu mục tiêu là Q1 thì nên nhắm tạp chí ngay từ đầu (T-ITS, TR-C, TIP, Pattern Recognition, EAAI, ESWA, Information Fusion).
- Bỏ cột tự chấm "độ mới" khỏi tài liệu gửi đi; thay bằng mục so sánh với công trình liên quan.
- Các trích dẫn trong tài liệu này được ghi theo hiểu biết chung — cần tra lại năm, hội nghị và nội dung trước khi đưa vào paper.
