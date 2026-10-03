# Hướng 2 mới — Phân rã cảnh giao thông không cần ảnh nền, tự học từ chuỗi ảnh

Oct 3, 2026 · @Hung Do

Nền của mỗi camera được học ra từ chính chuỗi ảnh nhiều ngày — một ảnh cảnh tĩnh cộng vài ảnh cơ sở ánh sáng — còn phương tiện là phần ngoại lai không lặp lại; không dùng ảnh nền median ở bất kỳ bước nào. Đây là bản phát triển của Hướng 2 (Scene Decomposition) trong [tài liệu đề xuất chung](https://claude.ai/code/artifact/0ff3f2c5-4116-47ea-9d41-4fdff8edc4f6), dùng thành phần TAM của [Hướng G](https://claude.ai/code/artifact/df7c5eb4-bca2-4eed-8397-ea4173c0351c).

## 1. Bối cảnh và vì sao bỏ được ảnh nền

Phiên bản hiện tại của Hướng 2 giám sát nhánh nền bằng ảnh background median. Median sai đúng ở những lúc quan trọng nhất: xe đứng yên lâu ở giờ kẹt bị "nuốt" vào nền (ghost), bóng đổ dịch chuyển, mặt đường ướt, camera bị lệch. Mô hình học theo median sẽ học luôn các lỗi đó.

**Vì sao bỏ được ảnh nền.** Với camera cố định, nền của mọi frame là cùng một cảnh tĩnh, chỉ khác ánh sáng, bóng đổ, độ ướt. Các biến đổi này có **ít bậc tự do** — mô tả được bằng vài hệ số cho mỗi frame. Phương tiện thì khác nhau ở mọi frame. Nếu buộc nền phải nằm trong một không gian ít chiều riêng cho camera, xe sẽ không thể "chui" vào nền. Đây là giả định nền tảng của background subtraction dạng tái tạo (nền là đa tạp ít chiều), và cũng là cách NeRF-W xử lý ảnh du lịch: embedding ngoại hình cho ánh sáng, embedding riêng cho vật tạm thời.

**Hai chế độ sử dụng:**

| Chế độ | Cần gì | Dùng khi |
| --- | --- | --- |
| T — khớp theo camera | Lịch sử ảnh **không nhãn** của camera (vài ngày) | Camera đã lắp — đúng thực tế vận hành, vì camera cố định luôn có lịch sử |
| S — một frame | Chỉ một frame | Camera mới chưa có lịch sử, hoặc cần chạy nhanh |

**Quan hệ với bản dùng prior median (II.A trong tài liệu chung):** nếu bản không cần ảnh nền thắng ở cổng quyết định (mục 12), nó trở thành phương pháp chính của paper Hướng 2; bản dùng prior median là biến thể "khi có sẵn prior".

## 2. Rà soát README code hiện tại

README `direction2_scene_decomposition/` có 8 điểm cần sửa trước khi viết paper:

| # | Điểm trong README | Vấn đề | Sửa thành |
| --- | --- | --- | --- |
| 1 | "Dùng ảnh background thật làm **ground-truth** giám sát nhánh nền" (λ\_bg = 1.5) | Median sai đúng ở giờ kẹt; mô hình học luôn cả ghost. Gọi là "ground-truth" sẽ bị reviewer bác | Bỏ hẳn L\_bg; nền học từ chuỗi ảnh (mục 7) |
| 2 | Gọi là "Unsupervised Road Inpainting" | Đang giám sát bằng ảnh nền tính sẵn → đúng ra là giám sát yếu bằng nhãn giả nhiễu | Bản mới mới đúng nghĩa không giám sát |
| 3 | `--img_size 256` (ảnh vuông) | Frame 16:9 bị méo; xe máy ở xa vốn đã nhỏ | 448 × 256, giữ tỷ lệ |
| 4 | `--freeze_backbone True` | Nhánh nền phải "đoán" mặt đường dưới xe — cần biểu diễn tốt hơn; đóng băng toàn bộ hạn chế điều này | Fine-tune 4 block cuối hoặc LoRA (ablation) |
| 5 | `--match_strategy route_hourly` ghép 1 frame với 1 ảnh nền | Không khai thác chuỗi nhiều ngày | Lấy mẫu theo camera, nhiều ngày (mục 8) |
| 6 | DataParallel, chọn "best model" theo loss huấn luyện | DataParallel chậm; loss huấn luyện không phản ánh chất lượng tách | DDP; chọn checkpoint theo IoU trên audit-val |
| 7 | `density_mask.png` | Đây là mask alpha, không phải mật độ | Đổi tên `alpha_mask.png` |
| 8 | Độ mới ⭐⭐⭐⭐⭐, nhắm CVPR/ECCV/NeurIPS | Đó là hội nghị, không phải tạp chí Q1; đã có Omnimatte, AE-NE và các phương pháp robust PCA học sâu | Định vị lại (mục 3) |

## 3. Công trình liên quan và khoảng trống

Tra cứu tháng 10/2026; mục có link đã xác nhận qua trang bài báo hoặc trang chính thức.

| Nhánh | Công trình | Ý chính | Ta học gì / khác gì |
| --- | --- | --- | --- |
| Cảnh tĩnh + vật tạm thời + ánh sáng | NeRF-W (Martin-Brualla et al., CVPR 2021) — mô tả qua [khảo sát NeRF](https://arxiv.org/pdf/2501.13104) | Embedding ngoại hình theo từng ảnh cho ánh sáng; embedding riêng cho vật tạm thời | Mượn ý "mã ánh sáng theo frame" — camera cố định nên không cần 3D |
|  | [RobustNeRF (Sabour et al., CVPR 2023)](https://arxiv.org/pdf/2302.00833) | Coi vật tạm thời là ngoại lai; huấn luyện như IRLS với trimmed least squares và giả định ngoại lai liền khối | Dùng làm loss giai đoạn 1; tác giả nêu giới hạn kém hiệu quả thống kê trên dữ liệu sạch |
|  | [NeRF On-the-go (Ren et al., CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.pdf) | Dự đoán độ bất định theo pixel để lọc vật tạm thời; ghi nhận NeRF-W, RobustNeRF giảm mạnh khi tỷ lệ che khuất cao | Đúng tình huống giờ kẹt — lý do cần thêm TAM và lấy mẫu nhiều ngày |
| Nền tái tạo, không giám sát | [AE-NE (Sauvalle & de La Fortelle, WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf) | Autoencoder học nền như đa tạp ít chiều cho **từng video**; ước lượng nhiễu nền; trọng số bootstrapping; tác giả nêu thất bại khi vật lớn đứng yên lâu | Baseline chính; ta học **một mô hình cho cả thành phố** + chế độ một frame |
| Robust PCA học sâu | [Deep-unfolded RPCA (Luong et al., EUSIPCO 2020)](https://arxiv.org/pdf/2010.00929); [masked RPCA unfolding (Joukovsky et al.)](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/Interpretable_Neural_Networks_for_Video_Separation_Deep_Unfolding_RPCA_With_Foreground_Masking.pdf); CORONA | Mở vòng lặp RPCA thành mạng; bản masked RPCA nhân mask thưa với thành phần hạng thấp thay vì cộng | Cùng giả định hạng thấp + thưa; chủ yếu thử trên video ngắn, dữ liệu tổng hợp |
| Phân rã lớp từ tập ảnh | [DTI-Sprites (Monnier et al., ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Monnier_Unsupervised_Layered_Image_Decomposition_Into_Object_Prototypes_ICCV_2021_paper.pdf) | Học nguyên mẫu đối tượng và nền như ảnh học được, ghép lớp có che khuất | Ý "nền là tham số học được" — ta học nền riêng cho mỗi camera |
| Phân rã video thành lớp | Omnimatte (CVPR 2021); [Generative Omnimatte (CVPR 2025)](https://cvpr.thecvf.com/virtual/2025/poster/34367); OmnimatteZero (SIGGRAPH Asia 2025) | Phân rã video thành lớp RGBA; bản 2025 dùng video diffusion và **cần mask đối tượng đầu vào** | Cần video liên tục và mask — không hợp dữ liệu thưa, không nhãn |
| Phát hiện rồi học | [CutLER (Wang et al., CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Cut_and_Learn_for_Unsupervised_Object_Detection_and_Instance_Segmentation_CVPR_2023_paper.pdf) | Tạo mask thô không giám sát rồi huấn luyện detector với loss bỏ qua vùng bị sót, tự huấn luyện nhiều vòng | Mượn khuôn "khớp rồi chưng cất" và DropLoss cho giai đoạn 2 |

**Khoảng trống:** chưa thấy công trình nào (trong phạm vi tra cứu) phân rã cảnh giao thông **không cần ảnh nền, không cần video liên tục, không cần mask**, học từ **ảnh chụp thưa nhiều ngày** của hàng trăm camera cố định, đồng thời cho ra mô hình **một frame** dùng được cho camera mới. Tình huống che khuất nặng (giờ kẹt) — điểm yếu đã được ghi nhận của các phương pháp robust — là trọng tâm đánh giá.

## 4. Mô hình hóa bài toán và điều kiện định danh

**Mô hình tạo ảnh.** Frame t của camera c:

```latex
I_t = M_t \odot F_t + (1 - M_t) \odot B_t, \qquad
B_t = \operatorname{clip}_{[0,1]}\Big( E_{c,0} + \sum_{j=1}^{J} \ell_{t,j}\, E_{c,j} \Big)
```

- E\_{c,0}: "cảnh tĩnh trung bình" của camera c — ảnh học được, độ phân giải đầy đủ 256 × 448.
- E\_{c,1..J}: J ảnh cơ sở mô tả biến đổi của nền (nắng/râm, ướt/khô, bóng cây) — học được, độ phân giải một nửa (128 × 224) rồi nội suy lên, ràng buộc trơn.
- ℓ\_t ∈ ℝ^J: **mã ánh sáng** của frame t, vài con số (mặc định J = 4).
- M\_t, F\_t: mask và lớp phương tiện.

Ví dụ: camera 125 có E\_0 là mặt đường lúc trưa nắng; E\_1 làm mọi thứ tối đi (chiều tà); E\_2 làm mặt đường sẫm và bóng (ướt mưa); E\_3 là vệt bóng cây di chuyển theo giờ. Frame 17:20 ngày mưa ≈ E\_0 + 0.6·E\_1 + 0.8·E\_2 + 0.3·E\_3. Một chiếc xe máy đỏ ở góc trái **không thể** biểu diễn bằng tổ hợp này: nếu E\_j chứa chiếc xe đó thì mọi frame khác dùng E\_j cũng bị "dính" xe.

Dung lượng tham số mỗi (camera, chế độ ngày/IR): khoảng 0.69 triệu số thực (2.8 MB fp32) với J = 4 → toàn hệ thống khoảng 3.3 GB, lưu trên đĩa, nạp theo lô camera.

**Điều kiện để bài toán định danh được** — mỗi điều kiện có một thí nghiệm kiểm chứng:

| # | Điều kiện | Ý nghĩa | Kiểm chứng |
| --- | --- | --- | --- |
| C1 | Nền có hạng thấp: J nhỏ, các ảnh cơ sở trơn | Xe (chi tiết cao tần, vị trí thay đổi) không biểu diễn được bằng nền | A2-AB1: quét J ∈ {0, 1, 2, 4, 8}, đo tỷ lệ "xe dính vào nền" |
| C2 | Tỷ lệ chiếm dụng mỗi pixel o\_c(p) — phần frame mà pixel p bị xe che — nhỏ hơn **điểm gãy** κ của bộ ước lượng robust | Giống median: pixel bị che hơn một nửa thời gian thì median hỏng | A2-M2: đường cong sai số nền theo o\_c(p) trên dữ liệu bán tổng hợp |
| C3 | Mã ánh sáng ℓ\_t không mang thông tin xe | Nếu ℓ\_t tự do, nó có thể "vẽ" một chiếc xe buýt to vào nền | Ở giai đoạn 2, ℓ\_t tính chỉ từ feature vùng tĩnh; A2-AB4 kiểm tra rò rỉ |
| C4 | Camera không dịch chuyển trong một cam\_epoch | Cùng vị trí ảnh = cùng điểm cảnh | Căn chỉnh (mục 9); A2-R1 bỏ căn chỉnh |

**Điều kiện C2 là giới hạn cơ bản — phải nói thẳng trong paper.** Không chữa được, nhưng làm giảm được o\_c(p) và đo được nó:

- Lấy mẫu frame trên **mọi giờ của nhiều ngày** (kể cả đêm, sáng sớm) thay vì một slot giờ như median theo slot → o\_c(p) giảm mạnh ở hầu hết pixel.
- Ước lượng o\_c(p) bằng bản đồ hoạt động A\_c(p) của TAM (mục 6) → sinh **bản đồ định danh** cho mỗi camera: pixel có A\_c(p) > κ bị đánh dấu "không định danh được", báo cáo riêng, không đưa vào tuyên bố chính.
- TAM bổ sung cho phần dư màu: một chiếc ô tô **xám** trên mặt đường xám có phần dư pixel nhỏ nên loss robust không loại được, nhưng feature DINO vẫn khác mặt đường → TAM bắt được.

## 5. Kiến trúc

&#91;embedded content: Quy trình hai giai đoạn · khớp theo camera rồi chưng cất sang mạng một frame\]

**Giai đoạn 1 — khớp nền theo camera (không phải mạng nơ-ron sâu).** Tham số của mỗi camera là E\_{c,0..J} và mã ánh sáng ℓ\_t tự do cho từng frame huấn luyện (kiểu GLO, như embedding ngoại hình của NeRF-W). Tối ưu bằng Adam trên GPU, nhiều camera cùng lúc (mỗi camera độc lập, gom thành tensor theo lô). Mask giai đoạn 1 sinh ra từ trọng số robust (mục 7), không do mạng dự đoán.

**Giai đoạn 2 — mạng một frame** (nâng cấp `TrafficDecompositionNet` hiện có):

| Thành phần | Cấu hình |
| --- | --- |
| Encoder | DINOv3 ViT-S/16 hoặc ViT-B/16 (hoặc backbone Hướng G nếu đã qua cổng), fine-tune 4 block cuối |
| Decoder | DPT đa tỉ lệ: token từ block 3, 6, 9, 12 → 256 kênh → fusion từ thô đến mịn → upsample về 448 × 256 |
| Đầu M | 1 kênh, sigmoid |
| Đầu F | 3 kênh, sigmoid |
| Đầu B̂ | 3 kênh, sigmoid — nền đầy đủ kể cả dưới xe |
| Đầu σ | 1 kênh — độ bất định của B̂ (đặc biệt vùng bị xe che); log σ kẹp trong \[log 0.01, log 0.5\] |
| Đầu ℓ̂ | Vector J chiều, tính từ **trung bình token patch tĩnh** (π < 0.2 theo TAM) qua MLP 2 tầng — không nhìn vùng xe, để thỏa điều kiện C3 |

Tổng tham số khoảng 86M (ViT-B) + 10–15M (decoder); ViT-S khoảng 22M để thử nhanh.

## 6. TAM — tín hiệu tiền cảnh từ feature

TAM (Temporal Atypicality Map) là thành phần của [Hướng G](https://claude.ai/code/artifact/df7c5eb4-bca2-4eed-8397-ea4173c0351c) (mục 5 bên đó có đặc tả đầy đủ). Hướng 2 mới chỉ cần TAM chạy trên **DINOv3 đóng băng** — không cần chờ G huấn luyện xong.

**Cách hoạt động, tóm tắt:**

1. Frame đã căn chỉnh, 448 × 256 → DINOv3 ViT-B/16 đóng băng → 448 token patch → PCA về 64 chiều, chuẩn hóa L2.
2. Với mỗi (camera, chế độ ngày/IR, vị trí patch) lưu K = 4 "trạng thái" trong không gian feature: tâm, tần suất, độ phân tán; cập nhật trực tuyến.
3. Trạng thái **tĩnh** = vừa thường xuyên (tần suất ≥ 0.15) vừa chặt (độ phân tán nhỏ). Mặt đường khô, ướt, có bóng tạo cụm như vậy; xe đa dạng về màu, loại, vị trí nên không tạo được.
4. Độ khác thường a\_t(p) = khoảng cách cosine tới trạng thái tĩnh gần nhất, chia cho độ phân tán của trạng thái đó.
5. GMM 2 thành phần trên log a\_t → xác suất tiền cảnh π\_t(p) ∈ \[0, 1\].
6. Bản đồ hoạt động A\_c(p) = tỷ lệ thời gian vị trí p không ở trạng thái tĩnh ≈ tỷ lệ chiếm dụng o\_c(p).

**Hướng 2 mới dùng TAM ở 4 chỗ:**

| Chỗ dùng | Vai trò |
| --- | --- |
| Trọng số robust giai đoạn 1 | Nhân (1 − π\_t)^γ — giảm trọng số pixel nhiều khả năng là xe, kể cả xe cùng màu nền |
| Tỷ lệ cắt κ\_t | Frame có nhiều tiền cảnh (kẹt xe) cắt nhiều pixel hơn |
| Bản đồ định danh | Pixel có A\_c(p) > 0.5 bị đánh dấu không định danh được |
| Đầu mã ánh sáng ℓ̂ ở giai đoạn 2 | Chỉ gộp token patch có π < 0.2 → mã ánh sáng không "nhìn" thấy xe (điều kiện C3) |

**Lưu ý:** TAM không dùng nhãn và không dùng ảnh nền. Thành phần này phải qua Cổng 1 của Hướng G (AUROC trên audit-val không thấp hơn Δ pixel) trước khi Hướng 2 mới dựa vào nó; nếu chưa qua, chạy giai đoạn 1 với γ = 0 (chỉ dùng phần dư) và ghi rõ.

## 7. Hàm loss — không có ảnh nền ở bất kỳ đâu

### Giai đoạn 1: khớp nền robust theo camera

Phần dư theo pixel r\_t(p) = trung bình 3 kênh của |I\_t(p) − B\_t(p)|. Trọng số **inlier** cập nhật lại mỗi bước (IRLS), không qua gradient, theo kiểu RobustNeRF — cắt bỏ phần dư lớn nhất và giả định ngoại lai liền khối:

```latex
\bar{r}_t = \operatorname{AvgPool}_{8\times 8}(r_t), \qquad
w^{\text{res}}_t(p) = \operatorname{sigmoid}\!\Big(\frac{Q_t(1-\kappa_t) - \bar{r}_t(p)}{s}\Big), \qquad
w_t(p) = w^{\text{res}}_t(p)\,\big(1 - \pi_t(p)\big)^{\gamma}
```

- Q\_t(1 − κ\_t): phân vị của r̄\_t trong frame t. **Tỷ lệ cắt κ\_t thích ứng theo frame**: κ\_t = clip(trung bình π\_t + 0.05, 0.05, 0.6) — frame kẹt xe cắt nhiều hơn frame vắng. RobustNeRF dùng phân vị cố định; thích ứng theo TAM là một điểm mới cần ablation (A2-AB2).
- π\_t: xác suất tiền cảnh từ TAM (mục 6); γ = 1.
- s = 0.02 (độ mềm).

```latex
\mathcal{L}_1 = \frac{\sum_{t,p} w_t(p)\, \rho\big(I_t(p) - B_t(p)\big)}{\sum_{t,p} w_t(p)} + \lambda_{TV}\sum_{j\ge 1}\operatorname{TV}(E_{c,j}) + \lambda_{\ell}\sum_t \lVert \ell_t \rVert^2 + \lambda_{\perp}\sum_{1\le i<j}\cos^2(E_{c,i}, E_{c,j})
```

ρ là Charbonnier. Mặc định λ\_TV = 0.01, λ\_ℓ = 1e-3, λ\_⊥ = 0.1.

**Khởi tạo không dùng thuật toán trích nền:** E\_{c,0} = trung bình có trọng số (1 − π\_t) của các frame; E\_{c,j≥1} = nhiễu nhỏ; ℓ\_t = 0. Ablation A2-AB3 so với khởi tạo bằng median và khởi tạo ngẫu nhiên, để chứng minh kết quả không phụ thuộc khởi tạo.

**Đầu ra giai đoạn 1** cho mỗi frame: nền B\_t^(1) (đầy đủ cả dưới xe, vì nền là tổ hợp các ảnh cơ sở), mask M\_t^(1) = 1 − w\_t, độ tin cậy c\_t(p) = |w\_t^res(p) − 0.5| × 2 (gần 0 khi mơ hồ), và bản đồ định danh U\_c(p) = 1 nếu A\_c(p) > 0.5.

### Giai đoạn 2: chưng cất sang mạng một frame

Î = M ⊙ F + (1 − M) ⊙ B̂.

| Loss | Công thức / cách tính | Vai trò | Trọng số |
| --- | --- | --- | --- |
| L\_rec | 0.85·(1 − SSIM)/2 + 0.15·L1 giữa Î và I | Ghép lại đúng ảnh gốc | 1.0 |
| L\_bgPL | NLL Laplace giữa B̂ và B^(1) với σ học được, chỉ trên pixel có U\_c = 0 | Học nền đầy đủ, kể cả "đoán" mặt đường dưới xe; σ lớn ở chỗ nhãn giả không chắc | 1.0 |
| L\_maskPL | BCE giữa M và M^(1), nhân c\_t; **DropLoss**: bỏ phạt ở pixel mạng dự đoán là xe trong khi M^(1) = 0 nhưng π\_t > 0.5 | Học mask; cho phép mạng tìm ra xe mà giai đoạn 1 bỏ sót (mượn ý CutLER) | 0.5 |
| L\_ℓ | ‖ℓ̂ − ℓ\_t‖² | Chưng cất mã ánh sáng | 0.1 |
| L\_excl | Trung bình M · exp(−‖I − sg\[B̂\]‖₁ / τ), τ = 0.05 | Chỗ nền đã giải thích được pixel thì không gọi là xe — thay cho phạt diện tích, vốn phạt oan khi kẹt xe | 0.5 |
| L\_tv, L\_bin | Total variation của M; trung bình M(1 − M) | Mask mượt, gần nhị phân | 0.01; 0 → 0.05 từ epoch 10 |

### Vòng tự cải thiện

Sau giai đoạn 2, thay (1 − π\_t)^γ trong trọng số giai đoạn 1 bằng (1 − M̂\_t) do mạng dự đoán, khớp lại E\_c, sinh nhãn giả mới, huấn luyện tiếp giai đoạn 2. Mặc định 2 vòng; dừng khi IoU trên audit-val tăng dưới 0.5 điểm.

## 8. Quy trình hai giai đoạn

```python
# ---------- Giai đoạn 1: khớp nền theo camera (song song nhiều camera trên GPU) ----------
for batch_of_cams in chunk(all_cameras, size=32):                     # mỗi camera độc lập
    for c in batch_of_cams:
        frames[c] = sample_frames(c, n=600, hours='all', min_days=10, mode=m)
        pi[c]     = tam.posterior(c, frames[c])                         # mục 6, không gradient
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

**Quy định dữ liệu quan trọng:**

| Mục | Quy định | Lý do |
| --- | --- | --- |
| Camera được khớp giai đoạn 1 | **Mọi camera**, kể cả val/test — giai đoạn 1 không dùng nhãn | Đánh giá chế độ T trên camera test là hợp lệ: chỉ dùng ảnh không nhãn của chính camera đó, giống vận hành thực tế |
| Camera dùng nhãn giả để train giai đoạn 2 | **Chỉ camera train** | Chế độ S phải được đánh giá trên camera chưa thấy |
| Lấy mẫu frame giai đoạn 1 | 600 frame mỗi (camera, chế độ), rải đều mọi giờ của chế độ, ít nhất 10 ngày | Giảm tỷ lệ chiếm dụng o\_c(p) (điều kiện C2) |
| Ảnh IR | Khớp riêng (chế độ `ir`), E riêng | Ảnh hồng ngoại khác hẳn về màu |
| Camera đổi góc (cam\_epoch mới) | Khớp lại từ đầu cho epoch mới | Điều kiện C4 |
| Giai đoạn 2 | AdamW, lr 1e-4 decoder / 2e-5 encoder, weight decay 0.05, cosine, bf16, 448 × 256, 60.000 bước | Cấu hình khởi điểm, chỉnh trên audit-val |

**Suy luận chế độ T** (camera đã có E\_c): với frame mới, giải ℓ\_t bằng bình phương tối thiểu có trọng số trên pixel tĩnh (J = 4 ẩn, 3–5 vòng IRLS, dưới 5 ms), dựng B\_t, rồi mask = hợp nhất phần dư robust và đầu M của mạng. **Suy luận chế độ S:** chỉ chạy mạng.

**Chi phí ước tính (cần đo):** giai đoạn 1 khoảng vài phút GPU cho mỗi (camera, chế độ) với 600 frame × 2.000 vòng ở 256 × 448; khoảng 1.200 cặp (camera, chế độ) → vài chục giờ GPU cho toàn hệ thống, chạy một lần mỗi vòng tự cải thiện.

## 9. Dữ liệu và tập đánh giá

Tóm tắt những gì cần từ hạ tầng chung (đầy đủ ở mục IV của tài liệu đề xuất chung). Agent cài các mục này **trước** code giai đoạn 1.

**Chỉ mục và tiền xử lý:** `data/index/frames.parquet` với `frame_id`, `camera_id`, `ts_utc`, `day` và `slot` theo giờ Asia/Ho\_Chi\_Minh, `is_corrupt`, `is_ir`, `dx`, `dy`, `align_ok`, `cam_epoch`. Căn chỉnh bằng phase correlation trên vùng tĩnh có kết cấu (vùng tĩnh tính từ phương sai theo thời gian, **không** lấy phần bù road mask vì vỉa hè có xe đậu): ≤ 4 px bỏ qua; 4–16 px warp; > 16 px loại. Road mask = lòng đường xe chạy, không gồm vỉa hè.

**Chia dữ liệu theo cụm địa lý:** camera cùng nút giao (hoặc trong bán kính 150 m) thuộc cùng một cụm; 70 / 10 / 20% số camera cho train / val / test, phân tầng theo loại đường và quận; dùng chung `configs/splits.json`.

**Năm tập đánh giá** (chỉ trên camera test, trừ khi ghi khác):

| Tập | Quy mô | Ground truth | Dùng cho |
| --- | --- | --- | --- |
| Audit — mask xe | Khoảng 500 frame từ 60 camera, phân tầng theo loại đường và nhóm giờ (đêm / thấp điểm / cao điểm); SAM + người duyệt; 10% gán đôi, IoU giữa người gán ≥ 0.8 | Mask xe | IoU, F1 của M |
| Bán tổng hợp **kiểm soát chiếm dụng** | Khoảng 2.000 ảnh + bộ dữ liệu cho A2-M2 | Mask xe và mặt đường dưới xe | PSNR/LPIPS vùng che; đường cong điểm gãy |
| Frame trống thật | 300–500 frame (detector đếm 0 xe, người duyệt) | Mặt đường thật | PSNR, LPIPS, KID của B̂ |
| Gold chiếm dụng | 800–1.000 frame: mask xe ∩ road mask | Tỷ lệ chiếm dụng | MAE (A2-M5) |
| CDnet 2014 | Hạng mục baseline, badWeather, lowFramerate, nightVideos, intermittentObjectMotion, shadow, cameraJitter | Mask theo chuẩn CDnet | F-measure (A2-X1) |

**Cách tạo tập bán tổng hợp (`synth_occupancy.py`):**

1. **Nền:** frame gần như trống của camera test (detector đếm ≤ 1 xe, người duyệt), 3–5 frame mỗi camera.
2. **Xe:** crop có mask từ tập audit của **camera train**, gom thành thư viện vài nghìn xe máy, ô tô, xe buýt.
3. **Phối cảnh:** hồi quy "tọa độ y → chiều cao xe" từ box detector trên frame thật của camera đó; scale crop theo y; chỉ dán trong road mask.
4. **Hòa trộn:** Poisson blending, bóng đổ giả nhẹ, điều chỉnh màu theo độ sáng nền.
5. **Mật độ:** 3 mức phủ 5–15%, 15–35%, 35–60% diện tích mặt đường.
6. **Kiểm soát chiếm dụng (cho A2-M2):** tạo chuỗi frame cho giai đoạn 1 trong đó một vùng cố định bị xe (màu, loại ngẫu nhiên) che trong x% số frame, x = 10, 20, … 80; nền thật của vùng đó biết trước.

Hạn chế cần nêu: ảnh dán không giống hệt ảnh thật (thiếu che khuất phức tạp, phản chiếu), nên tập bán tổng hợp chỉ là một trong năm cách đánh giá.

**Metric:** IoU, F1, boundary F (dung sai 2 px) cho mask; PSNR và LPIPS **chỉ trong vùng bị xe che** cho nền; KID thay cho FID khi ít hơn 2.000 ảnh; tỷ lệ "xe dính vào nền" = phần diện tích xe audit mà B̂ giống frame hơn giống mặt đường.

**Thống kê:** 3 seed cho thí nghiệm P0; khoảng tin cậy 95% bằng bootstrap theo **cụm camera**; Wilcoxon theo cụm với baseline mạnh nhất, hiệu chỉnh Holm trong mỗi bảng; cùng ngân sách chỉnh siêu tham số cho mọi phương pháp học được. Tập test chỉ được đọc trong `eval_a2.py`.

## 10. Spec cho agent

**Sửa các file đang có trong `direction2_scene_decomposition/`:**

| File | Hiện tại | Sửa thành |
| --- | --- | --- |
| `dataset.py` | `DecompositionDataset` ghép origin với background theo `route_hourly` | `FrameDataset` đọc `frames.parquet`, trả frame 448 × 256 đã căn chỉnh, `cid` (camera, chế độ, epoch) và nhãn giả giai đoạn 1 nếu có. Giữ class cũ làm baseline H2 gốc, gọi bằng cờ `--legacy_bg` |
| `models.py` | `TrafficDecompositionNet`: DINOv3 đóng băng, 3 nhánh | Thêm decoder DPT, đầu `sigma` và đầu `ell` (gộp token patch tĩnh); tùy chọn `--unfreeze_last 4` |
| `losses.py` | Recon + λ\_bg·L1(nền, ảnh nền) + sparsity + TV | Bỏ hạng λ\_bg; thêm `L_bgPL` (Laplace NLL), `L_maskPL` có DropLoss, `L_ell`, `L_excl`, `L_bin`; recon dùng SSIM + L1 |
| `train.py` | DataParallel, chọn "best" theo loss | DDP (`torchrun`); chọn checkpoint theo IoU trên audit-val; giữ ảnh tiến trình mỗi epoch |
| `infer.py` | Xuất `clean_road`, `vehicles_only`, `density_mask` | Thêm `--mode S` hoặc `--mode T --stage1_dir`; đổi `density_mask` → `alpha_mask`; thêm `uncertainty.png` |

**File mới:**

| File | Nội dung |
| --- | --- |
| `scene_fit.py` | Giai đoạn 1: class `SceneBasis`, hàm `robust_weights`, CLI khớp theo lô camera |
| `stage1_store.py` | Lưu/đọc E\_c (npz) và nhãn giả từng frame (fp16, zarr hoặc npz theo camera) |
| `solve_ell.py` | Suy luận chế độ T: giải ℓ\_t bằng bình phương tối thiểu có trọng số trên pixel tĩnh |
| `selftrain.py` | Điều phối vòng tự cải thiện |
| `synth_occupancy.py` | Dữ liệu bán tổng hợp có kiểm soát chiếm dụng (mục 9) |
| `eval_a2.py` | Mọi metric của mục 11 |

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

**Nghiệm thu** (dữ liệu đồ chơi: nền kết cấu cố định, hình chữ nhật màu ngẫu nhiên làm "xe", 50 "ngày" × 40 frame):

1. Nền × 3 yếu tố ánh sáng biết trước + hình chữ nhật phủ 20%: giai đoạn 1 khôi phục nền **dưới các hình chữ nhật** với PSNR > 35 dB và mask IoU > 0.9.
2. Một vị trí bị hình chữ nhật (màu ngẫu nhiên) che trong x% frame, x = 10 → 80: sai số nền thấp khi x nhỏ, tăng vọt quanh x ≈ κ — xác nhận điểm gãy như phân tích C2.
3. Hình chữ nhật cùng màu nền (lệch ±2%): chỉ dùng phần dư thì bỏ sót; thêm prior TAM thì bắt được.
4. `solve_ell` khôi phục ℓ biết trước với sai số < 1e-3.
5. Đầu `ell` không đổi khi thay đổi các patch có π > 0.5 (kiểm tra C3 bằng test).
6. Dataloader giai đoạn 2 không đọc nhãn giả của camera val/test — assert theo `splits.json`.

## 11. Kế hoạch thực nghiệm

Mức ưu tiên: **P0** bắt buộc; **P1** reviewer thường hỏi; **P2** phụ lục. Kết quả chính 3 seed.

**Baseline:** Δ trên median theo slot; Δ trên median mọi giờ; MOG2, KNN (OpenCV); SuBSENSE, PAWCS (BGSLibrary); AE-NE (khớp riêng mỗi camera test); kiểu RobustNeRF (= giai đoạn 1 với κ cố định, không TAM); detector huấn luyện trên dữ liệu xe Việt Nam + inpainting LaMa; H2 gốc (L1 với ảnh nền, cờ `--legacy_bg`); bản dùng prior median có σ (II.A trong tài liệu chung); deep-unfolded RPCA nếu có code công khai (P2).

| ID | Loại | Câu hỏi | Thiết lập | Metric | Ưu tiên |
| --- | --- | --- | --- | --- | --- |
| A2-M1 | Chính | Tách xe và xóa xe tốt đến đâu? | Mọi baseline; chế độ T và S | IoU, F1, boundary F; PSNR/LPIPS vùng che; KID; tách theo cao điểm, đêm, mưa, mật độ | P0 |
| A2-M2 | Chính | Điểm gãy khi pixel bị che nhiều | Dữ liệu kiểm soát chiếm dụng: một vùng bị xe che trong x% frame, x = 10, 20, … 80 | Sai số nền tại vùng đó theo x, cho: median theo slot, median mọi giờ, AE-NE, giai đoạn 1 không TAM, giai đoạn 1 đầy đủ — **hình chính** | P0 |
| A2-M3 | Chính | Chế độ T cần bao nhiêu lịch sử? | Số ngày 1 / 3 / 7 / 14 / 30; số frame 100 / 300 / 600 / 1.200 | IoU, PSNR nền; thời gian khớp | P0 |
| A2-M4 | Chính | Một frame có bằng khớp theo camera? | Chế độ S trên camera chưa thấy vs chế độ T trên cùng camera | IoU, PSNR, tốc độ | P0 |
| A2-M5 | Chính | Có ích cho ứng dụng? | Chiếm dụng tính từ M trong road mask | MAE so với gold chiếm dụng | P1 |
| A2-X1 | Benchmark ngoài | So với số công bố | CDnet 2014, chế độ T **theo từng video** — cùng thiết lập theo video như AE-NE; TAM tính từ DINOv3 đóng băng trên chính video | F-measure theo hạng mục, đặt cạnh số công bố của AE-NE | P0 nếu nhắm TIP/PR; P1 nếu nhắm T-ITS |
| A2-AB1 | Ablation | Số ảnh cơ sở | J ∈ {0, 1, 2, 4, 8}; độ phân giải E\_j: đầy đủ / nửa / một phần tư | PSNR nền, tỷ lệ "xe dính vào nền" | P0 |
| A2-AB2 | Ablation | Trọng số robust | Charbonnier thuần / κ cố định (RobustNeRF) / κ\_t theo TAM / + hệ số (1 − π) / kích thước pool 1, 4, 8, 16 | IoU, A2-M2 | P0 |
| A2-AB3 | Ablation | Khởi tạo | Trung bình có trọng số TAM / median mọi giờ / ngẫu nhiên | IoU, số vòng hội tụ | P1 |
| A2-AB4 | Ablation | Rò rỉ xe vào mã ánh sáng | Đầu ℓ̂ gộp từ patch tĩnh / từ mọi patch | Tỷ lệ "xe dính vào nền" trong B̂ | P1 |
| A2-AB5 | Ablation | Loss giai đoạn 2 | Bỏ DropLoss / bỏ trọng số tin cậy / bỏ L\_bgPL / bỏ σ | IoU, PSNR | P0 (DropLoss, tin cậy) |
| A2-AB6 | Ablation | Vòng tự cải thiện | R = 0, 1, 2, 3 | IoU theo vòng | P1 |
| A2-AB7 | Ablation | Cách lấy mẫu giai đoạn 1 | Một slot giờ (như median theo slot) / mọi giờ một ngày / mọi giờ nhiều ngày | IoU; sai số theo A\_c(p) trên dữ liệu thật | P0 |
| A2-AB8 | Ablation | Backbone giai đoạn 2 | Đóng băng / mở 4 block / LoRA; ViT-S / ViT-B / backbone Hướng G | IoU | P1 |
| A2-AB9 | Ablation | Kích thước đầu vào | 256 × 256 (README cũ) / 448 × 256 | IoU, IoU riêng xe máy | P1 |
| A2-R1 | Robustness | Không căn chỉnh camera | Bỏ bước căn chỉnh | IoU, PSNR | P1 |
| A2-R2 | Robustness | Cảnh thay đổi lâu dài | Bán tổng hợp: một vật cố định (rào chắn) xuất hiện từ ngày k; khớp với cửa sổ trượt 7 / 14 ngày | Thời gian để nền cập nhật | P2 |
| A2-R3 | Robustness | Frame xấu | 8 loại nhiễu kiểu ImageNet-C × 5 mức lên chế độ S | Suy giảm tương đối | P1 |
| A2-E1 | Chi phí | Có triển khai được? | Phút GPU khớp mỗi camera; dung lượng E\_c; ms suy luận chế độ T và S; so với thời gian AE-NE phải huấn luyện mỗi camera | — | P0 |
| A2-Q1 | Định tính | Ảnh cơ sở học được là gì? | Hiển thị E\_0 … E\_4 của vài camera — kỳ vọng thấy ánh sáng, mặt đường ướt, bóng cây | Hình — thuyết phục về tính diễn giải | P0 |
| A2-Q2 | Định tính | Kết quả phân rã | Lưới: frame, B̂, M, F ở cao điểm, đêm, mưa; so với median và AE-NE | Hình | P0 |
| A2-Q3 | Định tính | Bản đồ định danh và lỗi | U\_c(p) cạnh các trường hợp thất bại (làn luôn kẹt, xe đậu cố định) | Hình | P0 |

**Bảng/hình cho paper:** Bảng 1 = A2-M1; Bảng 2 = A2-X1; Bảng 3 = A2-AB1, AB2, AB5, AB7. Hình chính = đường cong điểm gãy A2-M2; hình ảnh cơ sở A2-Q1; hình lịch sử cần thiết A2-M3.

## 12. Rủi ro, cổng quyết định, lịch, venue

**Phụ thuộc:** chỉ cần **thành phần TAM** của Hướng G (chạy trên DINOv3 đóng băng) — không cần chờ G huấn luyện xong. TAM qua Cổng 1 của G là đủ để bắt đầu.

**Hai cổng quyết định:**

1. **Cổng 1 (khoảng 3 tuần: dữ liệu đồ chơi + 20 camera thật):** giai đoạn 1 trên camera của audit-val phải (a) cho mask IoU ở lát cao điểm không thấp hơn bản dùng prior median có σ, và (b) có điểm gãy trong A2-M2 cao hơn median theo slot. Trượt → giữ bản dùng prior median làm phương pháp chính, và dùng B^(1) của giai đoạn 1 làm **prior tốt hơn** thay median — công sức không mất.
2. **Cổng 2 (sau giai đoạn 2):** chế độ S trên camera chưa thấy kém chế độ T không quá 0.05 IoU. Kém hơn nhiều → paper nhấn mạnh chế độ T (vẫn thực tế vì camera cố định luôn có lịch sử), chế độ S đưa vào phần thảo luận.

**Rủi ro:**

| Rủi ro | Dấu hiệu | Cách giảm |
| --- | --- | --- |
| Ánh sáng không "ít chiều": đèn pha ban đêm, bóng tòa nhà quét qua theo giờ | Mask dương giả theo vệt bóng | Khớp E riêng cho các nhóm giờ (ví dụ 4 nhóm); tăng J; ablation A2-AB1 |
| Pixel bị che quá nhiều (vi phạm C2) | U\_c(p) = 1 trên diện rộng ở vài camera | Báo cáo riêng, không giấu; lấy mẫu thêm giờ đêm |
| Tự huấn luyện khuếch đại lỗi nhãn giả | IoU trên audit-val giảm qua vòng | Giới hạn R ≤ 3; dừng sớm theo audit-val; giữ DropLoss |
| Xe cùng màu mặt đường | Mask thiếu xe xám/đen trên nhựa | Hệ số TAM trong trọng số; A2-AB2 chứng minh |
| Chi phí giai đoạn 1 cho khoảng 1.200 cặp (camera, chế độ) | Vài chục giờ GPU mỗi vòng | Khớp theo lô 32 camera; giảm còn 300 frame nếu A2-M3 cho thấy đủ |

**Lịch khoảng 5 tháng:**

1. **Tháng 1:** `scene_fit.py`, `robust_weights`, `synth_occupancy.py`; nghiệm thu đồ chơi; chạy 20 camera → **Cổng 1**.
2. **Tháng 2:** giai đoạn 1 cho mọi camera; sửa `dataset.py`, `models.py`, `losses.py`, `train.py` theo mục 10; huấn luyện giai đoạn 2.
3. **Tháng 3:** A2-M1–M4 → **Cổng 2**; baseline (AE-NE theo camera, SuBSENSE, PAWCS, detector + LaMa).
4. **Tháng 4:** ablation P0, CDnet 2014 (A2-X1), vòng tự cải thiện.
5. **Tháng 5:** hình, bảng, viết.

**Gộp paper:** đề xuất **một paper Hướng 2** với bản không cần ảnh nền là phương pháp chính và bản dùng prior median là biến thể "khi có prior" — câu chuyện liền mạch từ prior nhiễu đến không cần prior, kèm phân tích điểm gãy.

**Venue:** IEEE TIP, Pattern Recognition, IEEE TCSVT; IEEE T-ITS nếu nhấn mạnh ứng dụng giao thông.

## Nguồn tham khảo

**Đã xác nhận qua trang bài báo hoặc trang chính thức (10/2026):**

- [RobustNeRF — Sabour et al., CVPR 2023](https://arxiv.org/pdf/2302.00833)
- [NeRF On-the-go — Ren et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_NeRF_On-the-go_Exploiting_Uncertainty_for_Distractor-free_NeRFs_in_the_Wild_CVPR_2024_paper.pdf)
- [Khảo sát Neural Radiance Fields for the Real World (mô tả NeRF-W, RobustNeRF, NeRF On-the-go)](https://arxiv.org/pdf/2501.13104)
- [AE-NE — Sauvalle & de La Fortelle, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Sauvalle_Autoencoder-Based_Background_Reconstruction_and_Foreground_Segmentation_With_Background_Noise_Estimation_WACV_2023_paper.pdf)
- [Deep-unfolded reference-based RPCA — Luong et al., EUSIPCO 2020](https://arxiv.org/pdf/2010.00929)
- [Interpretable Neural Networks for Video Separation: Deep Unfolding RPCA with Foreground Masking — Joukovsky et al.](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/Interpretable_Neural_Networks_for_Video_Separation_Deep_Unfolding_RPCA_With_Foreground_Masking.pdf)
- [DTI-Sprites — Monnier et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Monnier_Unsupervised_Layered_Image_Decomposition_Into_Object_Prototypes_ICCV_2021_paper.pdf)
- [Generative Omnimatte — Lee et al., CVPR 2025](https://cvpr.thecvf.com/virtual/2025/poster/34367)
- [OmnimatteZero — SIGGRAPH Asia 2025 (arXiv 2503.18033)](https://arxiv.org/html/2503.18033v3)
- [CutLER — Wang et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Cut_and_Learn_for_Unsupervised_Object_Detection_and_Instance_Segmentation_CVPR_2023_paper.pdf)
- [Bộ dữ liệu xe Việt Nam (Đà Nẵng) — Vo et al., CMC 2026](https://www.techscience.com/cmc/online/detail/26156/pdf)

**Xác nhận qua danh mục tham khảo của bài AE-NE:** CDnet 2014 (Wang et al., CVPRW 2014); SuBSENSE (St-Charles et al., IEEE TIP 2015); PAWCS (St-Charles et al., IEEE TIP 2016); BGSLibrary (Sobral, 2013); Kendall & Gal (NeurIPS 2017).

**Ghi theo hiểu biết chung, chưa kiểm chứng — tra DOI trước khi đưa vào paper:** NeRF-W (Martin-Brualla et al., CVPR 2021, chỉ đọc qua khảo sát); Omnimatte (Lu et al., CVPR 2021); CORONA; LaMa (Suvorov et al., WACV 2022); ImageNet-C (Hendrycks & Dietterich, ICLR 2019); Robust PCA (Candès et al., 2011).
