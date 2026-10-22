"""
=============================================================================
 Bộ Công Cụ Chạy Đối Chứng Per-Image PCA: DINOv3 Gốc vs Student (Hướng G)
 Chạy trực tiếp trên ảnh camera giao thông hoặc toàn bộ thư mục dữ liệu.
 
 Hỗ trợ:
   1. Chạy độc lập DINOv3 Vanilla Pretrained (chưa qua fine-tuning) để xem PCA gốc.
   2. So sánh đối chứng Side-by-Side: [Ảnh Gốc] | [DINO Gốc PCA] | [Student PCA] | [Foreground PCA] | [Độ Sai Khác]
   3. Căn chỉnh dấu Eigenvector (Sign Ambiguity Resolution) để màu sắc nền ổn định.
   4. Tự động tính toán Explained Variance Ratio và độ dịch chuyển biểu diễn.
=============================================================================
"""

import os
# Ngăn ngừa lỗi OpenMP xung đột thư viện trên Windows (bắt buộc đặt trước torch, numpy, matplotlib)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import time
from typing import Optional, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image

# Đảm bảo đường dẫn import tương thích mọi môi trường
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dino_root = os.path.dirname(_current_dir)
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from common.backbone_loader import get_dino_backbone, extract_tokens, imagenet_normalize
from common.gpu_utils import smart_load_state_dict


# =============================================================================
# 1. HÀM TÍNH TOÁN PER-IMAGE PCA CHUẨN META DINOV2
# =============================================================================
def compute_per_image_pca(
    feat: np.ndarray,
    pi: Optional[np.ndarray] = None,
    fg_threshold: float = 0.40,
    grid_h: int = 16,
    grid_w: int = 28,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính PCA chuẩn Meta DINOv2 độc lập trên từng ảnh (Intra-image SVD).
    
    Args:
        feat: Ma trận patch tokens (N_patches, D) của một ảnh đơn lẻ.
        pi: Bản đồ tiền cảnh TAM (N_patches,) trong [0, 1] (nếu có).
        fg_threshold: Ngưỡng lọc patch tiền cảnh cho second-stage PCA.
        grid_h, grid_w: Kích thước lưới patch (16x28 = 448).
        
    Returns:
        std_pca: (grid_h, grid_w, 3) Standard PCA toàn cảnh nội tại của ảnh.
        fg_pca: (grid_h, grid_w, 3) Foreground PCA bóc tách chi tiết các bộ phận xe.
        explained_var: (3,) Tỷ lệ phương sai giải thích được của top 3 thành phần chính.
    """
    N, D = feat.shape
    # Chỉ số 4 góc của ảnh trên lưới patch (dùng làm mỏ neo định hướng dấu eigenvector)
    corners_idx = [0, 1, grid_w - 2, grid_w - 1,
                   N - grid_w, N - grid_w + 1, N - 2, N - 1]

    # Bước 1: Standard PCA toàn cảnh nội tại của ảnh
    f_mean = feat.mean(axis=0, keepdims=True)
    f_cent = feat - f_mean
    u_svd, s_svd, vt_svd = np.linalg.svd(f_cent, full_matrices=False)
    
    # Tính Explained Variance Ratio cho top 3 components
    total_var = np.sum(s_svd ** 2)
    explained_var = (s_svd[:3] ** 2) / np.maximum(total_var, 1e-12)

    # Chiếu lên 3 hướng phương sai lớn nhất
    pca3 = f_cent @ vt_svd[:3].T  # (N, 3)

    # Bước 2: Căn chỉnh dấu Eigenvector (Sign Ambiguity Resolution)
    # Quy ước: Nền ở 4 góc ảnh có giá trị thấp hơn trung bình khung hình
    for c in range(3):
        if pca3[corners_idx, c].mean() > pca3[:, c].mean():
            pca3[:, c] = -pca3[:, c]

    # Chuẩn hóa dải phân vị 1% - 99% để triệt tiêu các điểm ngoại lai cục bộ
    p_min = np.percentile(pca3, 1, axis=0, keepdims=True)
    p_max = np.percentile(pca3, 99, axis=0, keepdims=True)
    std_pca = np.clip((pca3 - p_min) / np.maximum(p_max - p_min, 1e-6), 0.0, 1.0)

    # Bước 3: Foreground Second-Stage PCA (Tập trung bóc tách các bộ phận xe)
    fg_pca = np.full((N, 3), 0.06, dtype=np.float32)  # Nền tối dịu (Dark background)
    if pi is not None:
        pi_flat = pi.reshape(-1)
        fg_mask = (pi_flat > fg_threshold)
        if fg_mask.sum() < 4:
            fg_mask = (pi_flat >= np.percentile(pi_flat, 80))

        if fg_mask.sum() >= 4:
            f_fg = feat[fg_mask]
            f_fg_mean = f_fg.mean(axis=0, keepdims=True)
            f_fg_cent = f_fg - f_fg_mean
            _, _, vt_fg = np.linalg.svd(f_fg_cent, full_matrices=False)
            fg_pca3 = f_fg_cent @ vt_fg[:3].T  # (N_fg, 3)

            f_min = np.percentile(fg_pca3, 2, axis=0, keepdims=True)
            f_max = np.percentile(fg_pca3, 98, axis=0, keepdims=True)
            fg_scaled = np.clip((fg_pca3 - f_min) / np.maximum(f_max - f_min, 1e-6), 0.0, 1.0)
            fg_pca[fg_mask] = fg_scaled
        else:
            fg_pca = std_pca * np.clip(pi_flat[:, None] * 2.0, 0.1, 1.0)
    else:
        # Nếu không có bản đồ TAM pi, tạo foreground mask ước lượng từ norm của gradient feature
        feat_norm = np.linalg.norm(feat, axis=-1)
        norm_p80 = np.percentile(feat_norm, 80)
        est_mask = (feat_norm >= norm_p80)
        if est_mask.sum() >= 4:
            f_fg = feat[est_mask]
            f_fg_cent = f_fg - f_fg.mean(axis=0, keepdims=True)
            _, _, vt_fg = np.linalg.svd(f_fg_cent, full_matrices=False)
            fg_pca3 = f_fg_cent @ vt_fg[:3].T
            f_min = np.percentile(fg_pca3, 2, axis=0, keepdims=True)
            f_max = np.percentile(fg_pca3, 98, axis=0, keepdims=True)
            fg_pca[est_mask] = np.clip((fg_pca3 - f_min) / np.maximum(f_max - f_min, 1e-6), 0.0, 1.0)
        else:
            fg_pca = std_pca.copy()

    return std_pca.reshape(grid_h, grid_w, 3), fg_pca.reshape(grid_h, grid_w, 3), explained_var


def compute_batch_consistent_pca(
    feats_batch: np.ndarray,
    grid_h: int = 16,
    grid_w: int = 28,
) -> np.ndarray:
    """
    Tính PCA đồng bộ trên toàn bộ batch (Batch-Consistent SVD).
    Tất cả các ảnh trong batch được chiếu lên cùng một hệ trục 3D chung.
    Màu sắc các thực thể (mặt đường, xe cộ, bầu trời) hoàn toàn nhất quán xuyên camera.
    
    Args:
        feats_batch: (B, N_patches, D) ma trận patch tokens của toàn bộ batch.
        grid_h, grid_w: Kích thước lưới patch (16x28 = 448).
        
    Returns:
        batch_pca: (B, grid_h, grid_w, 3) ảnh PCA màu sắc đồng bộ hoàn hảo.
    """
    B, N, D = feats_batch.shape
    corners_idx = [0, 1, grid_w - 2, grid_w - 1,
                   N - grid_w, N - grid_w + 1, N - 2, N - 1]

    feats_flat = feats_batch.reshape(B * N, D)
    mean_all = feats_flat.mean(axis=0, keepdims=True)
    feats_cent = feats_flat - mean_all
    _, _, vt_svd = np.linalg.svd(feats_cent, full_matrices=False)

    pca3_all = feats_cent @ vt_svd[:3].T  # (B * N, 3)

    # Căn chỉnh dấu eigenvector theo các góc ảnh của toàn bộ batch
    corners_batch_idx = [b * N + c for b in range(B) for c in corners_idx]
    for c in range(3):
        if pca3_all[corners_batch_idx, c].mean() > pca3_all[:, c].mean():
            pca3_all[:, c] = -pca3_all[:, c]

    # Chuẩn hóa chung toàn batch
    p_min = np.percentile(pca3_all, 1, axis=0, keepdims=True)
    p_max = np.percentile(pca3_all, 99, axis=0, keepdims=True)
    pca3_norm = np.clip((pca3_all - p_min) / np.maximum(p_max - p_min, 1e-6), 0.0, 1.0)
    return pca3_norm.reshape(B, grid_h, grid_w, 3)


# =============================================================================
# 2. DATASET NẠP ẢNH TÙY CHỌN (ĐƠN LẺ HOẶC THƯ MỤC)
# =============================================================================
class SimpleTrafficDataset(Dataset):
    """Dataset đọc ảnh linh hoạt từ đường dẫn file lẻ hoặc danh sách thư mục."""
    def __init__(self, file_paths: List[str], target_size: Tuple[int, int] = (256, 448)):
        self.file_paths = file_paths
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        p = self.file_paths[idx]
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)
                tensor_img = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            # Fallback tạo ảnh placeholder nếu file lỗi
            tensor_img = torch.zeros((3, self.target_size[0], self.target_size[1]), dtype=torch.float32)

        return {
            "img": tensor_img,
            "path": p,
            "filename": os.path.basename(p),
        }


def collect_image_files(input_path: str, max_samples: int = 16) -> List[str]:
    """Thu thập danh sách file ảnh hợp lệ từ file hoặc thư mục."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = []
        for root, _, filenames in os.walk(input_path):
            for fn in sorted(filenames):
                if os.path.splitext(fn)[1].lower() in valid_exts:
                    files.append(os.path.join(root, fn))
                    if len(files) >= max_samples:
                        break
            if len(files) >= max_samples:
                break
        return files
    return []


# =============================================================================
# 3. TRÍCH XUẤT ĐẶC TRƯNG TỪ MÔ HÌNH DINO
# =============================================================================
@torch.no_grad()
def extract_model_patch_tokens(
    model: nn.Module,
    imgs: torch.Tensor,
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Trích xuất ma trận patch tokens từ mô hình DINO.
    
    Args:
        model: Backbone ViT (DINOv3).
        imgs: Tensor (B, 3, H, W) ở dải [0, 1].
        patch_size: Kích thước patch (mặc định 16).
        
    Returns:
        patch_tokens: (B, N_patches, D).
    """
    model.eval()
    x_norm = imagenet_normalize(imgs)
    _, patch_spatial = extract_tokens(model, x_norm, patch_size=patch_size)
    B, Hp, Wp, D = patch_spatial.shape
    return patch_spatial.reshape(B, Hp * Wp, D)


# =============================================================================
# 4. HÀM VẼ SO SÁNH VÀ KẾT XUẤT HÌNH ẢNH
# =============================================================================
def render_comparison_figure(
    samples_data: List[dict],
    out_path: str,
    has_student: bool = True,
):
    """
    Kết xuất biểu đồ đối chiếu PCA giữa DINOv3 gốc và Student.
    
    Nếu has_student = True (6 Cột):
      Cột 1: Ảnh gốc RGB
      Cột 2: DINOv3 Gốc PCA (Pretrained Baseline)
      Cột 3: Student PCA Đồng Bộ Camera (Batch-Consistent PCA)
      Cột 4: Student PCA Nội Tại (Per-Image PCA)
      Cột 5: Foreground-Focused PCA (Chi tiết bộ phận xe cộ)
      Cột 6: Bản đồ sai khác đặc trưng |F_student - F_vanilla|
      
    Nếu has_student = False (5 Cột):
      Cột 1: Ảnh gốc RGB
      Cột 2: DINOv3 Gốc PCA (Tổ hợp 3 thành phần chính)
      Cột 3: Kênh PC1 (Cấu trúc không gian tổng quát)
      Cột 4: Kênh PC2 (Đặc trưng thứ cấp)
      Cột 5: Kênh PC3 (Chi tiết cục bộ)
    """
    n_rows = len(samples_data)
    n_cols = 6 if has_student else 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26 if has_student else 22, 3.8 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, item in enumerate(samples_data):
        img_np = item["img"]  # (H, W, 3)
        H, W = img_np.shape[:2]

        # Cột 1: Ảnh gốc RGB
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Mẫu #{i+1}: {item['filename'][:20]}", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Cột 2: DINOv3 Gốc PCA (Nội suy mượt mà Bilinear)
        vanilla_tensor = torch.from_numpy(item["vanilla_pca"]).permute(2, 0, 1).unsqueeze(0).float()
        vanilla_smooth = F.interpolate(vanilla_tensor, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
        axes[i, 1].imshow(vanilla_smooth)
        var_v = item.get("vanilla_var", [0, 0, 0])
        axes[i, 1].set_title(f"DINOv3 Gốc PCA (Pretrained)\nTop3 Var: {sum(var_v)*100:.1f}%", fontsize=10, fontweight="bold", color="#1f77b4")
        axes[i, 1].axis("off")

        if has_student:
            # Cột 3: Student PCA Đồng Bộ Camera (Batch-Consistent PCA)
            student_batch_p = item.get("student_batch_pca", item["student_pca"])
            sb_tensor = torch.from_numpy(student_batch_p).permute(2, 0, 1).unsqueeze(0).float()
            sb_smooth = F.interpolate(sb_tensor, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
            axes[i, 2].imshow(sb_smooth)
            axes[i, 2].set_title("Student PCA Đồng Bộ Cam\n(Batch-Consistent SVD)", fontsize=10, fontweight="bold", color="#008080")
            axes[i, 2].axis("off")

            # Cột 4: Student PCA Nội Tại (Per-Image PCA)
            student_tensor = torch.from_numpy(item["student_pca"]).permute(2, 0, 1).unsqueeze(0).float()
            student_smooth = F.interpolate(student_tensor, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
            axes[i, 3].imshow(student_smooth)
            var_s = item.get("student_var", [0, 0, 0])
            axes[i, 3].set_title(f"Student PCA Nội Tại\nTop3 Var: {sum(var_s)*100:.1f}%", fontsize=10, fontweight="bold", color="#2ca02c")
            axes[i, 3].axis("off")

            # Cột 5: Foreground-Focused PCA (Object-Centric)
            fg_tensor = torch.from_numpy(item["fg_pca"]).permute(2, 0, 1).unsqueeze(0).float()
            fg_smooth = F.interpolate(fg_tensor, size=(H, W), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
            axes[i, 4].imshow(fg_smooth)
            axes[i, 4].set_title("Student Foreground PCA\n(Bóc tách chi tiết thân xe)", fontsize=10, fontweight="bold", color="#d62728")
            axes[i, 4].axis("off")

            # Cột 6: Bản đồ sai khác đặc trưng |F_student - F_vanilla|
            diff_map = item["diff_map"]  # (16, 28)
            diff_tensor = torch.from_numpy(diff_map).unsqueeze(0).unsqueeze(0).float()
            diff_smooth = F.interpolate(diff_tensor, size=(H, W), mode="bilinear", align_corners=False).squeeze().numpy()
            
            im_diff = axes[i, 5].imshow(diff_smooth, cmap="turbo", vmin=0.0, vmax=max(0.6, float(diff_smooth.max())))
            mean_diff = float(diff_map.mean())
            axes[i, 5].set_title(f"Sai Khác Đặc Trưng\nTB Δ = {mean_diff:.3f}", fontsize=10, fontweight="bold", color="#9467bd")
            axes[i, 5].axis("off")
            if i == 0:
                plt.colorbar(im_diff, ax=axes[i, 5], fraction=0.046, pad=0.04)

        else:
            # Khi chỉ có DINO gốc: Hiển thị 3 kênh PC riêng lẻ
            pca_np = item["vanilla_pca"]  # (16, 28, 3)
            pc_names = ["PC1 (Cấu trúc vĩ mô)", "PC2 (Đặc trưng thứ cấp)", "PC3 (Chi tiết vi mô)"]
            cmaps = ["viridis", "plasma", "inferno"]
            for col_idx, (pc_name, cmap_name) in enumerate(zip(pc_names, cmaps)):
                pc_c = pca_np[..., col_idx]
                pc_tensor = torch.from_numpy(pc_c).unsqueeze(0).unsqueeze(0).float()
                pc_smooth = F.interpolate(pc_tensor, size=(H, W), mode="bilinear", align_corners=False).squeeze().numpy()
                axes[i, 2 + col_idx].imshow(pc_smooth, cmap=cmap_name)
                var_c = var_v[col_idx] * 100 if col_idx < len(var_v) else 0.0
                axes[i, 2 + col_idx].set_title(f"{pc_name}\n(Var: {var_c:.1f}%)", fontsize=10, fontweight="bold")
                axes[i, 2 + col_idx].axis("off")

    main_title = ("ĐỐI CHIẾU PER-IMAGE PCA: DINOv3 GỐC (PRETRAINED) vs STUDENT (HƯỚNG G: TAM + SRS + AGM)"
                  if has_student else
                  "PHÂN TÍCH PER-IMAGE PCA ĐỘC LẬP TRÊN DINOv3 GỐC (VANILLA PRETRAINED)")
    plt.suptitle(main_title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🎉 [KẾT QUẢ] Đã xuất thành công biểu đồ PCA đối chứng tại:\n   -> {os.path.abspath(out_path)}")


# =============================================================================
# 5. HÀM CHÍNH (MAIN CLI ENTRYPOINT)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Script chạy PCA đối chứng DINOv3 gốc (Vanilla) vs Student (Hướng G)."
    )
    parser.add_argument("--img", "--image_path", type=str, default=None,
                        help="Đường dẫn đến 1 ảnh duy nhất để chạy thử nghiệm.")
    parser.add_argument("--data_dir", "--img_dir", type=str, default=None,
                        help="Thư mục chứa ảnh camera giao thông.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn file checkpoint Student (.pth). Nếu không cung cấp, script tự động tìm checkpoint có sẵn.")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16",
                        help="Tên backbone kiến trúc DINOv3 (mặc định: dinov3_vits16).")
    parser.add_argument("--weights", type=str, default=None,
                        help="Đường dẫn trọng số checkpoint baseline gốc nếu có sẵn offline.")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face Token để tải trọng số nếu cần.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Thiết bị tính toán: cuda hoặc cpu.")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Số lượng ảnh mẫu cần trích xuất và hiển thị (mặc định: 4).")
    parser.add_argument("--output", type=str, default="compare_vanilla_vs_student_pca.png",
                        help="Tên file ảnh kết xuất biểu đồ kết quả.")
    parser.add_argument("--img_h", type=int, default=256, help="Chiều cao ảnh chuẩn hóa.")
    parser.add_argument("--img_w", type=int, default=448, help="Chiều rộng ảnh chuẩn hóa.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(" 🚗 [DINOv3 PCA BENCHMARK] CÔNG CỤ CHẠY ĐỐI CHỨNG PER-IMAGE PCA")
    print("=" * 80)
    print(f" • Thiết bị tính toán  : {device}")
    print(f" • Tên mô hình ViT     : {args.model_name}")
    print(f" • Kích thước chuẩn hóa: {args.img_w}x{args.img_h} (Patch size: 16 -> 28x16 = 448 patches)")
    print(f" • File xuất kết quả   : {args.output}")

    # 1. Tìm kiếm ảnh đầu vào
    candidate_files = []
    if args.img and os.path.isfile(args.img):
        candidate_files = [args.img]
    elif args.data_dir and os.path.isdir(args.data_dir):
        candidate_files = collect_image_files(args.data_dir, max_samples=args.num_samples)
    else:
        # Tự động dò tìm các thư mục ảnh quen thuộc trong dự án
        for dir_candidate in [
            "/workspace/output",
            "G:\\nckh\\output",
            "g:\\nckh\\output",
            "direction_data_article/figures",
            "paper_figures",
        ]:
            if os.path.isdir(dir_candidate):
                candidate_files = collect_image_files(dir_candidate, max_samples=args.num_samples)
                if candidate_files:
                    print(f"💡 Tự động tìm thấy {len(candidate_files)} ảnh từ thư mục '{dir_candidate}'.")
                    break

    if not candidate_files:
        print("⚠️ Không tìm thấy ảnh thực tế nào trên đĩa! Đang tạo ảnh mô phỏng Synthetic để chạy kiểm thử...")
        os.makedirs("sample_test_images", exist_ok=True)
        synthetic_path = "sample_test_images/synthetic_traffic_sample.png"
        dummy_arr = np.zeros((256, 448, 3), dtype=np.uint8)
        # Vẽ giả lập đường và 2 chiếc xe
        dummy_arr[100:250, :] = [60, 60, 65]  # Mặt đường xám
        dummy_arr[140:190, 80:160] = [200, 30, 30]   # Xe đỏ
        dummy_arr[160:210, 260:360] = [30, 100, 220] # Xe xanh
        Image.fromarray(dummy_arr).save(synthetic_path)
        candidate_files = [synthetic_path]

    num_eval = min(args.num_samples, len(candidate_files))
    eval_files = candidate_files[:num_eval]
    print(f"✅ Sẽ xử lý {len(eval_files)} ảnh mẫu:")
    for idx, f in enumerate(eval_files):
        print(f"   [{idx+1}] {f}")

    # 2. Tải DINOv3 Vanilla Pretrained (Mô hình gốc đối chứng)
    print("\n📦 [1/2] Đang nạp DINOv3 Gốc (Vanilla Pretrained - Baseline đối chứng)...")
    backbone_vanilla, embed_dim, patch_size = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=args.weights,
        device=device,
        hf_token=args.hf_token,
    )
    backbone_vanilla.eval()
    for p in backbone_vanilla.parameters():
        p.requires_grad = False
    print("   -> Tải thành công DINOv3 Gốc!")

    # 3. Kiểm tra và tải Student Checkpoint (Hướng G) nếu có
    student_ckpt_path = args.checkpoint
    if not student_ckpt_path:
        for candidate_ckpt in [
            "checkpoints/directionG/checkpoint_last.pth",
            "checkpoints/directionG/best_checkpoint.pth",
            "checkpoints/direction1_new/checkpoint_last.pth",
            "checkpoints/direction1_new/best_checkpoint.pth",
            "/workspace/checkpoints/directionG/checkpoint_last.pth",
            "/workspace/checkpoints/directionG/best_checkpoint.pth",
            "/workspace/checkpoints/direction1_new/checkpoint_last.pth",
            "/workspace/checkpoints/direction1_new/best_checkpoint.pth",
            "/workspace/checkpoint_last.pth",
            "/workspace/best_checkpoint.pth",
            "G:\\nckh\\checkpoints\\directionG\\checkpoint_last.pth",
            "G:\\nckh\\checkpoints\\directionG\\best_checkpoint.pth",
            "G:\\nckh\\DINO\\checkpoints\\direction1_new\\best_checkpoint.pth",
            "checkpoint_last.pth",
            "best_checkpoint.pth",
        ]:
            if os.path.isfile(candidate_ckpt):
                student_ckpt_path = candidate_ckpt
                break
    elif not os.path.isfile(student_ckpt_path):
        print(f"❌ [CẢNH BÁO] Không tìm thấy file checkpoint chỉ định: '{student_ckpt_path}'!")

    backbone_student = None
    has_student = False

    if student_ckpt_path and os.path.isfile(student_ckpt_path):
        print(f"\n🔄 [2/2] Đang nạp Student Checkpoint (Hướng G) từ:\n   -> {student_ckpt_path}")
        try:
            backbone_student, _, _ = get_dino_backbone(
                model_name=args.model_name,
                pretrained=False,
                device=device,
            )
            try:
                ckpt = torch.load(student_ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(student_ckpt_path, map_location="cpu")

            bb_sd = ckpt.get("student_backbone", ckpt.get("student", ckpt.get("model_state", ckpt.get("state_dict", ckpt))))
            smart_load_state_dict(backbone_student, bb_sd, strict=False, verbose=False)
            backbone_student.eval()
            for p in backbone_student.parameters():
                p.requires_grad = False
            has_student = True
            print("   -> Tải thành công Student Checkpoint!")
        except Exception as e:
            print(f"⚠️ Không thể nạp checkpoint ({e}). Chuyển sang chế độ chỉ chạy DINO gốc.")
            backbone_student = None
            has_student = False
    else:
        print("\n💡 Không phát hiện Checkpoint Student. Chạy chế độ phân tích độc lập DINOv3 gốc.")

    # 4. Thực thi Forward & Trích xuất PCA
    dataset = SimpleTrafficDataset(eval_files, target_size=(args.img_h, args.img_w))
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False)

    print("\n⚡ Đang thực thi tính toán Per-Image PCA và phân tích phương sai...")
    samples_data = []
    t0 = time.time()

    for batch in loader:
        imgs = batch["img"].to(device)  # (B, 3, H, W)
        filenames = batch["filename"]
        B = imgs.shape[0]

        # Trích xuất patch tokens từ DINO gốc
        tokens_vanilla = extract_model_patch_tokens(backbone_vanilla, imgs, patch_size=patch_size)  # (B, 448, D)
        tokens_vanilla_np = F.normalize(tokens_vanilla.float(), p=2, dim=-1).cpu().numpy()

        # Trích xuất patch tokens từ Student nếu có
        tokens_student_np = None
        student_batch_pca = None
        if has_student and backbone_student is not None:
            tokens_student = extract_model_patch_tokens(backbone_student, imgs, patch_size=patch_size)
            tokens_student_np = F.normalize(tokens_student.float(), p=2, dim=-1).cpu().numpy()
            student_batch_pca = compute_batch_consistent_pca(tokens_student_np, grid_h=16, grid_w=28)

        for b in range(B):
            img_b = imgs[b].permute(1, 2, 0).cpu().numpy().clip(0, 1)
            f_v = tokens_vanilla_np[b]  # (448, D)

            # Tính PCA cho DINO gốc
            pca_v, _, var_v = compute_per_image_pca(f_v, grid_h=16, grid_w=28)

            item = {
                "img": img_b,
                "filename": filenames[b],
                "vanilla_pca": pca_v,
                "vanilla_var": var_v,
            }

            if has_student and tokens_student_np is not None:
                f_s = tokens_student_np[b]  # (448, D)
                # Tính PCA cho Student
                pca_s, fg_s, var_s = compute_per_image_pca(f_s, grid_h=16, grid_w=28)
                item["student_pca"] = pca_s
                item["student_batch_pca"] = student_batch_pca[b] if student_batch_pca is not None else pca_s
                item["fg_pca"] = fg_s
                item["student_var"] = var_s

                # Tính bản đồ sai khác đặc trưng L2 giữa Student và Vanilla
                diff = np.linalg.norm(f_s - f_v, axis=-1).reshape(16, 28)
                item["diff_map"] = diff

                # In thông số định lượng cho từng mẫu
                cos_sim = np.sum(f_s * f_v, axis=-1).mean()
                print(f"   • Mẫu '{filenames[b]}': Cosine Sim Vanilla vs Student: {cos_sim:.4f} | Trung bình L2 Diff: {diff.mean():.4f}")
            else:
                print(f"   • Mẫu '{filenames[b]}': Vanilla Top-3 Explained Var: PC1={var_v[0]*100:.1f}%, PC2={var_v[1]*100:.1f}%, PC3={var_v[2]*100:.1f}%")

            samples_data.append(item)

    elapsed = time.time() - t0
    fps = len(samples_data) / max(elapsed, 1e-4)
    print(f"\n⏱️ Hoàn tất tính toán trong {elapsed:.2f}s ({fps:.1f} FPS).")

    # 5. Xuất hình ảnh biểu đồ
    render_comparison_figure(samples_data, args.output, has_student=has_student)
    print("=" * 80)


if __name__ == "__main__":
    main()
