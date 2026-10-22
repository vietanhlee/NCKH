"""
=============================================================================
  BỘ CÔNG CỤ TRỰC QUAN HÓA SELF-ATTENTION MAPS: DINOv3 GỐC VS STUDENT (HƯỚNG G)
  Nghiên cứu khoa học: Học biểu diễn SSL định hướng phương tiện giao thông.
  
  Tính năng:
    1. Trích xuất Self-Attention Matrix ở Transformer Block cuối cùng (Last Block).
    2. Hỗ trợ RoPE Attention của DINOv3 và chuẩn Attention của DINOv2.
    3. Tách biệt token [CLS] và loại bỏ chính xác Storage/Register Tokens (4 tokens).
    4. Trực quan hóa Multi-Head Attention (6 heads cho ViT-S, 12 heads cho ViT-B).
    5. Hiển thị Mean Attention Map, Alpha-blended Overlay và Thresholded Mask.
    6. So sánh đối chứng Side-by-Side: Vanilla DINOv3 vs Student Checkpoint.
    7. Tính toán định lượng: Attention Entropy (độ sắc nét/tập trung vào vật thể).
=============================================================================
"""

import os
# Ngăn ngừa lỗi xung đột OpenMP trên hệ thống Windows/Docker
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import time
from typing import Optional, Tuple, List, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# Thiết lập đường dẫn import linh hoạt cho môi trường Linux/Docker và Windows
_current_dir = os.path.dirname(os.path.abspath(__file__))
_dino_root = os.path.dirname(_current_dir)
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from common.backbone_loader import get_dino_backbone, imagenet_normalize
from common.gpu_utils import smart_load_state_dict


# =============================================================================
# 1. BỘ THU THẬP ATTENTION MA TRẬN CHUẨN XÁC CHO DINOV3 & DINOV2
# =============================================================================
class ViTAttentionExtractor:
    """
    Trình trích xuất ma trận Attention an toàn từ Transformer Block cuối cùng của ViT.
    Tự động xử lý RoPE Attention trong DINOv3 và loại bỏ tiền tố CLS + Storage Tokens.
    """
    def __init__(self, model: nn.Module, block_idx: int = -1):
        self.model = model
        self.block_idx = block_idx
        self.captured_attn = None
        self._hook_handle = None
        self._orig_compute_attn = None
        self._target_attn_module = None

    def _setup_hook(self):
        """Thiết lập hook hoặc monkey-patch hàm compute_attention của SelfAttention."""
        try:
            # Truy cập tới module attention của block cuối
            block = self.model.blocks[self.block_idx]
            attn_mod = getattr(block, "attn", None)
            if attn_mod is None:
                raise AttributeError("Không tìm thấy thuộc tính .attn trong Transformer Block")
            
            self._target_attn_module = attn_mod
            self._orig_compute_attn = attn_mod.compute_attention

            def custom_compute_attention(qkv: torch.Tensor, attn_bias=None, rope=None):
                B, N, _ = qkv.shape
                C = attn_mod.qkv.in_features
                num_heads = attn_mod.num_heads
                head_dim = C // num_heads

                # Tách q, k, v chuẩn xác
                qkv_r = qkv.reshape(B, N, 3, num_heads, head_dim)
                q, k, v = torch.unbind(qkv_r, 2)
                q, k, v = [t.transpose(1, 2) for t in [q, k, v]]  # (B, H, N, D)

                # Áp dụng RoPE (Rotary Position Embedding) nếu có của DINOv3
                if rope is not None and hasattr(attn_mod, "apply_rope"):
                    q, k = attn_mod.apply_rope(q, k, rope)

                # Tính toán phân phối xác suất Attention Softmax
                scale = float(head_dim) ** -0.5
                attn_weights = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
                self.captured_attn = attn_weights.detach().cpu()

                # Tiếp tục thực thi SDPA gốc của PyTorch để mô hình hoạt động bình thường
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                x = x.transpose(1, 2)
                return x.reshape([B, N, C])

            # Gán hàm patch
            attn_mod.compute_attention = custom_compute_attention

        except Exception as e:
            # Fallback nếu mô hình có API get_last_selfattention (chuẩn DINOv2 PyTorch Hub)
            pass

    def _remove_hook(self):
        """Khôi phục lại trạng thái ban đầu của module."""
        if self._target_attn_module is not None and self._orig_compute_attn is not None:
            self._target_attn_module.compute_attention = self._orig_compute_attn
            self._orig_compute_attn = None
            self._target_attn_module = None

    def extract_cls_attention_maps(
        self,
        x: torch.Tensor,
        grid_h: int = 16,
        grid_w: int = 28,
        patch_size: int = 16,
    ) -> torch.Tensor:
        """
        Thực thi forward pass và trích xuất CLS-to-patch attention maps.

        Args:
            x: Tensor ảnh (B, 3, H, W).
            grid_h: Số lượng patch theo chiều dọc (H // patch_size).
            grid_w: Số lượng patch theo chiều ngang (W // patch_size).
            patch_size: Kích thước patch (mặc định 16 cho DINOv3).

        Returns:
            cls_spatial_attn: Tensor (B, num_heads, grid_h, grid_w) chứa attention map chuẩn hóa.
        """
        self.captured_attn = None
        self._setup_hook()

        try:
            with torch.no_grad():
                # Thực hiện forward pass qua backbone
                _ = self.model(x)
        finally:
            self._remove_hook()

        if self.captured_attn is None:
            # Thử phương thức dự phòng nếu model hỗ trợ get_last_selfattention (DINOv2)
            if hasattr(self.model, "get_last_selfattention"):
                with torch.no_grad():
                    self.captured_attn = self.model.get_last_selfattention(x).detach().cpu()
            else:
                raise RuntimeError("Không thể trích xuất ma trận Attention từ mô hình ViT này.")

        # Phân giải cấu trúc token: [CLS] + [Storage/Register Tokens] + [Patch Tokens]
        attn = self.captured_attn  # Shape: (B, num_heads, N, N)
        B, num_heads, N, _ = attn.shape

        # Xác định số lượng storage tokens (DINOv3 có 4 storage tokens, DINOv2 có 4 registers)
        n_storage = getattr(self.model, "n_storage_tokens", getattr(self.model, "num_register_tokens", 0))
        if n_storage == 0 and N == (1 + 4 + grid_h * grid_w):
            n_storage = 4  # Tự động nhận diện cấu hình DINOv3 tiêu chuẩn

        num_prefix = 1 + n_storage  # 1 cho [CLS] token + n_storage tokens
        expected_patches = grid_h * grid_w

        # Lấy attention từ [CLS] token (dòng 0) tới tất cả các patch tokens
        cls_to_patches = attn[:, :, 0, num_prefix:]  # Shape: (B, num_heads, expected_patches)

        if cls_to_patches.shape[-1] != expected_patches:
            # Nếu có chênh lệch kích thước (do aspect ratio hoặc padding), cắt hoặc căn chỉnh
            cls_to_patches = cls_to_patches[:, :, :expected_patches]

        # Reshape về lưới 2D (B, num_heads, grid_h, grid_w)
        cls_spatial = cls_to_patches.reshape(B, num_heads, grid_h, grid_w)
        return cls_spatial


# =============================================================================
# 2. HÀM TÍNH ĐỘ PHÂN TÁN ATTENTION (ENTROPY & FOCUS METRICS)
# =============================================================================
def compute_attention_entropy(attn_map: np.ndarray, eps: float = 1e-9) -> float:
    """
    Tính Shannon Entropy của bản đồ Attention:
      H = - sum(p * log(p))
    Entropy thấp chứng tỏ sự chú ý tập trung sắc nét vào vật thể cụ thể (xe cộ).
    Entropy cao thể hiện sự chú ý bị phân tán mông lung trên toàn cảnh nền đường/bầu trời.
    """
    p = attn_map.flatten()
    p_sum = p.sum()
    if p_sum > 0:
        p = p / p_sum
    p = np.clip(p, eps, 1.0)
    entropy = -np.sum(p * np.log2(p))
    # Chuẩn hóa về [0, 1] theo log2(N_patches)
    max_entropy = np.log2(len(p))
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# =============================================================================
# 3. DATASET VÀ TIỀN XỬ LÝ ẢNH GIAO THÔNG
# =============================================================================
class SimpleTrafficDataset(Dataset):
    """Dataset nạp và tiền xử lý ảnh camera giao thông chuẩn hóa cho ViT."""
    def __init__(self, file_paths: List[str], target_size: Tuple[int, int] = (256, 448)):
        self.file_paths = file_paths
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.file_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            # Resize giữ trọn khung hình giao thông
            img_resized = img.resize((self.target_size[1], self.target_size[0]), Image.Resampling.BILINEAR)

        arr = np.array(img_resized, dtype=np.float32) / 255.0
        tensor_img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        norm_img = imagenet_normalize(tensor_img)
        if norm_img.ndim == 4:
            norm_img = norm_img.squeeze(0)

        return {
            "img": norm_img,
            "raw_img": arr,
            "path": path,
            "filename": os.path.basename(path),
        }


def collect_image_files(input_path: str, max_samples: int = 4) -> List[str]:
    """Thu thập danh sách file ảnh hợp lệ từ đường dẫn file hoặc thư mục."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    found_files = []

    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in valid_exts:
                    found_files.append(os.path.join(root, f))
                    if len(found_files) >= max_samples:
                        return found_files

    return found_files


# =============================================================================
# 4. HÀM TẠO VÀ XUẤT HÌNH ẢNH SO SÁNH ĐỐI CHỨNG ATTENTION (SIDE-BY-SIDE)
# =============================================================================
def render_attention_comparison_figure(
    img_rgb: np.ndarray,
    attn_vanilla: np.ndarray,      # (num_heads, grid_h, grid_w)
    attn_student: Optional[np.ndarray] = None, # (num_heads, grid_h, grid_w)
    filename: str = "",
    output_path: str = "attention_comparison.png",
    threshold_percentile: float = 0.80,
):
    """
    Kết xuất hình ảnh đối chiếu chất lượng cao giữa DINO gốc và Student:
      - Các Attention Heads độc lập (Head 0 .. 5).
      - Mean Attention Map (Tổng hợp toàn bộ các heads).
      - Overlay Blended Map (Hòa trộn lên ảnh RGB gốc).
      - Thresholded Focus Mask (Lọc top percentile chú ý mạnh nhất).
      - Bản đồ Attention Difference (Độ dịch chuyển chú ý delta).
    """
    num_heads = attn_vanilla.shape[0]
    has_student = attn_student is not None

    # Số hàng: Hàng 1 = Vanilla, Hàng 2 = Student (nếu có), Hàng 3 = So sánh / Hiệu số
    n_rows = 3 if has_student else 1
    # Số cột: Ảnh gốc (1) + Multi Heads (6) + Mean (1) + Overlay (1) + Thresholded (1) = 10 cột
    n_cols = 1 + num_heads + 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.0), dpi=150)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    # Tiêu đề tổng quan
    student_tag = "VS STUDENT (Hướng G)" if has_student else "[CHẾ ĐỘ ĐỘC LẬP]"
    fig.suptitle(
        f"ĐỐI CHIẾU SELF-ATTENTION MAPS: DINOv3 PRETRAINED {student_tag}\n"
        f"File: {filename} | Kích thước phân giải Patch: 16x16 | Ngưỡng lọc trọng tâm: {int((1-threshold_percentile)*100)}%",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    colormap = "inferno"

    def process_attn_visuals(attn_h_w: np.ndarray):
        """Nội suy attention map lên độ phân giải ảnh gốc và chuẩn hóa min-max."""
        # Upsample bilinear bằng PIL
        attn_pil = Image.fromarray(attn_h_w.astype(np.float32))
        attn_upsampled = np.array(attn_pil.resize((img_rgb.shape[1], img_rgb.shape[0]), Image.Resampling.BILINEAR))
        # Min-max norm
        mi, ma = attn_upsampled.min(), attn_upsampled.max()
        norm_map = (attn_upsampled - mi) / (ma - mi + 1e-8)
        return norm_map

    # -------------------------------------------------------------------------
    # HÀNG 1: DINOv3 VANILLA (PRETRAINED GỐC)
    # -------------------------------------------------------------------------
    mean_v = attn_vanilla.mean(axis=0)
    mean_v_up = process_attn_visuals(mean_v)
    entropy_v = compute_attention_entropy(mean_v)

    # Cột 0: Ảnh gốc
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f"[1] Ảnh Đầu Vào RGB\n{filename}", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    # Cột 1 đến num_heads: Từng Attention Head
    for h in range(num_heads):
        head_up = process_attn_visuals(attn_vanilla[h])
        axes[0, 1 + h].imshow(head_up, cmap=colormap)
        axes[0, 1 + h].set_title(f"Vanilla Head #{h}", fontsize=9)
        axes[0, 1 + h].axis("off")

    # Cột Mean Attention
    axes[0, 1 + num_heads].imshow(mean_v_up, cmap=colormap)
    axes[0, 1 + num_heads].set_title(f"Vanilla Mean Attn\nEntropy: {entropy_v:.3f}", fontsize=9, fontweight="bold", color="darkblue")
    axes[0, 1 + num_heads].axis("off")

    # Cột Overlay Blended
    heatmap_v_rgba = plt.get_cmap(colormap)(mean_v_up)[:, :, :3]
    blended_v = 0.55 * img_rgb + 0.45 * heatmap_v_rgba
    axes[0, 2 + num_heads].imshow(np.clip(blended_v, 0, 1))
    axes[0, 2 + num_heads].set_title("Vanilla Overlay", fontsize=9, fontweight="bold")
    axes[0, 2 + num_heads].axis("off")

    # Cột Thresholded Mask (Top Focus)
    thresh_val_v = np.percentile(mean_v_up, threshold_percentile * 100)
    focus_mask_v = (mean_v_up >= thresh_val_v).astype(np.float32)
    masked_img_v = img_rgb.copy()
    masked_img_v[focus_mask_v < 0.5] *= 0.25  # Làm tối nền không quan trọng
    axes[0, 3 + num_heads].imshow(np.clip(masked_img_v, 0, 1))
    axes[0, 3 + num_heads].set_title(f"Vanilla Top {int((1-threshold_percentile)*100)}% Focus", fontsize=9, fontweight="bold")
    axes[0, 3 + num_heads].axis("off")

    # Gắn nhãn hàng 1
    axes[0, 0].text(-0.15, 0.5, "DINOv3 GỐC\n(Vanilla)", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=12, fontweight="bold", color="navy")

    # -------------------------------------------------------------------------
    # HÀNG 2: STUDENT CHECKPOINT (HƯỚNG G)
    # -------------------------------------------------------------------------
    if has_student:
        mean_s = attn_student.mean(axis=0)
        mean_s_up = process_attn_visuals(mean_s)
        entropy_s = compute_attention_entropy(mean_s)

        # Cột 0: Ảnh gốc
        axes[1, 0].imshow(img_rgb)
        axes[1, 0].set_title(f"[2] Ảnh RGB\n(Student Input)", fontsize=10, fontweight="bold")
        axes[1, 0].axis("off")

        # Cột 1 đến num_heads: Từng Head của Student
        for h in range(num_heads):
            head_up = process_attn_visuals(attn_student[h])
            axes[1, 1 + h].imshow(head_up, cmap=colormap)
            axes[1, 1 + h].set_title(f"Student Head #{h}", fontsize=9)
            axes[1, 1 + h].axis("off")

        # Cột Mean Attention
        axes[1, 1 + num_heads].imshow(mean_s_up, cmap=colormap)
        axes[1, 1 + num_heads].set_title(f"Student Mean Attn\nEntropy: {entropy_s:.3f}", fontsize=9, fontweight="bold", color="darkred")
        axes[1, 1 + num_heads].axis("off")

        # Cột Overlay Blended
        heatmap_s_rgba = plt.get_cmap(colormap)(mean_s_up)[:, :, :3]
        blended_s = 0.55 * img_rgb + 0.45 * heatmap_s_rgba
        axes[1, 2 + num_heads].imshow(np.clip(blended_s, 0, 1))
        axes[1, 2 + num_heads].set_title("Student Overlay", fontsize=9, fontweight="bold")
        axes[1, 2 + num_heads].axis("off")

        # Cột Thresholded Mask (Top Focus)
        thresh_val_s = np.percentile(mean_s_up, threshold_percentile * 100)
        focus_mask_s = (mean_s_up >= thresh_val_s).astype(np.float32)
        masked_img_s = img_rgb.copy()
        masked_img_s[focus_mask_s < 0.5] *= 0.25
        axes[1, 3 + num_heads].imshow(np.clip(masked_img_s, 0, 1))
        axes[1, 3 + num_heads].set_title(f"Student Top {int((1-threshold_percentile)*100)}% Focus", fontsize=9, fontweight="bold")
        axes[1, 3 + num_heads].axis("off")

        # Gắn nhãn hàng 2
        axes[1, 0].text(-0.15, 0.5, "STUDENT\n(Hướng G)", transform=axes[1, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=12, fontweight="bold", color="darkred")

        # ---------------------------------------------------------------------
        # HÀNG 3: HIỆU SỐ ATTENTION & DỊCH CHUYỂN BIỂU DIỄN (DELTA ANALYSIS)
        # ---------------------------------------------------------------------
        diff_mean = mean_s_up - mean_v_up  # Dương: Student chú ý hơn; Âm: Vanilla chú ý hơn
        abs_diff_mean = np.abs(diff_mean)

        axes[2, 0].imshow(img_rgb)
        axes[2, 0].set_title("Mặt Nạ Đối Chiếu", fontsize=10)
        axes[2, 0].axis("off")

        # Delta từng head
        for h in range(num_heads):
            head_v_up = process_attn_visuals(attn_vanilla[h])
            head_s_up = process_attn_visuals(attn_student[h])
            head_diff = head_s_up - head_v_up
            im_d = axes[2, 1 + h].imshow(head_diff, cmap="bwr", vmin=-0.6, vmax=0.6)
            axes[2, 1 + h].set_title(f"Δ Head #{h} (S - V)", fontsize=9)
            axes[2, 1 + h].axis("off")

        # Mean Difference (Diverging Colormap: Xanh = Vanilla mạnh hơn, Đỏ = Student tập trung hơn)
        axes[2, 1 + num_heads].imshow(diff_mean, cmap="bwr", vmin=-0.6, vmax=0.6)
        axes[2, 1 + num_heads].set_title("Độ Dịch Chuyển (Δ)\n(Đỏ: +Student, Xanh: +Vanilla)", fontsize=9, fontweight="bold")
        axes[2, 1 + num_heads].axis("off")

        # Tóm tắt Entropy
        entropy_gain = ((entropy_v - entropy_s) / (entropy_v + 1e-8)) * 100.0
        entropy_status = f"Tăng tập trung +{entropy_gain:.1f}%" if entropy_gain > 0 else f"Phân tán {entropy_gain:.1f}%"
        axes[2, 2 + num_heads].text(
            0.5, 0.5,
            f"THỐNG KÊ ĐỊNH LƯỢNG:\n"
            f"────────────────────\n"
            f"• Vanilla Entropy: {entropy_v:.3f}\n"
            f"• Student Entropy: {entropy_s:.3f}\n"
            f"• Hiệu ứng: {entropy_status}\n"
            f"• Lọc ngưỡng: Top {int((1-threshold_percentile)*100)}%\n"
            f"• Delta Mean: {abs_diff_mean.mean():.4f}",
            transform=axes[2, 2 + num_heads].transAxes,
            va="center", ha="center", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5)
        )
        axes[2, 2 + num_heads].axis("off")

        # Phủ màu chênh lệch lên ảnh gốc
        pos_shift = np.clip(diff_mean, 0, 1)  # Vùng Student chú ý vượt trội
        shift_overlay = img_rgb.copy()
        shift_overlay[:, :, 0] = np.clip(shift_overlay[:, :, 0] + 0.6 * pos_shift, 0, 1)
        axes[2, 3 + num_heads].imshow(shift_overlay)
        axes[2, 3 + num_heads].set_title("Vùng Student Ưu Tiên\n(Ánh Đỏ = Tăng Focus)", fontsize=9, fontweight="bold")
        axes[2, 3 + num_heads].axis("off")

        # Gắn nhãn hàng 3
        axes[2, 0].text(-0.15, 0.5, "PHÂN TÍCH Δ\n(Chuyển Dịch)", transform=axes[2, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=12, fontweight="bold", color="darkgreen")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"🎉 [KẾT QUẢ] Đã xuất bản đồ Attention Maps thành công tại: {output_path}")


# =============================================================================
# 5. HÀM MAIN ĐIỀU PHỐI VÀ CHẠY
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bộ công cụ trích xuất và trực quan hóa Self-Attention Maps DINOv3 (Vanilla vs Student)"
    )
    parser.add_argument("--img", type=str, default=None,
                        help="Đường dẫn tới file ảnh cụ thể (ví dụ: /workspace/output/437_1791009998.jpg).")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Thư mục chứa ảnh nếu muốn trích xuất theo lô.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Đường dẫn tới checkpoint của Student (ví dụ: checkpoints/directionG/best_checkpoint.pth).")
    parser.add_argument("--weights", type=str, default=None,
                        help="Đường dẫn tới trọng số DINOv3 Vanilla offline.")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16",
                        help="Tên kiến trúc ViT: dinov3_vits16, dinov3_vitb16, dinov2_vits14...")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Thiết bị tính toán: cuda hoặc cpu.")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Ngưỡng phân vị (percentile) để lọc vùng chú ý trọng tâm (mặc định: 0.80 -> Top 20%).")
    parser.add_argument("--output", type=str, default="attention_comparison.png",
                        help="Đường dẫn file ảnh xuất ra kết quả.")
    parser.add_argument("--img_h", type=int, default=256, help="Chiều cao chuẩn hóa.")
    parser.add_argument("--img_w", type=int, default=448, help="Chiều rộng chuẩn hóa.")
    parser.add_argument("--num_samples", type=int, default=1, help="Số lượng ảnh tối đa cần xử lý.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(" 👁️ [DINOv3 ATTENTION BENCHMARK] CÔNG CỤ TRỰC QUAN HÓA SELF-ATTENTION MAPS")
    print("=" * 80)
    print(f" • Thiết bị tính toán      : {device}")
    print(f" • Kiến trúc Transformer   : {args.model_name}")
    print(f" • Độ phân giải chuẩn hóa  : {args.img_w}x{args.img_h} (Patch size: 16 -> 28x16 patches)")
    print(f" • Ngưỡng lọc trọng tâm     : Top {int((1 - args.threshold) * 100)}%")
    print(f" • File kết xuất biểu đồ   : {args.output}")

    # 1. Tìm kiếm và nạp danh sách ảnh
    candidate_files = []
    if args.img and os.path.isfile(args.img):
        candidate_files = [args.img]
    elif args.data_dir and os.path.isdir(args.data_dir):
        candidate_files = collect_image_files(args.data_dir, max_samples=args.num_samples)
    else:
        # Tự động quét các thư mục quen thuộc
        for search_dir in [
            "/workspace/output",
            "G:\\nckh\\output",
            "g:\\nckh\\output",
            "direction_data_article/figures",
            "paper_figures",
        ]:
            if os.path.isdir(search_dir):
                candidate_files = collect_image_files(search_dir, max_samples=args.num_samples)
                if candidate_files:
                    print(f"💡 Tự động tìm thấy {len(candidate_files)} ảnh từ thư mục '{search_dir}'.")
                    break

    if not candidate_files:
        print("⚠️ Không tìm thấy ảnh thực tế nào trên đĩa! Đang tạo ảnh Synthetic giả lập để kiểm thử luồng...")
        os.makedirs("sample_test_images", exist_ok=True)
        synthetic_path = "sample_test_images/synthetic_traffic_sample.png"
        dummy_arr = np.zeros((args.img_h, args.img_w, 3), dtype=np.uint8)
        dummy_arr[100:250, :] = [60, 60, 65]
        dummy_arr[140:190, 80:160] = [200, 30, 30]
        dummy_arr[160:210, 260:360] = [30, 100, 220]
        Image.fromarray(dummy_arr).save(synthetic_path)
        candidate_files = [synthetic_path]

    eval_files = candidate_files[:args.num_samples]
    print(f"✅ Sẽ xử lý {len(eval_files)} ảnh:")
    for idx, f in enumerate(eval_files):
        print(f"   [{idx+1}] {f}")

    # 2. Nạp DINOv3 Vanilla Pretrained (Mô hình gốc đối chứng)
    print("\n📦 [1/2] Đang nạp DINOv3 Gốc (Vanilla Pretrained - Baseline đối chứng)...")
    backbone_vanilla, embed_dim, patch_size = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=args.weights,
        device=device,
    )
    backbone_vanilla.eval()
    for p in backbone_vanilla.parameters():
        p.requires_grad = False
    print("   -> Nạp thành công DINOv3 Gốc!")

    # 3. Nạp Student Checkpoint (nếu có)
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
        print(f"\n🔄 [2/2] Đang nạp Student Checkpoint từ:\n   -> {student_ckpt_path}")
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
            print(f"⚠️ Không thể nạp checkpoint Student ({e}). Chuyển sang chế độ chỉ trích xuất DINO gốc.")
            backbone_student = None
            has_student = False
    else:
        print("\n💡 Không phát hiện Checkpoint Student. Chạy chế độ phân tích độc lập DINOv3 gốc.")

    # 4. Trích xuất Attention Maps
    dataset = SimpleTrafficDataset(eval_files, target_size=(args.img_h, args.img_w))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    extractor_vanilla = ViTAttentionExtractor(backbone_vanilla)
    extractor_student = ViTAttentionExtractor(backbone_student) if has_student else None

    grid_h = args.img_h // patch_size
    grid_w = args.img_w // patch_size

    print("\n⚡ Đang thực thi tính toán Self-Attention Maps tại tầng Transformer cuối...")
    t0 = time.time()

    for idx, batch in enumerate(loader):
        imgs = batch["img"].to(device)
        if imgs.ndim == 5:
            imgs = imgs.squeeze(1)
        raw_img = batch["raw_img"][0].numpy()
        fname = batch["filename"][0]

        # Trích xuất Vanilla Attention
        attn_v = extractor_vanilla.extract_cls_attention_maps(
            imgs, grid_h=grid_h, grid_w=grid_w, patch_size=patch_size
        )  # (1, num_heads, grid_h, grid_w)
        attn_v_np = attn_v[0].numpy()

        # Trích xuất Student Attention nếu có
        attn_s_np = None
        if has_student and extractor_student is not None:
            attn_s = extractor_student.extract_cls_attention_maps(
                imgs, grid_h=grid_h, grid_w=grid_w, patch_size=patch_size
            )
            attn_s_np = attn_s[0].numpy()

        out_file = args.output
        if len(eval_files) > 1:
            base, ext = os.path.splitext(args.output)
            out_file = f"{base}_{idx+1}_{fname}{ext}"

        # Xuất biểu đồ so sánh
        render_attention_comparison_figure(
            img_rgb=raw_img,
            attn_vanilla=attn_v_np,
            attn_student=attn_s_np,
            filename=fname,
            output_path=out_file,
            threshold_percentile=args.threshold,
        )

    print(f"\n⏱️ Hoàn tất toàn bộ trong {time.time() - t0:.2f}s!")


if __name__ == "__main__":
    main()
