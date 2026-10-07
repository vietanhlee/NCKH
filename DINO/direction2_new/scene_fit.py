"""
=============================================================================
 Hướng 2 Mới: Giai đoạn 1 — Khớp Nền Đa Tạp Ít Chiều (Scene Basis)
 Scene Decomposition Không Cần Ảnh Nền Median
 Nền của camera cố định được biểu diễn dưới dạng:
   B_t = clip(E_{c,0} + sum_{j=1}^J ell_{t,j} * E_{c,j})
 Tối ưu hóa bằng Robust IRLS kết hợp TAM để loại trừ phương tiện hoàn toàn.
=============================================================================
"""

import argparse
import glob
import os
import sys
import time
from typing import Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn để nạp tiện ích chung Multi-GPU
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_CURR_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

try:
    from common.gpu_utils import get_available_devices, clean_state_dict
except ImportError:
    try:
        from DINO.common.gpu_utils import get_available_devices, clean_state_dict
    except ImportError:
        def get_available_devices():
            if torch.cuda.is_available():
                return torch.device("cuda:0"), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            return torch.device("cpu"), 0, []
        def clean_state_dict(sd):
            return {k.replace("module.", ""): v for k, v in sd.items()}



class SceneBasis(nn.Module):
    """
    Tham số đa tạp nền cho tập hợp các camera cố định.
    E0: (C, 3, 256, 448) Cảnh tĩnh trung bình đầy đủ độ phân giải
    Ej: (C, J, 3, 128, 224) J ảnh cơ sở biến đổi ánh sáng/độ ướt (độ phân giải 1/2)
    ell: (C, N_frames, J) Mã ánh sáng tự do của từng frame
    """
    def __init__(
        self,
        num_cams: int = 1,
        num_frames_per_cam: int = 600,
        J: int = 4,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_cams = num_cams
        self.num_frames_per_cam = num_frames_per_cam
        self.J = J
        self.device = device

        # E0 khởi tạo bằng 0.5 (màu xám trung tính)
        self.E0 = nn.Parameter(torch.full((num_cams, 3, 256, 448), 0.5, device=device))
        # Ej khởi tạo bằng nhiễu nhỏ
        self.Ej = nn.Parameter(torch.randn(num_cams, J, 3, 128, 224, device=device) * 0.02)
        # ell khởi tạo bằng 0
        self.ell = nn.Parameter(torch.zeros(num_cams, num_frames_per_cam, J, device=device))

    def init_from_frames(self, cam_idx: int, frames: torch.Tensor, pi: Optional[torch.Tensor] = None):
        """
        Khởi tạo E0 bằng trung bình có trọng số (1 - pi) của các frame để hội tụ cực nhanh.
        frames: (N, 3, 256, 448)
        pi: (N, 16, 28) hoặc (N, 256, 448)
        """
        N, C, H, W = frames.shape
        if pi is not None:
            if pi.shape[-2:] != (H, W):
                pi_up = F.interpolate(pi.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False)
            else:
                pi_up = pi.unsqueeze(1).float()
            w_static = (1.0 - pi_up).clamp(min=1e-3)  # (N, 1, H, W)
            weighted_avg = (frames * w_static).sum(dim=0) / w_static.sum(dim=0).clamp(min=1e-4)
            self.E0.data[cam_idx] = weighted_avg.clamp(0.0, 1.0)
        else:
            self.E0.data[cam_idx] = frames.mean(dim=0).clamp(0.0, 1.0)
        self.Ej.data[cam_idx].zero_()
        self.ell.data[cam_idx].zero_()

    def get_background(self, cam_idx: int, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Tái tạo ảnh nền B_t cho danh sách các frame_indices của camera cam_idx.
        frame_indices: (B,)
        Returns:
            B: (B, 3, 256, 448)
        """
        B_count = frame_indices.shape[0]
        # Ej nội suy lên kích thước đầy đủ (J, 3, 256, 448)
        Ej_cur = self.Ej[cam_idx]  # (J, 3, 128, 224)
        Ej_up = F.interpolate(Ej_cur, size=(256, 448), mode="bilinear", align_corners=False)  # (J, 3, 256, 448)

        # ell_cur: (B, J)
        ell_cur = self.ell[cam_idx, frame_indices]

        # Tổ hợp tuyến tính: B = E0 + sum_j (ell_j * Ej_up)
        # einsum('bj, jchw -> bchw')
        delta_B = torch.einsum("bj,jchw->bchw", ell_cur, Ej_up)
        E0_cur = self.E0[cam_idx].unsqueeze(0).expand(B_count, -1, -1, -1)
        B_out = (E0_cur + delta_B).clamp(0.0, 1.0)
        return B_out


def robust_weights(
    I: torch.Tensor,
    B: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
    pool: int = 8,
    s: float = 0.02,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính trọng số inlier w_t(p) in [0, 1] bằng IRLS thích ứng theo TAM (Mục 7 Hướng 2 Mới).

    Args:
        I: Tensor ảnh gốc (N, 3, H, W)
        B: Tensor ảnh nền tái tạo (N, 3, H, W)
        pi: Tensor xác suất tiền cảnh từ TAM (N, H, W) hoặc (N, Hp, Wp)
        pool: Kích thước average pooling làm trơn phần dư không gian (mặc định 8)
        s: Độ dốc hàm sigmoid (mặc định 0.02)
        gamma: Hệ số mũ TAM (mặc định 1.0)

    Returns:
        w: (N, 1, H, W) Trọng số inlier kết hợp
        w_res: (N, 1, H, W) Trọng số phần dư thuần túy
    """
    N, C, H, W = I.shape
    # 1. Phần dư tuyệt đối trung bình kênh: r = mean_c(|I - B|) -> (N, 1, H, W)
    r = torch.abs(I - B).mean(dim=1, keepdim=True)

    # 2. AvgPool 8x8 làm trơn theo cụm ngoại lai liền khối
    if pool > 1:
        pad = pool // 2
        r_bar = F.avg_pool2d(r, kernel_size=pool, stride=1, padding=pad)
        if r_bar.shape[-2:] != (H, W):
            r_bar = r_bar[:, :, :H, :W]
    else:
        r_bar = r

    # 3. Tính tỷ lệ cắt thích ứng kappa_t
    if pi is not None:
        if pi.shape[-2:] != (H, W):
            pi_up = F.interpolate(pi.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False)
        else:
            pi_up = pi.unsqueeze(1).float()
        mean_pi = pi_up.view(N, -1).mean(dim=-1)  # (N,)
        kappa_t = torch.clamp(mean_pi + 0.05, min=0.05, max=0.60)
    else:
        pi_up = torch.zeros((N, 1, H, W), device=I.device)
        kappa_t = torch.full((N,), 0.30, device=I.device)

    # 4. Phân vị Q_t(1 - kappa_t) theo từng frame
    q_vals = []
    r_flat = r_bar.reshape(N, -1)
    for i in range(N):
        q_i = torch.quantile(r_flat[i], 1.0 - kappa_t[i].item())
        q_vals.append(q_i)
    Q = torch.stack(q_vals).view(N, 1, 1, 1)

    # 5. Trọng số phần dư w_res = sigmoid((Q - r_bar) / s)
    w_res = torch.sigmoid((Q - r_bar) / s)

    # 6. Kết hợp prior TAM: w = w_res * (1 - pi)^gamma
    w = w_res * torch.pow((1.0 - pi_up).clamp(min=0.0, max=1.0), gamma)
    return w.detach(), w_res.detach()


def fit_single_camera_scene_basis(
    frames: torch.Tensor,
    pi: Optional[torch.Tensor] = None,
    J: int = 4,
    iters: int = 2000,
    lr: float = 1e-2,
    lambda_tv: float = 0.01,
    lambda_ell: float = 1e-3,
    lambda_orth: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> Tuple[SceneBasis, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quy trình Giai đoạn 1: Khớp SceneBasis cho 1 camera từ tập frame.
    Returns:
        basis: Mô hình SceneBasis đã tối ưu
        B_all: Tensor nền của toàn bộ frame (N, 3, 256, 448)
        M_all: Tensor nhãn giả mask xe M = 1 - w (N, 1, 256, 448)
        conf_all: Độ tin cậy nhãn giả c_t
    """
    N, C, H, W = frames.shape
    basis = SceneBasis(num_cams=1, num_frames_per_cam=N, J=J, device=device)
    basis.init_from_frames(cam_idx=0, frames=frames, pi=pi)

    optimizer = torch.optim.Adam(basis.parameters(), lr=lr)
    all_indices = torch.arange(N, device=device)

    for it in range(iters):
        # Chọn ngẫu nhiên một batch frame để tối ưu
        batch_size = min(32, N)
        batch_idx = torch.randperm(N, device=device)[:batch_size]

        I_b = frames[batch_idx]
        pi_b = pi[batch_idx] if pi is not None else None

        # Dựng nền
        B_b = basis.get_background(cam_idx=0, frame_indices=batch_idx)

        # Tính trọng số robust IRLS
        w_b, w_res_b = robust_weights(I_b, B_b, pi=pi_b)

        # Weighted Charbonnier Loss: sqrt(diff^2 + eps^2)
        diff = I_b - B_b
        charb = torch.sqrt(diff * diff + 1e-6)
        l_recon = (w_b * charb).sum() / (w_b.sum().clamp(min=1.0))

        # Total Variation trên Ej
        Ej = basis.Ej[0]  # (J, 3, 128, 224)
        tv_h = torch.abs(Ej[:, :, 1:, :] - Ej[:, :, :-1, :]).mean()
        tv_w = torch.abs(Ej[:, :, :, 1:] - Ej[:, :, :, :-1]).mean()
        l_tv = lambda_tv * (tv_h + tv_w)

        # L2 norm trên ell
        ell_cur = basis.ell[0, batch_idx]
        l_ell = lambda_ell * torch.mean(ell_cur * ell_cur)

        # Trực giao giữa các ảnh cơ sở Ej
        Ej_flat = Ej.view(J, -1)  # (J, 3*128*224)
        Ej_norm = F.normalize(Ej_flat, p=2, dim=-1)
        cos_matrix = torch.matmul(Ej_norm, Ej_norm.T)  # (J, J)
        # Triệt tiêu đường chéo
        cos_matrix.fill_diagonal_(0.0)
        l_orth = lambda_orth * torch.mean(cos_matrix * cos_matrix)

        total_loss = l_recon + l_tv + l_ell + l_orth

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (it + 1) % 500 == 0 or it == 0:
            print(f"   [Giai đoạn 1 Iter {it+1}/{iters}] Loss: {total_loss.item():.4f} "
                  f"(Recon: {l_recon.item():.4f}, TV: {l_tv.item():.4f}, Orth: {l_orth.item():.4f})")

    # Dựng toàn bộ nền và xuất nhãn giả
    with torch.no_grad():
        B_all = basis.get_background(cam_idx=0, frame_indices=all_indices)
        w_all, w_res_all = robust_weights(frames, B_all, pi=pi)
        M_all = 1.0 - w_all  # Mask phương tiện
        conf_all = torch.abs(w_res_all - 0.5) * 2.0  # Độ tin cậy [0, 1]

    return basis, B_all, M_all, conf_all


def generate_synthetic_traffic_sequence(
    num_frames: int = 16,
    H: int = 256,
    W: int = 448,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tự động sinh chuỗi ảnh giao thông thời gian thực phục vụ demo thuật toán.
    Gồm:
      - Nền tĩnh B_true (mặt đường, vỉa hè, vạch kẻ đường ngắt quãng)
      - Biến đổi ánh sáng qua thời gian (mặt trời thay đổi cường độ)
      - Các phương tiện (xe ô tô/xe máy với màu sắc tương phản) di chuyển trên các làn đường.
    Returns:
        frames: (N, 3, H, W) trong [0, 1]
        gt_masks: (N, 1, H, W) ground truth mask vị trí xe
    """
    frames = []
    masks = []

    # Tạo nền cơ sở
    base_bg = torch.full((3, H, W), 0.35)  # Mặt đường nhựa xám
    # Vỉa hè trên
    base_bg[:, :int(H * 0.25), :] = 0.55
    # Dải phân cách / cây xanh
    base_bg[1, :int(H * 0.12), :] = 0.65
    base_bg[0, :int(H * 0.12), :] = 0.30
    base_bg[2, :int(H * 0.12), :] = 0.20
    # Vạch kẻ đường giữa làn
    dash_w = 24
    for w in range(0, W, dash_w * 2):
        base_bg[:, int(H * 0.58):int(H * 0.60), w:w+dash_w] = 0.95

    for t in range(num_frames):
        # Biến đổi ánh sáng theo thời gian (chu kỳ sáng / tối nhẹ)
        light_factor = 0.85 + 0.25 * np.sin(2 * np.pi * t / max(num_frames, 1))
        frame = (base_bg * light_factor).clamp(0.0, 1.0).clone()
        mask = torch.zeros((1, H, W))

        # Thêm 2-4 xe ngẫu nhiên trên các làn đường
        num_cars = np.random.randint(2, 5)
        for _ in range(num_cars):
            car_h = np.random.randint(20, 36)
            car_w = np.random.randint(40, 70)
            top = np.random.randint(int(H * 0.28), int(H * 0.85 - car_h))
            left = np.random.randint(10, W - car_w - 10)

            # Màu xe ngẫu nhiên nổi bật (đỏ, vàng, trắng, xanh)
            car_color = torch.tensor([
                np.random.uniform(0.6, 1.0),
                np.random.uniform(0.1, 0.4),
                np.random.uniform(0.1, 0.4),
            ]).view(3, 1, 1)
            frame[:, top:top+car_h, left:left+car_w] = car_color
            mask[:, top:top+car_h, left:left+car_w] = 1.0

        frames.append(frame)
        masks.append(mask)

    return torch.stack(frames), torch.stack(masks)


def visualize_and_save_results(
    frames: torch.Tensor,
    B_all: torch.Tensor,
    M_all: torch.Tensor,
    basis: SceneBasis,
    save_path: str,
    max_display: int = 4,
):
    """Vẽ và lưu kết quả trực quan hóa phân tách nền không cần ảnh nền."""
    num_samples = min(max_display, frames.shape[0])
    J = basis.J
    fig, axes = plt.subplots(num_samples, 3 + J, figsize=(3.8 * (3 + J), 2.8 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # 1. Ảnh gốc I_t
        img_np = frames[i].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Frame #{i+1} Gốc $I_t$", fontsize=10)
        axes[i, 0].axis("off")

        # 2. Nền tái tạo B_t từ Scene Basis
        bg_np = B_all[i].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
        axes[i, 1].imshow(bg_np)
        axes[i, 1].set_title(f"Nền Tái Tạo $B_t$", fontsize=10)
        axes[i, 1].axis("off")

        # 3. Mask xe tách được M_t = 1 - w_t
        mask_np = M_all[i, 0].detach().cpu().numpy().clip(0, 1)
        axes[i, 2].imshow(mask_np, cmap="hot", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Mask Xe $M_t = 1 - w_t$", fontsize=10)
        axes[i, 2].axis("off")

        # 4. Các mode cơ sở ánh sáng Ej
        for j in range(J):
            ej_cur = basis.Ej[0, j].permute(1, 2, 0).detach().cpu().numpy()
            ej_norm = (ej_cur - ej_cur.min()) / (ej_cur.max() - ej_cur.min() + 1e-6)
            axes[i, 3 + j].imshow(ej_norm)
            axes[i, 3 + j].set_title(f"Mode Cơ Sở $E_{j+1}$", fontsize=9)
            axes[i, 3 + j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"📊 [Trực Quan Hóa] Đã lưu biểu đồ phân rã nền tại: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Khớp Nền Đa Tạp Ít Chiều (Scene Basis Fitting) — Hướng 2 Mới")
    parser.add_argument("--frames_dir", "--data_dir", "--origin_dir", dest="frames_dir", type=str, default=None, help="Thư mục chứa chuỗi ảnh frame giao thông")
    parser.add_argument("--num_frames", type=int, default=16, help="Số lượng frame huấn luyện")
    parser.add_argument("--iters", type=int, default=200, help="Số vòng lặp tối ưu hóa IRLS")
    parser.add_argument("--J", type=int, default=4, help="Số chiều không gian ảnh cơ sở ánh sáng Ej")
    parser.add_argument("--lr", type=float, default=1e-2, help="Tốc độ học")
    parser.add_argument("--save_dir", "--output_dir", dest="save_dir", type=str, default="checkpoints/direction2_new_scene_fit", help="Thư mục lưu kết quả")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị tính toán")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parsed, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ [CLI Warning] Bỏ qua các đối số chưa khai báo: {unknown}")
    args = parsed

    os.makedirs(args.save_dir, exist_ok=True)
    primary_dev, num_gpus, gpu_names = get_available_devices()
    if args.device.lower() == "cpu" or num_gpus == 0:
        device = torch.device("cpu")
        print("🖥️ [Hardware] Sử dụng CPU.")
    else:
        device = primary_dev
        print(f"⚡ [Hardware] Tự động kích hoạt GPU: {gpu_names[0]} (Tổng số GPU khả dụng: {num_gpus})")

    print("\n" + "=" * 80)
    print(" 🏙️ [HƯỚNG 2 MỚI] GIAI ĐOẠN 1: KHỚP NỀN ĐA TẠP ÍT CHIỀU (SCENE BASIS FITTING)")
    print("    Phân rã cảnh giao thông hoàn toàn KHÔNG CẦN ẢNH NỀN MEDIAN")
    print("=" * 80)
    print(f" Thiết bị tính toán  : {device}")
    print(f" Số GPU khả dụng     : {num_gpus}")
    print(f" Số chiều cơ sở (J)  : {args.J}")
    print(f" Số vòng lặp (iters) : {args.iters}")
    print(f" Thư mục lưu kết quả : {args.save_dir}")
    print("-" * 80)

    # 1. Thu thập hoặc khởi tạo chuỗi frames
    frames = None
    frames_loaded_paths = []
    if args.frames_dir and os.path.isdir(args.frames_dir):
        patterns = [
            os.path.join(args.frames_dir, "**", "*.jpg"),
            os.path.join(args.frames_dir, "**", "*.jpeg"),
            os.path.join(args.frames_dir, "**", "*.png"),
            os.path.join(args.frames_dir, "*.jpg"),
            os.path.join(args.frames_dir, "*.jpeg"),
            os.path.join(args.frames_dir, "*.png"),
        ]
        img_paths = []
        for pat in patterns:
            img_paths.extend(glob.glob(pat, recursive=True))
        img_paths = sorted(list(set(img_paths)))
        if len(img_paths) >= 4:
            print(f"📂 [Dữ Liệu] Tìm thấy {len(img_paths)} ảnh trong thư mục: {args.frames_dir}")
            sel_paths = img_paths[:args.num_frames]
            loaded = []
            for p in sel_paths:
                try:
                    im = Image.open(p).convert("RGB").resize((448, 256))
                    arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
                    loaded.append(arr)
                    frames_loaded_paths.append(p)
                except Exception:
                    pass
            if len(loaded) >= 4:
                frames = torch.stack(loaded)

    if frames is None:
        print("💡 [Dữ Liệu] Chưa chỉ định thư mục ảnh hoặc thư mục trống -> Tự động sinh chuỗi giao thông mô phỏng...")
        frames, gt_masks = generate_synthetic_traffic_sequence(num_frames=args.num_frames, H=256, W=448)

    frames = frames.to(device)
    print(f"✅ [Dữ Liệu] Đã chuẩn bị {frames.shape[0]} frame độ phân giải {frames.shape[2]}x{frames.shape[3]}.")

    # 2. Thực hiện tối ưu hóa Scene Basis bằng IRLS
    start_time = time.time()
    print("\n🚀 [Tối Ưu Hóa] Bắt đầu quá trình khớp Scene Basis với Robust IRLS...")
    basis, B_all, M_all, conf_all = fit_single_camera_scene_basis(
        frames=frames,
        pi=None,
        J=args.J,
        iters=args.iters,
        lr=args.lr,
        device=device,
    )
    elapsed = time.time() - start_time
    print(f"⏱️ [Hoàn Tất] Quá trình khớp hoàn tất sau {elapsed:.2f} giây.")

    # 3. Đánh giá nhanh kết quả tách nền
    mean_fg_sparsity = (M_all > 0.5).float().mean().item()
    mean_conf = conf_all.mean().item()
    print(f"\n📈 [Chỉ Số Đánh Giá]:")
    print(f"   - Tỷ lệ diện tích phương tiện tách được (Mask Sparsity): {mean_fg_sparsity * 100:.2f}%")
    print(f"   - Độ tin cậy trung bình của nhãn giả (Mean Confidence) : {mean_conf:.4f}")

    # 4. Lưu ảnh trực quan hóa
    vis_path = os.path.join(args.save_dir, "scene_basis_visualization.png")
    visualize_and_save_results(
        frames=frames,
        B_all=B_all,
        M_all=M_all,
        basis=basis,
        save_path=vis_path,
        max_display=min(4, frames.shape[0]),
    )

    # 5. Lưu ảnh nền pseudo-backgrounds phục vụ giai đoạn huấn luyện downstream
    pseudo_bg_dir = os.path.join(args.save_dir, "pseudo_bgs")
    os.makedirs(pseudo_bg_dir, exist_ok=True)
    num_to_export = frames.shape[0]
    for idx in range(num_to_export):
        bg_np = (B_all[idx].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
        if frames_loaded_paths and idx < len(frames_loaded_paths):
            fname = os.path.basename(frames_loaded_paths[idx])
        else:
            fname = f"bg_{idx:04d}.png"
        Image.fromarray(bg_np).save(os.path.join(pseudo_bg_dir, fname))
    print(f"🖼️ [Pseudo-Backgrounds] Đã xuất {num_to_export} ảnh nền tách được tại: {pseudo_bg_dir}")

    # 6. Lưu mô hình SceneBasis (Bao gồm scene_basis.pth, best_checkpoint.pth và last_checkpoint.pth)
    ckpt_dict = {
        "state_dict": clean_state_dict(basis.state_dict()),
        "J": args.J,
        "num_frames": frames.shape[0],
        "metrics": {
            "fg_sparsity": mean_fg_sparsity,
            "mean_conf": mean_conf,
        },
        "args": vars(args),
    }
    ckpt_path = os.path.join(args.save_dir, "scene_basis.pth")
    best_ckpt_path = os.path.join(args.save_dir, "best_checkpoint.pth")
    last_ckpt_path = os.path.join(args.save_dir, "last_checkpoint.pth")
    torch.save(ckpt_dict, ckpt_path)
    torch.save(ckpt_dict, best_ckpt_path)
    torch.save(ckpt_dict, last_ckpt_path)
    print(f"💾 [Lưu Trữ] Đã lưu checkpoint SceneBasis tại   : {ckpt_path}")
    print(f"🏆 [Lưu Trữ] Đã lưu checkpoint tốt nhất tại    : {best_ckpt_path}")
    print(f"📦 [Lưu Trữ] Đã lưu checkpoint cuối cùng tại   : {last_ckpt_path}")
    print("\n🎉 [SUCCESS] Chạy thực nghiệm Giai đoạn 1 Hướng 2 Mới thành công rực rỡ!\n")


if __name__ == "__main__":
    main()
