"""
=============================================================================
 Hướng 2 Mới: Giai đoạn 1 — Khớp Nền Đa Tạp Ít Chiều (Scene Basis)
 Scene Decomposition Không Cần Ảnh Nền Median
 Nền của camera cố định được biểu diễn dưới dạng:
   B_t = clip(E_{c,0} + sum_{j=1}^J ell_{t,j} * E_{c,j})
 Tối ưu hóa bằng Robust IRLS kết hợp TAM để loại trừ phương tiện hoàn toàn.
=============================================================================
"""

import os
import sys
from typing import Optional, Tuple
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
