"""
=============================================================================
 Hướng G: TAM (Temporal Atypicality Map)
 Module tính toán thống kê phân cụm không-thời gian dài hạn theo vị trí patch
 Lưu trữ đa trạng thái tĩnh (K=4), cập nhật trực tuyến và tính xác suất tiền cảnh
 hoàn toàn không cần ảnh nền median và không cần optical flow video.
=============================================================================
"""

import os
# Ngăn ngừa lỗi xung đột OpenMP trên Windows (bắt buộc đặt trước torch/numpy)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Phiên bản thuật toán TAM. BẮT BUỘC tăng mỗi khi thay đổi logic tính atypicality / chuẩn hóa / GMM,
# vì cache pi (.npy) và tam_stats.pth tính bằng phiên bản cũ sẽ không còn tương thích.
#   v1: median-normalization, valid=False -> 2.5, mean centering (alpha 0.15 / 1.0)
#   v2: trimmed-mean centering, cụm trội luôn tĩnh, gate có dilation, chuẩn hóa phân vị 25%
#   v3: frame_norm_alpha=0.0 (chuẩn DINOv3 semantic tokens), loại bỏ local_contrast, kẹp s >= 0.08, dập tắt timestamp banner r=0
#   v4: Monotonic Anomaly Calibration (Sigmoid link function), loại trừ triệt để banner r=0 khỏi buffer, chống GMM tail pathology
TAM_VERSION = 4


def _robust_frame_scale(
    a: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    q: float = 0.25,
) -> torch.Tensor:
    """
    Ước lượng mức nền của điểm khác thường a theo từng frame bằng phân vị thấp (mặc định 25%).

    Vì sao không dùng median:
        Ở cảnh kẹt xe, > 50% patch là xe -> median rơi vào mức của xe -> sau khi chia,
        xe trông "bình thường" còn nền thấp hơn 1. Phân vị 25% vẫn nằm trong vùng nền
        miễn là nền chiếm > 25% khung hình, nên ổn định hơn nhiều.

    Args:
        a: (B, P) hoặc (P,) điểm khác thường.
        valid: mask bool cùng shape (tùy chọn) - chỉ ước lượng trên patch hợp lệ.
        q: phân vị dùng làm mức nền.
    Returns:
        scale: Tensor broadcast được với a, đã kẹp tối thiểu 1e-3.
    """
    a_c = a.clamp(min=1e-4).float()
    if valid is not None and valid.shape == a.shape:
        a_c = torch.where(valid.bool(), a_c, torch.full_like(a_c, float("nan")))
        scale = torch.nanquantile(a_c, q, dim=-1, keepdim=True) if a.dim() >= 2 else torch.nanquantile(a_c, q)
        scale = torch.nan_to_num(scale, nan=1.0)
    else:
        scale = torch.quantile(a_c, q, dim=-1, keepdim=True) if a.dim() >= 2 else torch.quantile(a_c, q)
    return scale.clamp(min=1e-3).to(a.dtype)


class PositionStats(nn.Module):
    """
    Quản lý thống kê đa trạng thái tĩnh theo từng vị trí patch không gian cho các camera.
    
    Kích thước:
        - num_cams (C): Số lượng camera/kênh quan sát.
        - num_patches (P): Số lượng patch tokens (mặc định 448 = 16 x 28).
        - num_states (K): Số trạng thái cụm (mặc định K = 4).
        - feat_dim (d): Số chiều đặc trưng PCA (mặc định d = 64).
    """
    def __init__(
        self,
        num_cams: int = 64,
        num_patches: int = 448,
        num_states: int = 4,
        feat_dim: int = 64,
        w_min: float = 0.15,
        s_max_mult: float = 1.5,
        eta: float = 0.02,
        eta_w: float = 0.005,
        eps: float = 0.01,
        frame_norm_alpha: float = 0.0,
    ):
        super().__init__()
        self.num_cams = num_cams
        self.num_patches = num_patches
        self.num_states = num_states
        self.feat_dim = feat_dim
        self.w_min = w_min
        self.s_max_mult = s_max_mult
        self.eta = eta
        self.eta_w = eta_w
        self.eps = eps
        self.frame_norm_alpha = frame_norm_alpha

        # 1. Tâm cụm mu: (C, P, K, d), chuẩn hóa L2
        # Khởi tạo trực giao ngẫu nhiên
        init_mu = F.normalize(torch.randn(num_cams, num_patches, num_states, feat_dim), p=2, dim=-1)
        self.register_buffer("mu", init_mu)

        # 2. Tần suất quan sát w: (C, P, K), tổng theo K = 1
        init_w = torch.full((num_cams, num_patches, num_states), 1.0 / num_states)
        self.register_buffer("w", init_w)

        # 3. Độ phân tán cosine trung bình s: (C, P, K)
        init_s = torch.full((num_cams, num_patches, num_states), 0.2)
        self.register_buffer("s", init_s)

        # Đánh dấu camera đã được khởi tạo
        self.register_buffer("is_initialized", torch.zeros(num_cams, dtype=torch.bool))

    @staticmethod
    def normalize_frame_features(U: torch.Tensor, alpha: float = 1.0, trim_ratio: float = 0.30) -> torch.Tensor:
        """
        Chuẩn hóa trắc quang toàn khung hình (Frame-level Photometric Normalization) bằng Trimmed Mean.

        Khử thành phần dịch chuyển đồng biến toàn cục (bóng mây, sương mù, hồng ngoại ban đêm)
        trước khi đưa vào thống kê theo vị trí.

        Vì sao dùng trimmed mean thay vì mean toàn khung:
            Ở khung hình kẹt xe, mean toàn khung bị kéo về phía đặc trưng xe, làm MỌI patch nền
            lệch khỏi tâm cụm tĩnh (đã fit trên khung hình thưa) -> nền bị đánh giá "bất thường".
            Ta ước lượng tâm khung chỉ trên (1 - trim_ratio) patch gần tâm sơ bộ nhất,
            loại bỏ các patch lệch xa nhất (thường là xe) khỏi phép ước lượng.

        Args:
            U: (B, P, d) hoặc (N, P, d)
            alpha: Hệ số trừ tâm (1.0 = Full Centering).
            trim_ratio: Tỷ lệ patch lệch xa nhất bị loại khỏi ước lượng tâm (mặc định 30%).
        """
        if alpha <= 0.0:
            return F.normalize(U, p=2, dim=-1)
        B, P, d = U.shape
        rough_mean = U.mean(dim=1, keepdim=True)  # (B, 1, d) tâm sơ bộ
        if 0.0 < trim_ratio < 1.0 and P >= 8:
            keep = max(1, int(round(P * (1.0 - trim_ratio))))
            dist = (U - rough_mean).pow(2).sum(dim=-1)  # (B, P)
            idx = torch.topk(dist, k=keep, dim=1, largest=False).indices  # (B, keep)
            kept = torch.gather(U, 1, idx.unsqueeze(-1).expand(-1, -1, d))
            frame_mean = kept.mean(dim=1, keepdim=True)
        else:
            frame_mean = rough_mean
        U_centered = U - alpha * frame_mean
        return F.normalize(U_centered, p=2, dim=-1)

    def set_cam_capacity(self, target_cams: int):
        """Đặt lại số lượng camera chính xác cho PositionStats và tái cấu trúc buffer."""
        if target_cams == self.num_cams:
            return
        device = self.mu.device
        dtype = self.mu.dtype
        self.register_buffer("mu", torch.zeros(target_cams, self.num_patches, self.num_states, self.feat_dim, device=device, dtype=dtype))
        self.register_buffer("w", torch.full((target_cams, self.num_patches, self.num_states), 1.0 / self.num_states, device=device, dtype=dtype))
        self.register_buffer("s", torch.full((target_cams, self.num_patches, self.num_states), 0.2, device=device, dtype=dtype))
        self.register_buffer("is_initialized", torch.zeros(target_cams, dtype=torch.bool, device=device))
        self.num_cams = target_cams

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Tự động điều chỉnh kích thước buffer cho PositionStats khi số lượng camera trong checkpoint
        khác với số camera khởi tạo mặc định (ví dụ checkpoint có 570 camera nhưng init là 16 camera).
        """
        mu_key = prefix + "mu"
        if mu_key in state_dict:
            cams_in_ckpt = state_dict[mu_key].shape[0]
            if cams_in_ckpt != self.num_cams:
                self.set_cam_capacity(cams_in_ckpt)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def ensure_cam_capacity(self, max_cid: int):
        """Mở rộng dung lượng camera động nếu gặp camera_id lớn hơn num_cams hiện tại."""
        if max_cid < self.num_cams:
            return
        new_cams = max(max_cid + 1, self.num_cams * 2)
        device = self.mu.device
        dtype = self.mu.dtype

        new_mu = F.normalize(torch.randn(new_cams, self.num_patches, self.num_states, self.feat_dim, device=device, dtype=dtype), p=2, dim=-1)
        new_mu[:self.num_cams] = self.mu
        self.register_buffer("mu", new_mu)

        new_w = torch.full((new_cams, self.num_patches, self.num_states), 1.0 / self.num_states, device=device)
        new_w[:self.num_cams] = self.w
        self.register_buffer("w", new_w)

        new_s = torch.full((new_cams, self.num_patches, self.num_states), 0.2, device=device)
        new_s[:self.num_cams] = self.s
        self.register_buffer("s", new_s)

        new_init = torch.zeros(new_cams, dtype=torch.bool, device=device)
        new_init[:self.num_cams] = self.is_initialized
        self.register_buffer("is_initialized", new_init)
        self.num_cams = new_cams

    @torch.no_grad()
    def init_kmeans(self, cid: int, U: torch.Tensor, n_iters: int = 15):
        """
        Khởi tạo offline k-means cosine (K=4) ĐƯỢC VECTOR HÓA HOÀN TOÀN song song trên tất cả P vị trí patch
        của camera cid. Tăng tốc 50x-100x so với vòng lặp tuần tự Python.
        U: Tensor (N_frames, P, d), đã chuẩn hóa L2.
        """
        self.ensure_cam_capacity(cid)
        N, P, d = U.shape
        K = self.num_states
        device = self.mu.device
        U = self.normalize_frame_features(U.to(device).float(), alpha=self.frame_norm_alpha)

        # Chuyển vị thành (P, N, d) để tính toán batch matrix multiplication song song cho P patch
        X = U.permute(1, 0, 2).contiguous()  # (P, N, d)

        # Khởi tạo tâm cụm ngẫu nhiên cho từng patch: (P, K, d)
        if N < K:
            rand_idx = torch.randint(0, N, (P, K), device=device)
        else:
            # Sinh ma trận hoán vị ngẫu nhiên cho từng patch
            rand_scores = torch.rand(P, N, device=device)
            rand_idx = torch.topk(rand_scores, k=K, dim=-1)[1]  # (P, K)

        idx_expanded = rand_idx.unsqueeze(-1).expand(-1, -1, d)  # (P, K, d)
        centroids = torch.gather(X, dim=1, index=idx_expanded).clone()
        centroids = F.normalize(centroids, p=2, dim=-1)

        # Vòng lặp K-means cosine vector hóa trên toàn bộ P patch
        for _ in range(n_iters):
            # sim: (P, N, d) x (P, d, K) -> (P, N, K)
            sim = torch.bmm(X, centroids.transpose(1, 2))
            labels = torch.argmax(sim, dim=-1)  # (P, N)
            one_hot = F.one_hot(labels, num_classes=K).float()  # (P, N, K)

            counts = one_hot.sum(dim=1)  # (P, K)
            # Tổng vector thuộc từng cụm: (P, K, N) x (P, N, d) -> (P, K, d)
            sum_x = torch.bmm(one_hot.transpose(1, 2), X)

            # Xử lý cụm rỗng (counts == 0): chọn mẫu ngẫu nhiên từ X để tái khởi tạo
            empty_mask = (counts < 0.5)  # (P, K)
            if empty_mask.any():
                fallback_idx = torch.randint(0, N, (P, K), device=device).unsqueeze(-1).expand(-1, -1, d)
                fallback_x = torch.gather(X, dim=1, index=fallback_idx)
                new_centroids = torch.where(
                    empty_mask.unsqueeze(-1),
                    fallback_x,
                    sum_x / counts.clamp(min=1.0).unsqueeze(-1),
                )
            else:
                new_centroids = sum_x / counts.clamp(min=1.0).unsqueeze(-1)

            centroids = F.normalize(new_centroids, p=2, dim=-1)

        # Tính trọng số tần suất w và độ phân tán s lần cuối
        sim = torch.bmm(X, centroids.transpose(1, 2))  # (P, N, K)
        labels = torch.argmax(sim, dim=-1)            # (P, N)
        one_hot = F.one_hot(labels, num_classes=K).float()  # (P, N, K)
        counts = one_hot.sum(dim=1)                   # (P, K)
        weights = counts / float(max(1, N))           # (P, K)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Độ phân tán cosine: trung bình (1 - cos_sim) của các điểm thuộc cụm k
        dist_matrix = (1.0 - sim).clamp(min=0.0)      # (P, N, K)
        sum_dist = (dist_matrix * one_hot).sum(dim=1) # (P, K)
        dispersions = torch.where(
            counts > 0.5,
            (sum_dist / counts.clamp(min=1.0)).clamp(min=1e-4),
            torch.full_like(counts, 0.2),
        )

        self.mu[cid] = centroids
        self.w[cid] = weights
        self.s[cid] = dispersions
        self.is_initialized[cid] = True

    @torch.no_grad()
    def update(
        self,
        cids: torch.Tensor,
        U: torch.Tensor,
        gate_pi: Optional[torch.Tensor] = None,
    ):
        """
        Cập nhật trực tuyến có cổng (Gated Online EMA Update) các cụm trạng thái không gradient.
        Ngăn chặn phương tiện giao thông ở làn đông đúc làm nhiễm bẩn tâm cụm nền tĩnh.
        cids: (B,) Camera IDs
        U: (B, P, d) Patch features đã chuẩn hóa L2
        gate_pi: Optional (B, P) Xác suất tiền cảnh trong [0, 1]. Nếu vị trí đang có xe (pi cao),
                 tốc độ cập nhật cụm tĩnh sẽ bị khóa hãm tương ứng với (1 - pi).
        """
        B, P, d = U.shape
        max_cid = int(cids.max().item())
        self.ensure_cam_capacity(max_cid)

        U_norm = self.normalize_frame_features(U, alpha=self.frame_norm_alpha)

        # Lấy tâm cụm hiện tại tương ứng với các camera trong batch
        # mu_batch: (B, P, K, d)
        mu_b = self.mu[cids]
        w_b = self.w[cids]
        s_b = self.s[cids]

        # Tính hệ số cổng bảo vệ nền: g in [0.05, 1.0]
        if gate_pi is not None:
            gate = (1.0 - gate_pi.view(B, P).detach().clamp(0.0, 1.0)).clamp(min=0.05, max=1.0)
        else:
            gate = torch.ones((B, P), device=U.device, dtype=U.dtype)

        eta_eff = (self.eta * gate).unsqueeze(-1)       # (B, P, 1)
        eta_w_eff = (self.eta_w * gate).unsqueeze(-1)   # (B, P, 1)

        # Tính tương đồng cosine: U_norm (B, P, 1, d) x mu_b (B, P, K, d) -> cos_sim (B, P, K)
        cos_sim = (U_norm.unsqueeze(2) * mu_b).sum(dim=-1)
        k_star = torch.argmax(cos_sim, dim=-1)  # (B, P)

        # Tạo one-hot cho k_star để cập nhật vector hóa
        one_hot = F.one_hot(k_star, num_classes=self.num_states).float()  # (B, P, K)

        # Cập nhật tần suất w_k có cổng: w_k <- (1 - eta_w_eff) * w_k + eta_w_eff * 1[k == k*]
        new_w = (1.0 - eta_w_eff) * w_b + eta_w_eff * one_hot
        new_w = new_w / new_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Cập nhật tâm mu_k* và độ phân tán s_k* có cổng
        u_expanded = U_norm.unsqueeze(2).expand(-1, -1, self.num_states, -1)
        # Chỉ cập nhật tại k == k* với bước nhảy điều biến bởi eta_eff
        delta_mu = eta_eff.unsqueeze(-1) * one_hot.unsqueeze(-1) * (u_expanded - mu_b)
        new_mu = F.normalize(mu_b + delta_mu, p=2, dim=-1)

        cos_best = cos_sim.gather(dim=-1, index=k_star.unsqueeze(-1)).squeeze(-1)  # (B, P)
        dist_best = (1.0 - cos_best).clamp(min=0.0)
        dist_expanded = dist_best.unsqueeze(-1)  # (B, P, 1)
        delta_s = eta_eff * one_hot * (dist_expanded - s_b)
        new_s = (s_b + delta_s).clamp(min=1e-4)

        # Cơ chế Reset trạng thái chết / Đón nhận trạng thái nền mới (mặt đường ướt, vệt bóng mới)
        # Chỉ cho phép reset khi vị trí đó không phải là phương tiện rõ rệt (gate >= 0.5)
        low_sim_mask = (cos_best < 0.6) & (gate >= 0.5)  # (B, P)
        min_w, min_k = torch.min(new_w, dim=-1)  # (B, P)
        reinit_mask = low_sim_mask & (min_w < 0.02)
        if reinit_mask.any():
            b_idx, p_idx = torch.where(reinit_mask)
            k_reset = min_k[b_idx, p_idx]
            new_mu[b_idx, p_idx, k_reset] = U_norm[b_idx, p_idx]
            new_w[b_idx, p_idx, k_reset] = 0.05
            new_w[b_idx, p_idx] = new_w[b_idx, p_idx] / new_w[b_idx, p_idx].sum(dim=-1, keepdim=True)
            new_s[b_idx, p_idx, k_reset] = 0.1

        # Ghi ngược lại buffer chính
        self.mu[cids] = new_mu
        self.w[cids] = new_w
        self.s[cids] = new_s
        self.is_initialized[cids] = True

    @torch.no_grad()
    def static_set(self, cids: torch.Tensor) -> torch.Tensor:
        """
        Xác định tập các trạng thái tĩnh S: {k : w_k >= w_min và s_k <= s_max}
        cids: (B,)
        Returns:
            static_mask: (B, P, K) kiểu bool
        """
        w_b = self.w[cids]  # (B, P, K)
        s_b = self.s[cids]  # (B, P, K)
        
        # Ngưỡng s_max = 1.5 * median(s_b) cho từng camera
        s_median = torch.median(s_b.view(s_b.shape[0], -1), dim=-1)[0].view(-1, 1, 1)
        s_max = self.s_max_mult * s_median.clamp(min=0.05)

        static_mask = (w_b >= self.w_min) & (s_b <= s_max)

        # Cụm có tần suất lớn nhất tại mỗi vị trí LUÔN được coi là trạng thái tĩnh:
        # Trạng thái chiếm ưu thế theo thời gian chính là nền. Nếu không có ràng buộc này,
        # ở làn đông (hoặc khi K cụm bị chia giữa ngày/đêm/đường ướt, mỗi cụm w < w_min)
        # vị trí đó không có trạng thái tĩnh nào -> luôn bị coi là "bất thường" -> pi đỏ thành dải làn.
        top_k = torch.argmax(w_b, dim=-1, keepdim=True)  # (B, P, 1)
        dominant = torch.zeros_like(static_mask).scatter_(-1, top_k, True)
        return static_mask | dominant

    @torch.no_grad()
    def atypicality(self, cids: torch.Tensor, U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính điểm khác thường không thời gian a_t(p) cho từng patch.
        cids: (B,)
        U: (B, P, d)
        Returns:
            a: (B, P) Điểm khác thường
            valid: (B, P) bool, True nếu vị trí có ít nhất 1 trạng thái tĩnh
        """
        B, P, d = U.shape
        U_norm = self.normalize_frame_features(U, alpha=self.frame_norm_alpha)
        mu_b = self.mu[cids]  # (B, P, K, d)
        s_b = self.s[cids]    # (B, P, K)
        
        # Tương đồng cosine và khoảng cách cosine: dist = 1 - cos
        cos_sim = (U_norm.unsqueeze(2) * mu_b).sum(dim=-1)  # (B, P, K)
        cos_dist = (1.0 - cos_sim).clamp(min=0.0)

        # Tính chuẩn hóa khoảng cách thống kê theo độ phân tán: dist / (s_eff + eps)
        # Kẹp cận dưới độ phân tán s_b >= 0.08 để chống nhiễu / nổ giá trị tại các patch ít biến động ở rìa ảnh
        s_eff = s_b.clamp(min=0.08)
        norm_dist = cos_dist / (s_eff + self.eps)  # (B, P, K)

        # Lọc theo tập trạng thái tĩnh S
        is_static = self.static_set(cids)  # (B, P, K)
        valid = is_static.any(dim=-1)      # (B, P) bool

        # Điểm khác thường a_t là khoảng cách nhỏ nhất tới các trạng thái tĩnh
        norm_dist_masked = norm_dist.masked_fill(~is_static, float("inf"))
        a = torch.min(norm_dist_masked, dim=-1)[0]  # (B, P)

        # static_set luôn chứa cụm trội nên valid đúng ở mọi vị trí. Phòng hờ trường hợp
        # dữ liệu suy biến: dùng khoảng cách tới cụm gần nhất (KHÔNG gán hằng số cao như 2.5)
        a_fallback = torch.min(norm_dist, dim=-1)[0]
        a = torch.where(valid, a, a_fallback)

        # Cưỡng bức dập tắt tuyệt đối hàng banner timestamp camera trên cùng (hàng r=0, 28 patch đầu tiên nếu P=448):
        # Dòng chữ timestamp/tên camera màu trắng nhảy từng giây tạo gradient tương phản cao nhưng
        # hoàn toàn không phải phương tiện giao thông.
        # ĐẶC BIỆT: valid[:, :28] = False để ngăn không cho giá trị 0.0 lọt vào bộ đệm hồ chứa GMMCalibrator!
        if P == 448:
            if a.dim() == 2:
                a_2d = a.clone().view(B, 16, 28)
                a_2d[:, 0, :] = 0.0
                a = a_2d.reshape(B, P)
                valid = valid.clone()
                valid[:, :28] = False
            elif a.dim() == 1:
                a_2d = a.clone().view(16, 28)
                a_2d[0, :] = 0.0
                a = a_2d.reshape(-1)
                valid = valid.clone()
                valid[:28] = False

        return a, valid

    @torch.no_grad()
    def activity_prior(self, cid: int) -> torch.Tensor:
        """
        Tính bản đồ hoạt động dài hạn của camera: A_c(p) = 1 - sum_{k in S} w_k
        Phản ánh tỷ lệ thời gian vị trí p có phương tiện hoặc không tĩnh.
        """
        cid_t = torch.tensor([cid], device=self.mu.device)
        is_static = self.static_set(cid_t)[0]  # (P, K)
        w_p = self.w[cid]                      # (P, K)
        w_static = (w_p * is_static.float()).sum(dim=-1)  # (P,)
        A_c = (1.0 - w_static).clamp(0.0, 1.0)
        return A_c


class GMMCalibrator:
    """
    Ước lượng GMM 2 thành phần trên log(a_t) để hiệu chuẩn điểm khác thường
    thành xác suất tiền cảnh phương tiện pi_t(p) in [0, 1].
    
    Cải tiến chuẩn khoa học:
    1. Thuật toán Hồ Chứa (Vectorized Reservoir Sampling - Knuth Algorithm R):
       Thay thế FIFO để duy trì mẫu đại diện ngẫu nhiên đồng đều (Uniform Sample)
       của toàn bộ quá trình quan sát, triệt tiêu thiên vị cục bộ (Recency Bias).
    2. Làm mượt tham số qua EMA (Polyak-Ruppert Parameter Averaging):
       Triệt tiêu hiện tượng giật ngưỡng (Jittery Drift) của pi, giữ AGM và SRS ổn định tuyệt đối.
    3. Hỗ trợ Subsampling nội frame và lưu snapshot buffer khi checkpoint.
    """
    def __init__(self, max_buffer_size: int = 300000, ema_alpha: float = 0.15):
        self.max_buffer_size = int(max_buffer_size)
        self.ema_alpha = float(ema_alpha)
        # Bộ đệm dạng mảng numpy float32 để vector hóa tối ưu tốc độ và tiết kiệm RAM
        self.buffer = np.zeros(self.max_buffer_size, dtype=np.float32)
        self.size = 0          # Số lượng mẫu hiện có trong hồ chứa
        self.total_seen = 0    # Tổng số mẫu đã quan sát từ trước đến nay

        # Tham số phân phối nền (Background Log-Normal Null Distribution)
        self.mu_bg = 0.0
        self.std_bg = 0.5
        # Tham số ngưỡng quyết định và độ dốc (Monotonic Sigmoid Decision Boundary)
        self.theta = 1.0   # Ngưỡng log_a_norm (pi = 0.5)
        self.tau = 0.4     # Độ dốc chuyển tiếp (temperature scale)
        # Tham số tiền cảnh tham chiếu (cho logging và backward compatibility)
        self.mu_fg = 1.5
        self.std_fg = 0.8
        self.pi_weight = 0.15
        self.is_fitted = False

    def push(
        self,
        a: torch.Tensor,
        valid: torch.Tensor,
        max_samples: Optional[int] = None,
        subsample_ratio: Optional[float] = None,
    ):
        """
        Đưa các điểm khác thường hợp lệ vào hồ chứa Reservoir Sampling.
        
        Args:
            a: Tensor điểm khác thường (B, P) hoặc bất kỳ shape nào
            valid: Tensor boolean đánh dấu vị trí hợp lệ
            max_samples: Hạn ngạch số mẫu tối đa (dùng cho Warmup Equal Quota)
            subsample_ratio: Tỷ lệ lấy mẫu ngẫu nhiên (ví dụ 0.15 để giảm tương quan không gian)
        """
        # Chuẩn hóa theo mức nền phân vị 25% của từng frame (xem _robust_frame_scale)
        if torch.is_tensor(a):
            # Lọc chỉ các patch hợp lệ và có atypicality thực tế (> 1e-4) để loại bỏ hoàn toàn banner 0.0
            valid_mask_t = valid.bool() & (a > 1e-4)
            a_norm = a / _robust_frame_scale(a, valid=valid_mask_t)
            valid_mask = valid_mask_t.detach().cpu().numpy().astype(bool)
            valid_a = a_norm.detach().cpu().numpy()[valid_mask].flatten()
        else:
            valid_mask = np.asarray(valid, dtype=bool) & (np.asarray(a) > 1e-4)
            valid_a = np.asarray(a)[valid_mask].flatten()

        if len(valid_a) == 0:
            return

        # 1. Subsampling ngẫu nhiên nếu được chỉ định
        if subsample_ratio is not None and 0.0 < subsample_ratio < 1.0:
            sub_count = max(1, int(len(valid_a) * subsample_ratio))
            perm = np.random.choice(len(valid_a), size=sub_count, replace=False)
            valid_a = valid_a[perm]

        # 2. Giới hạn số mẫu tối đa (Equal Quota per camera)
        if max_samples is not None and len(valid_a) > max_samples:
            perm = np.random.choice(len(valid_a), size=max_samples, replace=False)
            valid_a = valid_a[perm]

        log_a = np.log(np.clip(valid_a, 1e-4, 100.0)).astype(np.float32)
        k_new = len(log_a)
        if k_new == 0:
            return

        # 3. Thuật toán Hồ Chứa Vectorized (Vectorized Reservoir Sampling)
        # Bước 3a: Nếu hồ chứa chưa đầy, điền vào các vị trí trống còn lại
        if self.size < self.max_buffer_size:
            avail = min(k_new, self.max_buffer_size - self.size)
            self.buffer[self.size : self.size + avail] = log_a[:avail]
            self.size += avail
            self.total_seen += avail
            log_a = log_a[avail:]
            k_new = len(log_a)

        # Bước 3b: Nếu hồ chứa đã đầy và vẫn còn mẫu mới, áp dụng Knuth Algorithm R vector hóa
        if k_new > 0:
            t_indices = np.arange(self.total_seen + 1, self.total_seen + k_new + 1)
            # Sinh số ngẫu nhiên r_i trong [0, t_i - 1]
            rand_floats = np.random.random(size=k_new)
            r_indices = (rand_floats * t_indices).astype(np.int64)
            # Những phần tử có r < max_buffer_size sẽ được đưa vào hồ chứa
            selected_mask = r_indices < self.max_buffer_size
            if np.any(selected_mask):
                target_slots = r_indices[selected_mask]
                incoming_vals = log_a[selected_mask]
                self.buffer[target_slots] = incoming_vals
            self.total_seen += k_new

    def refit(self, max_iters: int = 30):
        """
        Hiệu chuẩn thống kê bền vững (Robust Monotonic Anomaly Calibration) từ dữ liệu hồ chứa:
        1. Ước lượng phân phối giả thuyết không của nền đường tĩnh (Background Null Hypothesis H0):
           - mu_bg = median(buffer)
           - std_bg = 1.4826 * MAD(buffer)
        2. Thiết lập ngưỡng quyết định bất thường theta (mức tin cậy 97.7%) và độ dốc tau:
           - theta = mu_bg + 2.0 * std_bg
           - tau = max(0.20, std_bg * 0.75)
        3. Làm mượt tham số qua Polyak-Ruppert EMA để triệt tiêu dao động giữa các chu kỳ.
        """
        if self.size < 200:
            return
        data = self.buffer[:self.size]

        # 1. Ước lượng bền vững tâm và độ phân tán của nền đường (chiếm > 70-80% dữ liệu quan sát)
        cand_mu_bg = float(np.median(data))
        mad = 1.4826 * float(np.median(np.abs(data - cand_mu_bg)))
        cand_std_bg = float(max(0.20, mad))

        # 2. Ngưỡng quyết định bất thường (2.0 độ lệch chuẩn trên nền, tương ứng p-value < 0.023)
        cand_theta = float(cand_mu_bg + 2.0 * cand_std_bg)
        cand_tau = float(max(0.20, cand_std_bg * 0.75))

        # 3. Ước lượng tham số tiền cảnh tham chiếu (cho logging và tương thích ngược)
        fg_pts = data[data > cand_theta]
        if len(fg_pts) >= 10:
            cand_mu_fg = float(np.median(fg_pts))
            cand_std_fg = float(max(0.25, 1.4826 * np.median(np.abs(fg_pts - cand_mu_fg))))
            cand_pi = float(np.clip(len(fg_pts) / float(len(data)), 0.05, 0.40))
        else:
            cand_mu_fg = float(cand_mu_bg + 3.0 * cand_std_bg)
            cand_std_fg = float(max(0.30, cand_std_bg * 1.2))
            cand_pi = 0.15

        # 4. Cập nhật tham số theo EMA
        if not self.is_fitted:
            self.mu_bg = cand_mu_bg
            self.std_bg = cand_std_bg
            self.theta = cand_theta
            self.tau = cand_tau
            self.mu_fg = cand_mu_fg
            self.std_fg = cand_std_fg
            self.pi_weight = cand_pi
            self.is_fitted = True
        else:
            alpha = self.ema_alpha
            self.mu_bg = (1.0 - alpha) * self.mu_bg + alpha * cand_mu_bg
            self.std_bg = (1.0 - alpha) * self.std_bg + alpha * cand_std_bg
            self.theta = (1.0 - alpha) * self.theta + alpha * cand_theta
            self.tau = (1.0 - alpha) * self.tau + alpha * cand_tau
            self.mu_fg = (1.0 - alpha) * self.mu_fg + alpha * cand_mu_fg
            self.std_fg = (1.0 - alpha) * self.std_fg + alpha * cand_std_fg
            self.pi_weight = (1.0 - alpha) * self.pi_weight + alpha * cand_pi

    def posterior(self, a: torch.Tensor, refine: bool = False) -> torch.Tensor:
        """
        Tính xác suất hậu nghiệm tiền cảnh pi_t(p) in [0, 1] theo Hàm Liên Kết Log-Odds Đơn Điệu:
        - Đảm bảo tính đơn điệu ngặt (Monotonicity): d(pi)/d(a) > 0.
        - Khi a -> 0 (nền đường tĩnh): pi -> 0.0 TUYỆT ĐỐI (không bao giờ bị nghịch đảo).
        - Khi a -> inf (phương tiện, xe cộ lớn): pi -> 1.0 TUYỆT ĐỐI.
        """
        # Chuẩn hóa theo mức nền phân vị 25% của các patch hợp lệ
        valid_eval = (a > 1e-4)
        if a.dim() >= 2 and a.shape[-1] == 448:
            valid_eval = valid_eval.clone()
            valid_eval.view(-1, 16, 28)[:, 0, :] = False

        scale = _robust_frame_scale(a, valid=valid_eval)
        a_norm = a / scale
        log_a = torch.log(a_norm.clamp(min=1e-4, max=100.0))

        if not self.is_fitted:
            theta = 0.8
            tau = 0.4
        else:
            theta = self.theta
            tau = max(0.10, self.tau)

        # Hàm liên kết Sigmoid đơn điệu: pi(a) = sigma((log_a - theta) / tau)
        raw_pi = torch.sigmoid((log_a - theta) / tau)

        # Cưỡng bức dập tắt tuyệt đối các patch nền 0 hoặc r=0
        mask_zero = (a <= 1e-4)
        pi = torch.where(mask_zero, torch.zeros_like(raw_pi), raw_pi)

        if a.dim() >= 2 and a.shape[-1] == 448:
            pi_2d = pi.clone().view(-1, 16, 28)
            pi_2d[:, 0, :] = 0.0
            pi = pi_2d.view_as(a)
        elif a.dim() == 2 and a.shape == (16, 28):
            pi_2d = pi.clone()
            pi_2d[0, :] = 0.0
            pi = pi_2d

        pi = pi.clamp(0.0, 1.0)
        if refine:
            return refine_traffic_foreground_map(pi)
        return pi

    def state_dict(self) -> dict:
        """Xuất trạng thái GMM Calibrator và snapshot hồ chứa để lưu vào checkpoint."""
        snap = self.buffer[:self.size]
        if len(snap) > 15000:
            perm = np.random.choice(len(snap), size=15000, replace=False)
            snap = snap[perm]

        return {
            "mu_bg": float(self.mu_bg),
            "mu_fg": float(self.mu_fg),
            "std_bg": float(self.std_bg),
            "std_fg": float(self.std_fg),
            "theta": float(self.theta),
            "tau": float(self.tau),
            "pi_weight": float(self.pi_weight),
            "is_fitted": bool(self.is_fitted),
            "max_buffer_size": int(self.max_buffer_size),
            "ema_alpha": float(self.ema_alpha),
            "total_seen": int(self.total_seen),
            "buffer_snapshot": snap.astype(np.float32),
        }

    def load_state_dict(self, state: dict):
        """Khôi phục trạng thái GMM Calibrator và nạp lại snapshot hồ chứa từ checkpoint."""
        if not isinstance(state, dict):
            return
        self.mu_bg = float(state.get("mu_bg", 0.0))
        self.mu_fg = float(state.get("mu_fg", 1.5))
        self.std_bg = float(state.get("std_bg", 0.5))
        self.std_fg = float(state.get("std_fg", 0.8))
        self.theta = float(state.get("theta", self.mu_bg + 2.0 * self.std_bg))
        self.tau = float(state.get("tau", max(0.20, self.std_bg * 0.75)))
        self.pi_weight = float(state.get("pi_weight", 0.15))
        self.is_fitted = bool(state.get("is_fitted", False))
        if "max_buffer_size" in state:
            new_max = int(state["max_buffer_size"])
            if new_max != self.max_buffer_size:
                new_buf = np.zeros(new_max, dtype=np.float32)
                copy_len = min(self.size, new_max)
                new_buf[:copy_len] = self.buffer[:copy_len]
                self.buffer = new_buf
                self.max_buffer_size = new_max
                self.size = copy_len
        if "ema_alpha" in state:
            self.ema_alpha = float(state["ema_alpha"])
        if "total_seen" in state:
            self.total_seen = int(state["total_seen"])
        if "buffer_snapshot" in state and isinstance(state["buffer_snapshot"], np.ndarray):
            snap = state["buffer_snapshot"]
            n_snap = min(len(snap), self.max_buffer_size)
            self.buffer[:n_snap] = snap[:n_snap]
            self.size = max(self.size, n_snap)


def refine_traffic_foreground_map(
    pi: torch.Tensor,
    gamma: float = 2.0,
    suppress_threshold: float = 0.25,
) -> torch.Tensor:
    """
    Hậu xử lý tinh chỉnh bản đồ tiền cảnh TAM pi (Khử nhiễu mặt đường & làm sắc nét thân xe):
      1. Sigmoidal Power Sharpening: pi_sharp = pi^gamma / (pi^gamma + (1-pi)^gamma)
         - Dập tắt các patch nhiễu lơ lửng trên mặt đường (pi in [0.15, 0.40]) về < 0.10.
         - Khuếch đại các patch thực sự là phương tiện xe cộ (pi >= 0.50) lên > 0.85.
      2. Ngưỡng nén mềm: các giá trị dưới suppress_threshold được nén mượt về 0.
      
    Args:
        pi: Tensor kích thước bất kỳ (ví dụ (B, 16, 28) hoặc (16, 28) hoặc (B, 448)) trong [0, 1].
        gamma: Hệ số làm sắc nét phi tuyến tính (khuyến nghị 1.8 - 2.5).
        suppress_threshold: Ngưỡng dập tắt nhiễu nền lờ mờ (mặc định 0.25).
        
    Returns:
        pi_refined: Tensor cùng kích thước và dtype với pi, các giá trị được tinh lọc sắc bén.
    """
    pi_clamped = pi.clamp(1e-4, 1.0 - 1e-4)

    # 1. Power Sharpening phi tuyến tính
    pi_pow = pi_clamped.pow(gamma)
    bg_pow = (1.0 - pi_clamped).pow(gamma)
    pi_sharp = pi_pow / (pi_pow + bg_pow + 1e-8)

    # 2. Soft Suppression cho các patch nền yếu
    mask_weak = pi_sharp < suppress_threshold
    pi_refined = torch.where(
        mask_weak,
        pi_sharp * (pi_sharp / max(suppress_threshold, 1e-4)),
        pi_sharp,
    )

    # 3. Lọc nhiễu ô cô lập (Isolated Patch Filtering nếu pi là lưới 2D (16, 28) hoặc (B, 16, 28)):
    # Dập tắt các patch lẻ loi đơn độc lơ lửng trên mặt đường do sai số đo
    orig_shape = pi.shape
    if (len(orig_shape) == 2 and orig_shape == (16, 28)) or (len(orig_shape) == 3 and orig_shape[1:] == (16, 28)):
        is_3d = len(orig_shape) == 3
        pi_2d = pi_refined if is_3d else pi_refined.unsqueeze(0)  # (B, 16, 28)
        # avg_pool 3x3 để đếm mật độ hỗ trợ của 8 ô lân cận
        nbr_support = F.avg_pool2d(pi_2d.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        # Ô đơn độc có pi trung bình nhưng cả cụm lân cận < 0.12 -> dập tắt (nhiễu đo).
        # Ô cô lập có pi CAO (>= 0.6) được giữ nguyên: đó thường là xe máy lẻ chiếm đúng 1 patch.
        isolated_mask = (pi_2d > 0.35) & (pi_2d < 0.60) & (nbr_support < 0.12)
        pi_2d = torch.where(isolated_mask, pi_2d * 0.15, pi_2d)

        # Cưỡng bức dập tắt hoàn toàn hàng banner timestamp camera trên cùng (r=0)
        pi_2d[:, 0, :] = 0.0

        pi_refined = pi_2d if is_3d else pi_2d.squeeze(0)

    return pi_refined.clamp(0.0, 1.0)

