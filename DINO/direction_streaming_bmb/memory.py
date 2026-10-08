"""
=============================================================================
 ST-BMB: Spatio-Temporal Background Memory Bank
 Module Quản Lý Bộ Nhớ Nền Thời Gian - Không Gian Chuẩn Production
 Phục vụ phân rã cảnh và phân đoạn phương tiện luồng (Fixed CCTV Streaming)
=============================================================================
Triết lý & Nguyên tắc cốt lõi (Camera cố định, khung hình cách nhau 5 phút):
  1. Đọc theo CÙNG VỊ TRÍ KHÔNG GIAN (Point-wise), vì camera cố định.
  2. Ghi có CỔNG THEO ĐỘ TIN CẬY NỀN (Gated Write): vùng có xe (alpha cao)
     tuyệt đối không được ghi đè vào nền mặt đường.
  3. Chọn frame vào bộ nhớ theo ĐỘ TƯƠNG ĐỒNG ÁNH SÁNG thay vì FIFO máy móc.
  4. Token Detach bắt buộc: Cắt đứt gradient qua lịch sử (chống phình VRAM
     và triệt tiêu gian lận shortcut).
  5. Long-term Background Memory (EMA có cổng):
       M = (1 - beta) * M + beta * z.detach(), với beta = eta * (1 - alpha_p)
  6. Anchor Selection thông minh: Chọn frame sạch nhất (mean(alpha) nhỏ nhất).
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_position_embedding(h: int, w: int, embed_dim: int, device: torch.device) -> torch.Tensor:
    """
    Tạo 2D Sine-Cosine Positional Embedding chuẩn không gian (Vaswani et al. / ViT).
    Args:
        h, w: Kích thước lưới patch không gian (ví dụ 16x16).
        embed_dim: Số chiều vector đặc trưng (phải chia hết cho 4).
        device: Thiết bị tính toán (CPU hoặc CUDA).
    Returns:
        Tensor có shape (1, h * w, embed_dim).
    """
    assert embed_dim % 4 == 0, f"embed_dim ({embed_dim}) phải chia hết cho 4."
    dim_per_axis = embed_dim // 2

    # Lưới tọa độ chuẩn hóa [0, h-1], [0, w-1]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid_y = grid_y.flatten()  # (h * w,)
    grid_x = grid_x.flatten()  # (h * w,)

    omega = torch.arange(dim_per_axis // 2, dtype=torch.float32, device=device)
    omega = 1.0 / (10000.0 ** (2.0 * omega / dim_per_axis))

    # Mã hóa sin-cos trục X
    out_x = torch.einsum("m,d->md", grid_x, omega)
    emb_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)  # (h*w, dim_per_axis)

    # Mã hóa sin-cos trục Y
    out_y = torch.einsum("m,d->md", grid_y, omega)
    emb_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)  # (h*w, dim_per_axis)

    pos_emb = torch.cat([emb_x, emb_y], dim=-1).unsqueeze(0)  # (1, h*w, embed_dim)
    return pos_emb


class TemporalRealtimeEmbedding(nn.Module):
    """
    Nhúng thông tin thời gian thực tế:
      - log(1 + delta_phút): khoảng cách thời gian giữa frame truy vấn và frame bộ nhớ.
      - sin(hour_angle), cos(hour_angle): chu kỳ nhật triệt (ngày/đêm) theo giờ trong ngày.
    Ánh xạ qua MLP 2 tầng lên không gian embed_dim.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        # Đầu vào gồm 3 đặc trưng vật lý: [log(1 + delta_m), sin(hour_rad), cos(hour_rad)]
        hidden_dim = max(64, embed_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, delta_minutes: torch.Tensor, hour_of_day: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_minutes: Tensor shape (B, T) khoảng cách thời gian (phút).
            hour_of_day: Tensor shape (B, T) giờ trong ngày [0.0, 24.0).
        Returns:
            Tensor shape (B, T, 1, embed_dim).
        """
        B, T = delta_minutes.shape
        # f1: log(1 + delta_minutes)
        f1 = torch.log1p(delta_minutes.clamp_min(0.0))  # (B, T)
        # f2, f3: sin/cos chu kỳ 24h
        angle = (hour_of_day % 24.0) * (2.0 * math.pi / 24.0)  # (B, T)
        f2 = torch.sin(angle)
        f3 = torch.cos(angle)

        feat = torch.stack([f1, f2, f3], dim=-1)  # (B, T, 3)
        emb = self.mlp(feat)  # (B, T, embed_dim)
        return emb.unsqueeze(2)  # (B, T, 1, embed_dim)


class BackgroundMemoryBank(nn.Module):
    """
    Background Memory Bank (BMB) chuẩn CCTV tĩnh:
      1. Lưu trữ có cấu trúc tensor (B, T, L, D) kèm valid mask (B, T, L) in [0, 1].
      2. Gated Write: Chỉ các token ở vùng mặt đường sạch (valid = 1 - alpha_p cao)
         mới được ghi nhận. Token bắt buộc phải detach() để cắt graph gradient.
      3. Long-term Background Memory (EMA có cổng):
           M = (1 - beta) * M + beta * z.detach(), với beta = eta * (1 - alpha_p)
      4. Chọn Anchor thông minh: Frame có mean(alpha) nhỏ nhất làm mỏ neo chuẩn.
      5. Nhúng không gian 2D Sin-Cos và Nhúng thời gian thực (log delta_t, sin/cos giờ).
      6. Lựa chọn khung hình theo độ tương đồng ánh sáng (Photometric Similarity).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        max_recent_frames: int = 5,
        max_anchor_frames: int = 1,
        spatial_patch_size: Tuple[int, int] = (16, 16),
        use_temporal_pos_embed: bool = True,
        ema_eta: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_recent_frames = max_recent_frames
        self.max_anchor_frames = max_anchor_frames
        self.spatial_patch_size = spatial_patch_size
        self.use_temporal_pos_embed = use_temporal_pos_embed
        self.ema_eta = ema_eta

        # Nhúng thời gian thực tế
        if self.use_temporal_pos_embed:
            self.temporal_embed = TemporalRealtimeEmbedding(embed_dim)
        else:
            self.temporal_embed = None

        # Bộ nhớ tạm (Buffer runtime - không lưu trong state_dict)
        self.recent_tokens: List[torch.Tensor] = []       # Mỗi tensor: (B, L, D)
        self.recent_valid: List[torch.Tensor] = []        # Mỗi tensor: (B, L)
        self.recent_timestamps: List[torch.Tensor] = []   # Mỗi tensor: (B,)
        self.recent_mean_rgb: List[torch.Tensor] = []     # Mỗi tensor: (B, 3)

        # Mỏ neo chuẩn dài hạn Dual-Slot (Day và Night):
        # Tách biệt slot Ngày (06:00 - 18:00) và Đêm (18:00 - 06:00) tránh bẫy khóa Anchor đêm 2 AM
        self.anchor_day_tokens: Optional[torch.Tensor] = None      # (B, L, D)
        self.anchor_day_valid: Optional[torch.Tensor] = None       # (B, L)
        self.anchor_day_timestamp: Optional[torch.Tensor] = None   # (B,)
        self.anchor_day_mean_rgb: Optional[torch.Tensor] = None    # (B, 3)
        self.anchor_day_alpha_mean: Optional[torch.Tensor] = None  # (B,)

        self.anchor_night_tokens: Optional[torch.Tensor] = None      # (B, L, D)
        self.anchor_night_valid: Optional[torch.Tensor] = None       # (B, L)
        self.anchor_night_timestamp: Optional[torch.Tensor] = None   # (B,)
        self.anchor_night_mean_rgb: Optional[torch.Tensor] = None    # (B, 3)
        self.anchor_night_alpha_mean: Optional[torch.Tensor] = None  # (B,)

        # Bản đồ nền dài hạn Long-term EMA Background Memory
        self.long_term_memory: Optional[torch.Tensor] = None  # (B, L, D)
        self.long_term_valid: Optional[torch.Tensor] = None   # (B, L)

        # Cache Spatial Positional Embedding
        self._cached_spatial_pe: Optional[torch.Tensor] = None
        self._cached_hw: Optional[Tuple[int, int]] = None

    @property
    def anchor_tokens(self) -> Optional[torch.Tensor]:
        return self.anchor_day_tokens if self.anchor_day_tokens is not None else self.anchor_night_tokens

    @anchor_tokens.setter
    def anchor_tokens(self, val: Optional[torch.Tensor]):
        self.anchor_day_tokens = val

    @property
    def anchor_valid(self) -> Optional[torch.Tensor]:
        return self.anchor_day_valid if self.anchor_day_valid is not None else self.anchor_night_valid

    @anchor_valid.setter
    def anchor_valid(self, val: Optional[torch.Tensor]):
        self.anchor_day_valid = val

    @property
    def anchor_timestamp(self) -> Optional[torch.Tensor]:
        return self.anchor_day_timestamp if self.anchor_day_timestamp is not None else self.anchor_night_timestamp

    @anchor_timestamp.setter
    def anchor_timestamp(self, val: Optional[torch.Tensor]):
        self.anchor_day_timestamp = val

    @property
    def anchor_alpha_mean(self) -> Optional[torch.Tensor]:
        return self.anchor_day_alpha_mean if self.anchor_day_alpha_mean is not None else self.anchor_night_alpha_mean

    @anchor_alpha_mean.setter
    def anchor_alpha_mean(self, val: Optional[torch.Tensor]):
        self.anchor_day_alpha_mean = val

    @property
    def anchor_mean_rgb(self) -> Optional[torch.Tensor]:
        return self.anchor_day_mean_rgb if self.anchor_day_mean_rgb is not None else self.anchor_night_mean_rgb

    @anchor_mean_rgb.setter
    def anchor_mean_rgb(self, val: Optional[torch.Tensor]):
        self.anchor_day_mean_rgb = val

    def reset(self):
        """Xóa sạch bộ nhớ khi chuyển đổi video sequence hoặc camera mới."""
        self.recent_tokens.clear()
        self.recent_valid.clear()
        self.recent_timestamps.clear()
        self.recent_mean_rgb.clear()
        self.anchor_day_tokens = None
        self.anchor_day_valid = None
        self.anchor_day_timestamp = None
        self.anchor_day_mean_rgb = None
        self.anchor_day_alpha_mean = None
        self.anchor_night_tokens = None
        self.anchor_night_valid = None
        self.anchor_night_timestamp = None
        self.anchor_night_mean_rgb = None
        self.anchor_night_alpha_mean = None
        self.long_term_memory = None
        self.long_term_valid = None

    @property
    def is_empty(self) -> bool:
        """Kiểm tra xem bộ nhớ hiện có khung hình nào không."""
        return (
            len(self.recent_tokens) == 0
            and self.anchor_tokens is None
            and self.long_term_memory is None
        )

    @property
    def num_stored_frames(self) -> int:
        """Tổng số khung hình hiện đang lưu trữ."""
        count = len(self.recent_tokens)
        if self.anchor_tokens is not None:
            count += 1
        if self.long_term_memory is not None:
            count += 1
        return count

    def get_spatial_pos_embed(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Tạo hoặc lấy Positional Embedding 2D sin-cos được cache."""
        if (
            self._cached_spatial_pe is not None
            and self._cached_hw == (h, w)
            and self._cached_spatial_pe.device == device
        ):
            return self._cached_spatial_pe

        pe = build_2d_sincos_position_embedding(h, w, self.embed_dim, device)
        self._cached_spatial_pe = pe
        self._cached_hw = (h, w)
        return pe

    def write(
        self,
        bg_tokens: torch.Tensor,
        alpha_p: Optional[torch.Tensor] = None,
        sigma_p: Optional[torch.Tensor] = None,
        timestamp: Optional[Union[float, torch.Tensor]] = None,
        mean_rgb: Optional[torch.Tensor] = None,
        is_anchor: bool = False,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ):
        """
        Ghi có cổng (Gated Write) đặc trưng nền vào Memory Bank.
        Nguyên tắc bất khả xâm phạm:
          1. Bắt buộc detach() bg_tokens để cắt đồ thị gradient lịch sử.
          2. Vùng có xe (alpha_p cao) hoặc bất định quang học cao (sigma_p cao) không được ghi đè nền.
          3. Cập nhật Long-term Memory M theo EMA có cổng với cơ chế Fast-Replace khi nền sạch xuất hiện.
          4. Cập nhật Dual-Slot Anchor (Day và Night riêng biệt) tránh bẫy khóa Anchor đêm 2 AM.
        Args:
            bg_tokens: Tensor đặc trưng nền shape (B, L, D).
            alpha_p: Mặt nạ xe cấp độ patch shape (B, L) hoặc (B, 1, Hp, Wp) in [0, 1].
            sigma_p: Mặt nạ độ bất định quang học patch shape (B, L) hoặc (B, 1, Hp, Wp).
            timestamp: Thời điểm chụp khung hình (unix seconds hoặc số phút tương đối).
            mean_rgb: Màu trung bình của frame (B, 3) phục vụ căn chỉnh quang học.
            is_anchor: Cưỡng chế chỉ định frame làm anchor cho các mẫu.
            spatial_hw: Kích thước lưới patch (Hp, Wp) nếu có.
        """
        # 1. Bắt buộc detach() triệt để
        z = bg_tokens.detach()
        B, L, D = z.shape

        # 2. Xây dựng valid mask (B, L) kết hợp cả alpha_p và sigma_p
        pool_hw = spatial_hw
        if pool_hw is None and self._cached_hw is not None and self._cached_hw[0] * self._cached_hw[1] == L:
            pool_hw = self._cached_hw
        if pool_hw is None:
            h_cand = int(math.isqrt(L))
            while h_cand > 1 and L % h_cand != 0:
                h_cand -= 1
            pool_hw = (h_cand, L // h_cand)

        if alpha_p is not None:
            if alpha_p.dim() == 4:
                alpha_p = F.adaptive_avg_pool2d(alpha_p, pool_hw).flatten(1)
            elif alpha_p.dim() == 3:
                alpha_p = alpha_p.squeeze(1) if alpha_p.shape[1] == 1 else alpha_p.flatten(1)
            valid_clean = (1.0 - alpha_p.detach()).clamp(0.0, 1.0)
            if valid_clean.shape[-1] != L:
                valid_clean = valid_clean.view(B, -1)[:, :L]
        else:
            valid_clean = torch.ones((B, L), device=z.device, dtype=z.dtype)

        # Lọc nhiễu độ bất định quang học sigma_p:
        # Vệt chói đèn pha hoặc vũng nước mưa phản chiếu dù alpha ~ 0 nhưng sigma cao,
        # sẽ bị hạ độ tin cậy để không bị ghi đè làm nền chuẩn!
        if sigma_p is not None:
            if sigma_p.dim() == 4:
                sigma_p = F.adaptive_avg_pool2d(sigma_p, pool_hw).flatten(1)
            elif sigma_p.dim() == 3:
                sigma_p = sigma_p.squeeze(1) if sigma_p.shape[1] == 1 else sigma_p.flatten(1)
            sigma_det = sigma_p.detach().clamp_min(0.0)
            if sigma_det.shape[-1] != L:
                sigma_det = sigma_det.view(B, -1)[:, :L]
            valid_optical = torch.exp(-sigma_det / 0.25)
            valid = (valid_clean * valid_optical).clamp(0.0, 1.0)
        else:
            valid = valid_clean

        # 3. Chuẩn hóa timestamp (B,)
        if timestamp is None:
            step_idx = len(self.recent_tokens)
            ts_tensor = torch.full((B,), float(step_idx * 300.0), device=z.device, dtype=torch.float32)
        elif isinstance(timestamp, (int, float)):
            ts_tensor = torch.full((B,), float(timestamp), device=z.device, dtype=torch.float32)
        elif torch.is_tensor(timestamp):
            ts_tensor = timestamp.to(z.device).float().flatten()
            if ts_tensor.numel() != B:
                ts_tensor = ts_tensor[0].expand(B)
        else:
            ts_tensor = torch.zeros((B,), device=z.device, dtype=torch.float32)

        # 4. Chuẩn hóa mean_rgb (B, 3)
        if mean_rgb is None:
            mean_rgb_tensor = torch.full((B, 3), 0.5, device=z.device, dtype=torch.float32)
        else:
            mean_rgb_tensor = mean_rgb.detach().to(z.device)
            if mean_rgb_tensor.dim() == 1 and mean_rgb_tensor.shape[0] == 3:
                mean_rgb_tensor = mean_rgb_tensor.unsqueeze(0).expand(B, 3)

        # 5. Cập nhật Long-term Background Memory M (EMA có cổng) và độ tin cậy nền
        beta = (self.ema_eta * valid).unsqueeze(-1)
        if self.long_term_memory is None:
            self.long_term_memory = z.clone()
            self.long_term_valid = valid.clone()
        else:
            # Tăng tốc tẩy xe: nếu vùng trước đó bị kẹt xe (long_term_valid < 0.5), lập tức nhận nền mới khi valid >= 0.5
            fast_replace = (self.long_term_valid < 0.5).unsqueeze(-1) & (valid.unsqueeze(-1) >= 0.5)
            effective_beta = torch.where(fast_replace, torch.ones_like(beta), beta)
            self.long_term_memory = (1.0 - effective_beta) * self.long_term_memory + effective_beta * z
            self.long_term_valid = torch.maximum(self.long_term_valid, valid)

        # 6. Anchor Selection theo Dual-Slot (Day & Night) loại bỏ bẫy 2 AM:
        # Giờ địa phương Việt Nam (GMT+7: UTC + 7*3600)
        hour_vn = ((ts_tensor + 7.0 * 3600.0) % 86400.0) / 3600.0  # (B,)
        is_day_sample = (hour_vn >= 6.0) & (hour_vn < 18.0)        # (B,)
        alpha_per_sample = (1.0 - valid).mean(dim=-1)               # (B,)

        # A. Cập nhật Slot Ban Ngày (Day Anchor)
        if is_day_sample.any():
            if self.anchor_day_tokens is None or self.anchor_day_alpha_mean is None or self.anchor_day_alpha_mean.shape[0] != B:
                self.anchor_day_tokens = z.clone()
                self.anchor_day_valid = valid.clone()
                self.anchor_day_timestamp = ts_tensor.clone()
                self.anchor_day_mean_rgb = mean_rgb_tensor.clone()
                self.anchor_day_alpha_mean = alpha_per_sample.clone()
            else:
                up_day = is_day_sample & (is_anchor | (alpha_per_sample < self.anchor_day_alpha_mean))
                if up_day.any():
                    u_3d = up_day.view(B, 1, 1)
                    u_2d = up_day.view(B, 1)
                    self.anchor_day_tokens = torch.where(u_3d, z, self.anchor_day_tokens)
                    self.anchor_day_valid = torch.where(u_2d, valid, self.anchor_day_valid)
                    self.anchor_day_timestamp = torch.where(up_day, ts_tensor, self.anchor_day_timestamp)
                    self.anchor_day_mean_rgb = torch.where(u_2d, mean_rgb_tensor, self.anchor_day_mean_rgb)
                    self.anchor_day_alpha_mean = torch.where(up_day, alpha_per_sample, self.anchor_day_alpha_mean)

        # B. Cập nhật Slot Ban Đêm (Night Anchor)
        is_night_sample = ~is_day_sample
        if is_night_sample.any():
            if self.anchor_night_tokens is None or self.anchor_night_alpha_mean is None or self.anchor_night_alpha_mean.shape[0] != B:
                self.anchor_night_tokens = z.clone()
                self.anchor_night_valid = valid.clone()
                self.anchor_night_timestamp = ts_tensor.clone()
                self.anchor_night_mean_rgb = mean_rgb_tensor.clone()
                self.anchor_night_alpha_mean = alpha_per_sample.clone()
            else:
                up_night = is_night_sample & (is_anchor | (alpha_per_sample < self.anchor_night_alpha_mean))
                if up_night.any():
                    u_3d = up_night.view(B, 1, 1)
                    u_2d = up_night.view(B, 1)
                    self.anchor_night_tokens = torch.where(u_3d, z, self.anchor_night_tokens)
                    self.anchor_night_valid = torch.where(u_2d, valid, self.anchor_night_valid)
                    self.anchor_night_timestamp = torch.where(up_night, ts_tensor, self.anchor_night_timestamp)
                    self.anchor_night_mean_rgb = torch.where(u_2d, mean_rgb_tensor, self.anchor_night_mean_rgb)
                    self.anchor_night_alpha_mean = torch.where(up_night, alpha_per_sample, self.anchor_night_alpha_mean)

        # 7. Ghi vào hàng đợi Recent Frames (chọn lọc theo ánh sáng nếu vượt giới hạn)
        if len(self.recent_tokens) >= self.max_recent_frames:
            evict_idx = 0
            if len(self.recent_mean_rgb) > 0 and mean_rgb is not None:
                dists = [
                    float(torch.norm(m - mean_rgb_tensor, dim=-1).mean().item())
                    for m in self.recent_mean_rgb
                ]
                evict_idx = int(torch.tensor(dists).argmax().item())

            self.recent_tokens.pop(evict_idx)
            self.recent_valid.pop(evict_idx)
            self.recent_timestamps.pop(evict_idx)
            self.recent_mean_rgb.pop(evict_idx)

        self.recent_tokens.append(z)
        self.recent_valid.append(valid)
        self.recent_timestamps.append(ts_tensor)
        self.recent_mean_rgb.append(mean_rgb_tensor)

    def read(
        self,
        current_device: torch.device,
        query_timestamp: Optional[Union[float, torch.Tensor]] = None,
        query_mean_rgb: Optional[torch.Tensor] = None,
        memory_dropout_p: float = 0.0,
        spatial_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        Đọc toàn bộ token từ Memory Bank kết hợp nhúng thời gian thực và không gian.
        Returns:
            mem_tokens: (B, T, L, D) đã cộng Spatial PE và Real-time Temporal PE.
            raw_mem_tokens: (B, T, L, D) NGUYÊN BẢN CHƯA CỘNG TEMPORAL PE
                            (phục vụ tính Cosine Similarity cho Familiarity Map).
            mem_valid: (B, T, L) mask độ tin cậy nền in [0, 1].
            is_empty: Biến cờ boolean báo rỗng.
        """
        if self.is_empty:
            return None, None, None, True

        # Sắp xếp các recent frames theo độ tương đồng ánh sáng với query_mean_rgb (nếu có)
        recent_indices = list(range(len(self.recent_tokens)))
        if (
            query_mean_rgb is not None
            and len(self.recent_tokens) > 1
            and len(self.recent_mean_rgb) == len(self.recent_tokens)
        ):
            q_rgb = query_mean_rgb.detach().to(current_device)
            dists = [
                float(torch.norm(m.to(current_device) - q_rgb, dim=-1).mean().item())
                for m in self.recent_mean_rgb
            ]
            recent_indices = sorted(range(len(dists)), key=lambda idx: dists[idx])

        raw_list: List[torch.Tensor] = []
        valid_list: List[torch.Tensor] = []
        ts_list: List[torch.Tensor] = []

        # 1. Đưa Long-term Background Memory M vào danh sách (độ tin cậy nền tích lũy)
        if self.long_term_memory is not None:
            raw_list.append(self.long_term_memory.to(current_device))
            B_m, L_m, _ = self.long_term_memory.shape
            lt_valid = (
                self.long_term_valid.to(current_device)
                if self.long_term_valid is not None
                else torch.ones((B_m, L_m), device=current_device)
            )
            valid_list.append(lt_valid)
            if query_timestamp is not None and torch.is_tensor(query_timestamp):
                ts_list.append(query_timestamp.to(current_device).float().flatten())
            else:
                ts_list.append(torch.zeros((B_m,), device=current_device))

        # 2. Đưa Anchor sạch nhất vào danh sách (ưu tiên Anchor cùng pha ngày/đêm để tránh bẫy 2 AM)
        if query_timestamp is not None and torch.is_tensor(query_timestamp):
            q_hour = ((query_timestamp.to(current_device).float() + 7.0 * 3600.0) % 86400.0) / 3600.0
            query_is_day = bool((q_hour.mean() >= 6.0) and (q_hour.mean() < 18.0))
        elif isinstance(query_timestamp, (int, float)):
            q_hour = ((float(query_timestamp) + 7.0 * 3600.0) % 86400.0) / 3600.0
            query_is_day = (q_hour >= 6.0 and q_hour < 18.0)
        elif query_mean_rgb is not None:
            query_is_day = bool(query_mean_rgb.mean().item() > 0.20)
        else:
            query_is_day = True

        primary_anchor = self.anchor_day_tokens if query_is_day else self.anchor_night_tokens
        primary_valid = self.anchor_day_valid if query_is_day else self.anchor_night_valid
        primary_ts = self.anchor_day_timestamp if query_is_day else self.anchor_night_timestamp

        fallback_anchor = self.anchor_night_tokens if query_is_day else self.anchor_day_tokens
        fallback_valid = self.anchor_night_valid if query_is_day else self.anchor_day_valid
        fallback_ts = self.anchor_night_timestamp if query_is_day else self.anchor_day_timestamp

        if primary_anchor is not None:
            raw_list.append(primary_anchor.to(current_device))
            valid_list.append(primary_valid.to(current_device))
            ts_list.append(primary_ts.to(current_device))
        elif fallback_anchor is not None:
            raw_list.append(fallback_anchor.to(current_device))
            valid_list.append(fallback_valid.to(current_device))
            ts_list.append(fallback_ts.to(current_device))

        # 3. Đưa Recent Frames vào danh sách (ưu tiên thứ tự tương đồng ánh sáng)
        for r_idx in recent_indices:
            raw_list.append(self.recent_tokens[r_idx].to(current_device))
            valid_list.append(self.recent_valid[r_idx].to(current_device))
            ts_list.append(self.recent_timestamps[r_idx].to(current_device))

        if len(raw_list) == 0:
            return None, None, None, True

        raw_mem = torch.stack(raw_list, dim=1)        # (B, T, L, D)
        mem_valid = torch.stack(valid_list, dim=1)    # (B, T, L)
        mem_ts = torch.stack(ts_list, dim=1)          # (B, T)

        B, T, L, D = raw_mem.shape

        # 4. Memory Dropout trong quá trình huấn luyện: K in [0, K_max]
        if self.training and memory_dropout_p > 0.0:
            rand_val = float(torch.rand(1).item())
            if rand_val < memory_dropout_p * 0.25:
                # Mô phỏng Cold Start (K=0) kiểm thử cơ chế fallback
                return None, None, None, True
            elif T > 1 and rand_val < memory_dropout_p:
                keep_count = int(torch.randint(1, T + 1, (1,)).item())
                keep_indices = torch.randperm(T, device=current_device)[:keep_count]
                keep_indices = torch.sort(keep_indices)[0]
                raw_mem = raw_mem[:, keep_indices]
                mem_valid = mem_valid[:, keep_indices]
                mem_ts = mem_ts[:, keep_indices]
                T = raw_mem.shape[1]

        # 5. Cộng Spatial Positional Embedding an toàn cho mọi hình dạng không gian
        if spatial_hw is not None:
            Hp, Wp = spatial_hw
        elif self._cached_hw is not None and self._cached_hw[0] * self._cached_hw[1] == L:
            Hp, Wp = self._cached_hw
        else:
            Hp = int(math.isqrt(L))
            while Hp > 1 and L % Hp != 0:
                Hp -= 1
            Wp = L // Hp

        spatial_pe = self.get_spatial_pos_embed(Hp, Wp, current_device).unsqueeze(1)  # (1, 1, L, D)
        mem_tokens = raw_mem + spatial_pe

        # 6. Cộng Real-time Temporal Positional Embedding
        if self.use_temporal_pos_embed and self.temporal_embed is not None:
            if query_timestamp is None:
                q_ts = mem_ts[:, -1:]
            elif isinstance(query_timestamp, (int, float)):
                q_ts = torch.full((B, 1), float(query_timestamp), device=current_device)
            elif torch.is_tensor(query_timestamp):
                q_ts = query_timestamp.to(current_device).float().view(B, 1)
            else:
                q_ts = torch.zeros((B, 1), device=current_device)

            delta_sec = torch.abs(q_ts - mem_ts)
            delta_min = delta_sec / 60.0  # (B, T)
            # Chu kỳ nhật triệt (ngày/đêm) theo múi giờ địa phương Việt Nam (GMT+7: UTC + 7*3600s)
            hour_of_day = ((mem_ts + 7.0 * 3600.0) % 86400.0) / 3600.0  # (B, T)

            temporal_pe = self.temporal_embed(delta_min, hour_of_day)  # (B, T, 1, D)
            mem_tokens = mem_tokens + temporal_pe

        return mem_tokens, raw_mem, mem_valid, False
