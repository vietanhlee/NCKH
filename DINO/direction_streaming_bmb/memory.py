"""
=============================================================================
 ST-BMB: Spatio-Temporal Background Memory Bank
 Module Quản Lý Bộ Nhớ Nền Thời Gian - Không Gian Chuẩn Production
 Phục vụ phân rã cảnh và phân đoạn phương tiện luồng (Fixed CCTV Streaming)
 Hỗ trợ 100% DataParallel & Multi-GPU thông qua Cơ chế Cách Ly Trạng Thái (Per-Device State)
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
  7. Thread-Safe & Multi-GPU Isolation: Mỗi GPU/Device có không gian buffer
     độc lập hoàn toàn, triệt tiêu race condition và pop index out of range.
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import threading
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


class _MemoryBankDeviceState:
    """
    Trạng thái lưu trữ bộ nhớ của Background Memory Bank cho từng thiết bị phần cứng (GPU/CPU).
    Đảm bảo cách ly 100% không gian bộ nhớ giữa các GPU khi chạy nn.DataParallel đa luồng,
    triệt tiêu hoàn toàn race condition, shared list mutation và device mismatch.
    """

    def __init__(self):
        # 1. Hàng đợi Recent Frames
        self.recent_tokens: List[torch.Tensor] = []       # Mỗi tensor: (B, L, D)
        self.recent_valid: List[torch.Tensor] = []        # Mỗi tensor: (B, L)
        self.recent_timestamps: List[torch.Tensor] = []   # Mỗi tensor: (B,)
        self.recent_mean_rgb: List[torch.Tensor] = []     # Mỗi tensor: (B, 3)

        # 2. Slot Mỏ neo Ngày (06:00 - 18:00)
        self.anchor_day_tokens: Optional[torch.Tensor] = None      # (B, L, D)
        self.anchor_day_valid: Optional[torch.Tensor] = None       # (B, L)
        self.anchor_day_timestamp: Optional[torch.Tensor] = None   # (B,)
        self.anchor_day_mean_rgb: Optional[torch.Tensor] = None    # (B, 3)
        self.anchor_day_alpha_mean: Optional[torch.Tensor] = None  # (B,)

        # 3. Slot Mỏ neo Đêm (18:00 - 06:00)
        self.anchor_night_tokens: Optional[torch.Tensor] = None      # (B, L, D)
        self.anchor_night_valid: Optional[torch.Tensor] = None       # (B, L)
        self.anchor_night_timestamp: Optional[torch.Tensor] = None   # (B,)
        self.anchor_night_mean_rgb: Optional[torch.Tensor] = None    # (B, 3)
        self.anchor_night_alpha_mean: Optional[torch.Tensor] = None  # (B,)

        # 4. Bản đồ nền dài hạn EMA
        self.long_term_memory: Optional[torch.Tensor] = None  # (B, L, D)
        self.long_term_valid: Optional[torch.Tensor] = None   # (B, L)

    def reset(self):
        """Xóa sạch toàn bộ buffer của thiết bị này."""
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
    def anchor_mean_rgb(self) -> Optional[torch.Tensor]:
        return self.anchor_day_mean_rgb if self.anchor_day_mean_rgb is not None else self.anchor_night_mean_rgb

    @anchor_mean_rgb.setter
    def anchor_mean_rgb(self, val: Optional[torch.Tensor]):
        self.anchor_day_mean_rgb = val

    @property
    def anchor_alpha_mean(self) -> Optional[torch.Tensor]:
        return self.anchor_day_alpha_mean if self.anchor_day_alpha_mean is not None else self.anchor_night_alpha_mean

    @anchor_alpha_mean.setter
    def anchor_alpha_mean(self, val: Optional[torch.Tensor]):
        self.anchor_day_alpha_mean = val

    @property
    def is_empty(self) -> bool:
        return (
            len(self.recent_tokens) == 0
            and self.anchor_tokens is None
            and self.long_term_memory is None
        )

    @property
    def num_stored_frames(self) -> int:
        count = len(self.recent_tokens)
        if self.anchor_tokens is not None:
            count += 1
        if self.long_term_memory is not None:
            count += 1
        return count


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
      7. Multi-GPU DataParallel & Thread Safe: Tự động cách ly trạng thái theo từng
         thiết bị GPU, đảm bảo không race condition và không xung đột bộ đệm.
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

        # Quản lý trạng thái bộ nhớ theo thiết bị phần cứng (Thread-Safe & DataParallel-Safe)
        self._states: Dict[str, _MemoryBankDeviceState] = {}
        self._state_lock = threading.Lock()

        # Cache Spatial Positional Embedding theo (h, w, device)
        self._cached_spatial_pe: Dict[str, torch.Tensor] = {}

    def __copy__(self):
        """Hỗ trợ sao chép an toàn khi PyTorch nn.DataParallel tạo replica qua các GPU."""
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._states = {}
        result._state_lock = threading.Lock()
        result._cached_spatial_pe = {}
        return result

    def _device_to_key(self, device: Optional[Union[torch.device, str, int]]) -> str:
        if device is None:
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.current_device()}"
            return "cpu"
        if isinstance(device, int):
            return f"cuda:{device}"
        dev = torch.device(device)
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else 0
            return f"cuda:{idx}"
        return dev.type

    def _get_state(self, device: Optional[Union[torch.device, str, int]] = None) -> _MemoryBankDeviceState:
        key = self._device_to_key(device)
        if key not in self._states:
            with self._state_lock:
                if key not in self._states:
                    self._states[key] = _MemoryBankDeviceState()
        return self._states[key]

    @property
    def _active_state(self) -> _MemoryBankDeviceState:
        if len(self._states) == 0:
            return self._get_state(None)
        if torch.cuda.is_available():
            cur_key = f"cuda:{torch.cuda.current_device()}"
            if cur_key in self._states:
                return self._states[cur_key]
        return next(iter(self._states.values()))

    # --- Thuộc tính tương thích ngược cho unit test và kiểm thử ngoài ---
    @property
    def recent_tokens(self) -> List[torch.Tensor]:
        return self._active_state.recent_tokens

    @property
    def recent_valid(self) -> List[torch.Tensor]:
        return self._active_state.recent_valid

    @property
    def recent_timestamps(self) -> List[torch.Tensor]:
        return self._active_state.recent_timestamps

    @property
    def recent_mean_rgb(self) -> List[torch.Tensor]:
        return self._active_state.recent_mean_rgb

    @property
    def anchor_tokens(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_tokens

    @anchor_tokens.setter
    def anchor_tokens(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_tokens = val

    @property
    def anchor_valid(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_valid

    @anchor_valid.setter
    def anchor_valid(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_valid = val

    @property
    def anchor_timestamp(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_timestamp

    @anchor_timestamp.setter
    def anchor_timestamp(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_timestamp = val

    @property
    def anchor_alpha_mean(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_alpha_mean

    @anchor_alpha_mean.setter
    def anchor_alpha_mean(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_alpha_mean = val

    @property
    def anchor_mean_rgb(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_mean_rgb

    @anchor_mean_rgb.setter
    def anchor_mean_rgb(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_mean_rgb = val

    @property
    def anchor_day_tokens(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_day_tokens

    @anchor_day_tokens.setter
    def anchor_day_tokens(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_day_tokens = val

    @property
    def anchor_day_valid(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_day_valid

    @anchor_day_valid.setter
    def anchor_day_valid(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_day_valid = val

    @property
    def anchor_day_timestamp(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_day_timestamp

    @anchor_day_timestamp.setter
    def anchor_day_timestamp(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_day_timestamp = val

    @property
    def anchor_day_mean_rgb(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_day_mean_rgb

    @anchor_day_mean_rgb.setter
    def anchor_day_mean_rgb(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_day_mean_rgb = val

    @property
    def anchor_day_alpha_mean(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_day_alpha_mean

    @anchor_day_alpha_mean.setter
    def anchor_day_alpha_mean(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_day_alpha_mean = val

    @property
    def anchor_night_tokens(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_night_tokens

    @anchor_night_tokens.setter
    def anchor_night_tokens(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_night_tokens = val

    @property
    def anchor_night_valid(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_night_valid

    @anchor_night_valid.setter
    def anchor_night_valid(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_night_valid = val

    @property
    def anchor_night_timestamp(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_night_timestamp

    @anchor_night_timestamp.setter
    def anchor_night_timestamp(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_night_timestamp = val

    @property
    def anchor_night_mean_rgb(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_night_mean_rgb

    @anchor_night_mean_rgb.setter
    def anchor_night_mean_rgb(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_night_mean_rgb = val

    @property
    def anchor_night_alpha_mean(self) -> Optional[torch.Tensor]:
        return self._active_state.anchor_night_alpha_mean

    @anchor_night_alpha_mean.setter
    def anchor_night_alpha_mean(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).anchor_night_alpha_mean = val

    @property
    def long_term_memory(self) -> Optional[torch.Tensor]:
        return self._active_state.long_term_memory

    @long_term_memory.setter
    def long_term_memory(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).long_term_memory = val

    @property
    def long_term_valid(self) -> Optional[torch.Tensor]:
        return self._active_state.long_term_valid

    @long_term_valid.setter
    def long_term_valid(self, val: Optional[torch.Tensor]):
        dev = val.device if val is not None else None
        self._get_state(dev).long_term_valid = val

    def reset(self, device: Optional[Union[torch.device, str, int]] = None):
        """Xóa sạch bộ nhớ khi chuyển đổi video sequence hoặc camera mới."""
        if device is not None:
            key = self._device_to_key(device)
            if key in self._states:
                self._states[key].reset()
        else:
            with self._state_lock:
                for s in self._states.values():
                    s.reset()

    @property
    def is_empty(self) -> bool:
        """Kiểm tra xem bộ nhớ hiện có khung hình nào không."""
        return self._active_state.is_empty

    @property
    def num_stored_frames(self) -> int:
        """Tổng số khung hình hiện đang lưu trữ."""
        return self._active_state.num_stored_frames

    def get_spatial_pos_embed(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """Tạo hoặc lấy Positional Embedding 2D sin-cos được cache."""
        dev_key = self._device_to_key(device)
        cache_key = f"{h}x{w}_{dev_key}"
        if cache_key in self._cached_spatial_pe:
            return self._cached_spatial_pe[cache_key]

        pe = build_2d_sincos_position_embedding(h, w, self.embed_dim, device)
        self._cached_spatial_pe[cache_key] = pe
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
          5. Cách ly bộ nhớ theo thiết bị phần cứng z.device triệt tiêu race condition trên Multi-GPU.
        """
        # 1. Bắt buộc detach() triệt để
        z = bg_tokens.detach()
        B, L, D = z.shape
        state = self._get_state(z.device)

        # 2. Xây dựng valid mask (B, L) kết hợp cả alpha_p và sigma_p
        pool_hw = spatial_hw
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
            step_idx = len(state.recent_tokens)
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
        if state.long_term_memory is None:
            state.long_term_memory = z.clone()
            state.long_term_valid = valid.clone()
        else:
            # Tăng tốc tẩy xe: nếu vùng trước đó bị kẹt xe (long_term_valid < 0.5), lập tức nhận nền mới khi valid >= 0.5
            fast_replace = (state.long_term_valid.to(z.device) < 0.5).unsqueeze(-1) & (valid.unsqueeze(-1) >= 0.5)
            effective_beta = torch.where(fast_replace, torch.ones_like(beta), beta)
            state.long_term_memory = (1.0 - effective_beta) * state.long_term_memory.to(z.device) + effective_beta * z
            state.long_term_valid = torch.maximum(state.long_term_valid.to(z.device), valid)

        # 6. Anchor Selection theo Dual-Slot (Day & Night) loại bỏ bẫy 2 AM:
        # Giờ địa phương Việt Nam (GMT+7: UTC + 7*3600)
        hour_vn = ((ts_tensor + 7.0 * 3600.0) % 86400.0) / 3600.0  # (B,)
        is_day_sample = (hour_vn >= 6.0) & (hour_vn < 18.0)        # (B,)
        alpha_per_sample = (1.0 - valid).mean(dim=-1)               # (B,)

        # A. Cập nhật Slot Ban Ngày (Day Anchor)
        if is_day_sample.any():
            if state.anchor_day_tokens is None or state.anchor_day_alpha_mean is None or state.anchor_day_alpha_mean.shape[0] != B:
                state.anchor_day_tokens = z.clone()
                state.anchor_day_valid = valid.clone()
                state.anchor_day_timestamp = ts_tensor.clone()
                state.anchor_day_mean_rgb = mean_rgb_tensor.clone()
                state.anchor_day_alpha_mean = alpha_per_sample.clone()
            else:
                up_day = is_day_sample & (is_anchor | (alpha_per_sample < state.anchor_day_alpha_mean.to(z.device)))
                if up_day.any():
                    u_3d = up_day.view(B, 1, 1)
                    u_2d = up_day.view(B, 1)
                    state.anchor_day_tokens = torch.where(u_3d, z, state.anchor_day_tokens.to(z.device))
                    state.anchor_day_valid = torch.where(u_2d, valid, state.anchor_day_valid.to(z.device))
                    state.anchor_day_timestamp = torch.where(up_day, ts_tensor, state.anchor_day_timestamp.to(z.device))
                    state.anchor_day_mean_rgb = torch.where(u_2d, mean_rgb_tensor, state.anchor_day_mean_rgb.to(z.device))
                    state.anchor_day_alpha_mean = torch.where(up_day, alpha_per_sample, state.anchor_day_alpha_mean.to(z.device))

        # B. Cập nhật Slot Ban Đêm (Night Anchor)
        is_night_sample = ~is_day_sample
        if is_night_sample.any():
            if state.anchor_night_tokens is None or state.anchor_night_alpha_mean is None or state.anchor_night_alpha_mean.shape[0] != B:
                state.anchor_night_tokens = z.clone()
                state.anchor_night_valid = valid.clone()
                state.anchor_night_timestamp = ts_tensor.clone()
                state.anchor_night_mean_rgb = mean_rgb_tensor.clone()
                state.anchor_night_alpha_mean = alpha_per_sample.clone()
            else:
                up_night = is_night_sample & (is_anchor | (alpha_per_sample < state.anchor_night_alpha_mean.to(z.device)))
                if up_night.any():
                    u_3d = up_night.view(B, 1, 1)
                    u_2d = up_night.view(B, 1)
                    state.anchor_night_tokens = torch.where(u_3d, z, state.anchor_night_tokens.to(z.device))
                    state.anchor_night_valid = torch.where(u_2d, valid, state.anchor_night_valid.to(z.device))
                    state.anchor_night_timestamp = torch.where(up_night, ts_tensor, state.anchor_night_timestamp.to(z.device))
                    state.anchor_night_mean_rgb = torch.where(u_2d, mean_rgb_tensor, state.anchor_night_mean_rgb.to(z.device))
                    state.anchor_night_alpha_mean = torch.where(up_night, alpha_per_sample, state.anchor_night_alpha_mean.to(z.device))

        # 7. Ghi vào hàng đợi Recent Frames (chọn lọc theo ánh sáng nếu vượt giới hạn)
        if len(state.recent_tokens) >= self.max_recent_frames:
            evict_idx = 0
            if len(state.recent_mean_rgb) > 0 and mean_rgb is not None:
                dists = [
                    float(torch.norm(m.to(z.device) - mean_rgb_tensor, dim=-1).mean().item())
                    for m in state.recent_mean_rgb
                ]
                if len(dists) > 0:
                    evict_idx = int(torch.tensor(dists).argmax().item())

            # An toàn tuyệt đối chống IndexError đa luồng
            if 0 <= evict_idx < len(state.recent_tokens):
                state.recent_tokens.pop(evict_idx)
            elif len(state.recent_tokens) > 0:
                state.recent_tokens.pop(0)

            if 0 <= evict_idx < len(state.recent_valid):
                state.recent_valid.pop(evict_idx)
            elif len(state.recent_valid) > 0:
                state.recent_valid.pop(0)

            if 0 <= evict_idx < len(state.recent_timestamps):
                state.recent_timestamps.pop(evict_idx)
            elif len(state.recent_timestamps) > 0:
                state.recent_timestamps.pop(0)

            if 0 <= evict_idx < len(state.recent_mean_rgb):
                state.recent_mean_rgb.pop(evict_idx)
            elif len(state.recent_mean_rgb) > 0:
                state.recent_mean_rgb.pop(0)

        state.recent_tokens.append(z)
        state.recent_valid.append(valid)
        state.recent_timestamps.append(ts_tensor)
        state.recent_mean_rgb.append(mean_rgb_tensor)

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
        state = self._get_state(current_device)
        if state.is_empty:
            return None, None, None, True

        # Sắp xếp các recent frames theo độ tương đồng ánh sáng với query_mean_rgb (nếu có)
        recent_indices = list(range(len(state.recent_tokens)))
        if (
            query_mean_rgb is not None
            and len(state.recent_tokens) > 1
            and len(state.recent_mean_rgb) == len(state.recent_tokens)
        ):
            q_rgb = query_mean_rgb.detach().to(current_device)
            dists = [
                float(torch.norm(m.to(current_device) - q_rgb, dim=-1).mean().item())
                for m in state.recent_mean_rgb
            ]
            recent_indices = sorted(range(len(dists)), key=lambda idx: dists[idx])

        raw_list: List[torch.Tensor] = []
        valid_list: List[torch.Tensor] = []
        ts_list: List[torch.Tensor] = []

        # 1. Đưa Long-term Background Memory M vào danh sách (độ tin cậy nền tích lũy)
        if state.long_term_memory is not None:
            raw_list.append(state.long_term_memory.to(current_device))
            B_m, L_m, _ = state.long_term_memory.shape
            lt_valid = (
                state.long_term_valid.to(current_device)
                if state.long_term_valid is not None
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

        primary_anchor = state.anchor_day_tokens if query_is_day else state.anchor_night_tokens
        primary_valid = state.anchor_day_valid if query_is_day else state.anchor_night_valid
        primary_ts = state.anchor_day_timestamp if query_is_day else state.anchor_night_timestamp

        fallback_anchor = state.anchor_night_tokens if query_is_day else state.anchor_day_tokens
        fallback_valid = state.anchor_night_valid if query_is_day else state.anchor_day_valid
        fallback_ts = state.anchor_night_timestamp if query_is_day else state.anchor_day_timestamp

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
            raw_list.append(state.recent_tokens[r_idx].to(current_device))
            valid_list.append(state.recent_valid[r_idx].to(current_device))
            ts_list.append(state.recent_timestamps[r_idx].to(current_device))

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
