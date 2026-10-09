"""
=============================================================================
 ST-BMB: Streaming Decomposition Model Architecture
 Mạng Phân Rã Cảnh & Phân Đoạn Phương Tiện Tự Giám Sát Qua Bộ Nhớ Nền (ST-BMB)
=============================================================================
Cải tiến đột phá chuẩn Production:
  1. Tích hợp chặt chẽ Vision Transformer Backbone với Point-wise Temporal Attention
     và Spatial Sin-Cos Positional Embedding.
  2. Tạo mask alpha_p ở cấp độ patch (downsample hat_alpha xuống Hp x Wp) phục vụ
     Gated Write và bias valid cho attention.
  3. Bắt buộc Token Detach khi ghi bộ nhớ BMB (chống phình VRAM và gian lận shortcut).
  4. Đầu ra của sigma sử dụng F.softplus(raw_sigma) + 0.01 đảm bảo trơn tru,
     triệt tiêu hoàn toàn hiện tượng phẳng gradient của hàm clamp.
  5. Đồng bộ giao tiếp với BackgroundMemoryBank và PhotometricIlluminationAdaptor.
=============================================================================
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from common.backbone_loader import imagenet_normalize
except ImportError:
    def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

from .memory import BackgroundMemoryBank
from .attention import MemoryCrossAttention


class ConvBlock(nn.Module):
    """Khối tích chập kép cơ bản với GroupNorm và GELU."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BackgroundMemoryEncoder(nn.Module):
    """
    Memory Encoder nén thông tin nền đường sạch của khung hình t
    chuẩn bị ghi (WRITE) vào Background Memory Bank.
    
    Đầu vào:
      - Nền dự đoán B_hat (3 kênh RGB)
      - Trọng số tự tin nền (1 - M_alpha) (1 kênh)
      - Độ bất định quang học sigma (1 kênh)
    Tổng cộng 5 kênh đầu vào, nén thành tensor đặc trưng cùng không gian (Hp, Wp, D).
    """
    def __init__(self, in_channels: int = 5, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),  # /2
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),          # /4
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),         # /8
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),   # /16
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def forward(
        self,
        pred_bg: torch.Tensor,
        alpha_mask: torch.Tensor,
        sigma: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Nén nền dự đoán thành memory tokens.
        Returns:
            bg_tokens: (B, Hp * Wp, embed_dim)
        """
        bg_confidence = 1.0 - alpha_mask
        # Ghép 5 kênh: (B, 5, H, W)
        mem_input = torch.cat([pred_bg, bg_confidence, sigma], dim=1)
        feat_map = self.conv(mem_input)
        feat_map = self.proj(feat_map)

        if feat_map.shape[-2:] != target_hw:
            feat_map = F.interpolate(feat_map, size=target_hw, mode="bilinear", align_corners=False)

        # Chuyển đổi về dạng token (B, Hp * Wp, embed_dim)
        B, D, Hp, Wp = feat_map.shape
        bg_tokens = feat_map.flatten(2).transpose(1, 2).contiguous()
        return bg_tokens


class MultiScaleDecompDecoder(nn.Module):
    """
    Decoder đa tỉ lệ kiểu DPT / Feature Pyramid:
    Từ lưới patch token không gian (Hp, Wp) upsample dần 4 bậc lên kích thước ảnh (H, W).
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)

        self.up1 = nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = ConvBlock(64, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = ConvBlock(32, 32)

        self.up4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = ConvBlock(32, 32)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        feat = self.proj(x)
        feat = self.conv1(self.up1(feat))
        feat = self.conv2(self.up2(feat))
        feat = self.conv3(self.up3(feat))
        feat = self.conv4(self.up4(feat))
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return feat


class PhotometricIlluminationAdaptor(nn.Module):
    """
    Khối ước lượng và căn chỉnh quang sai / biến thiên ánh sáng thời gian thực (ST-BMB).
    Ước lượng:
      - Gain g in [0.2, 1.8] (RGB channel-wise)
      - Bias b in [-0.3, 0.3] (RGB channel-wise)
      - Soft Label Weight w_illum in [0.2, 1.0]
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3 + 3 + 1),  # 3 gain (RGB) + 3 bias (RGB) + 1 soft_weight
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)
        self.net[-1].bias.data[6] = 2.0  # sigmoid(2.0) ~ 0.88

    def forward(
        self,
        token_t: torch.Tensor,       # (B, embed_dim)
        token_prev: torch.Tensor,    # (B, embed_dim)
        frame_t: torch.Tensor,       # (B, 3, H, W)
        frame_prev: torch.Tensor,    # (B, 3, H, W)
        bg_prev: torch.Tensor,       # (B, 3, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = token_t.shape[0]
        mean_diff = frame_t.mean(dim=[-2, -1]) - frame_prev.mean(dim=[-2, -1])
        feat_in = torch.cat([token_t, token_prev, mean_diff], dim=1)  # (B, 2*D + 3)

        out = self.net(feat_in)
        raw_gain = out[:, 0:3]
        raw_bias = out[:, 3:6]
        raw_soft = out[:, 6:7]

        gain = (1.0 + torch.tanh(raw_gain) * 0.8).view(B, 3, 1, 1)
        bias = (torch.tanh(raw_bias) * 0.3).view(B, 3, 1, 1)
        soft_weight = (0.2 + 0.8 * torch.sigmoid(raw_soft)).view(B, 1, 1, 1)

        # Loại bỏ clamp cứng để gradient từ Laplace NLL và Chroma Loss truyền thông suốt đến gain và bias
        bg_prev_aligned = gain * bg_prev + bias
        return bg_prev_aligned, gain, bias, soft_weight


class StreamingDecompositionNet(nn.Module):
    """
    Mạng phân rã cảnh giao thông qua Bộ nhớ nền Spatio-Temporal Background Memory Bank (ST-BMB).
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int = 384,
        embed_dim: int = 256,
        patch_size: int = 16,
        max_recent_frames: int = 5,
        max_anchor_frames: int = 1,
        freeze_backbone: bool = True,
        unfreeze_last_blocks: int = 0,
        num_heads: int = 8,
        ema_eta: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.memory_dropout_p = 0.0

        # Cấu hình đóng băng backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif unfreeze_last_blocks > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if hasattr(self.backbone, "blocks"):
                blocks = self.backbone.blocks
                for b in blocks[-unfreeze_last_blocks:]:
                    for p in b.parameters():
                        p.requires_grad = True

        # Chiếu đặc trưng từ ViT Backbone sang không gian Memory Bank
        self.visual_proj = nn.Linear(backbone_dim, embed_dim)

        # Background Memory Bank & Memory Cross-Attention
        self.memory_bank = BackgroundMemoryBank(
            embed_dim=embed_dim,
            max_recent_frames=max_recent_frames,
            max_anchor_frames=max_anchor_frames,
            spatial_patch_size=(patch_size, patch_size),
            ema_eta=ema_eta,
        )
        self.memory_attn = MemoryCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            dropout=0.05,
        )

        # Background Memory Encoder (Khối Write)
        self.memory_encoder = BackgroundMemoryEncoder(
            in_channels=5,
            embed_dim=embed_dim,
        )

        # Photometric Illumination Adaptor
        self.illumination_adaptor = PhotometricIlluminationAdaptor(embed_dim=embed_dim)

        # Decoder chính
        self.decoder = MultiScaleDecompDecoder(in_dim=embed_dim, hidden_dim=256)

        # Skip-connection RGB
        self.rgb_skip = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )

        # 4 Chuyên biệt Heads:
        # Alpha Matte
        self.head_alpha = nn.Sequential(
            nn.Conv2d(32 + 1, 16, kernel_size=3, padding=1),  # +1 kênh từ bg_familiarity upsample
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Foreground
        self.head_fg = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Clean Road Background
        self.head_bg = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Optical Uncertainty Map (dùng softplus + 0.01 đảm bảo gradient trơn tru)
        self.head_sigma = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def set_memory_dropout(self, p: float):
        """Đặt tỷ lệ Memory Dropout khi huấn luyện."""
        self.memory_dropout_p = float(p)

    def extract_visual_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Trích xuất visual tokens từ Vision Transformer (B, L, embed_dim) cộng 2D sin-cos PE."""
        B, C, H, W = x.shape
        Hp = H // self.patch_size
        Wp = W // self.patch_size

        x_norm = imagenet_normalize(x)
        feats = self.backbone(x_norm)
        if isinstance(feats, dict):
            patch_tokens = feats.get("x_norm_patchtokens", None)
            if patch_tokens is None:
                patch_tokens = list(feats.values())[0]
        elif isinstance(feats, torch.Tensor):
            if feats.dim() == 3:
                patch_tokens = feats[:, 1:] if feats.shape[1] > Hp * Wp else feats
            elif feats.dim() == 2:
                patch_tokens = feats.unsqueeze(1).expand(-1, Hp * Wp, -1)
            else:
                patch_tokens = feats
        else:
            patch_tokens = feats[0]

        # Ánh xạ về embed_dim của memory system: (B, L, embed_dim)
        visual_tokens = self.visual_proj(patch_tokens)

        # Cộng 2D Sin-Cos Spatial Positional Embedding
        spatial_pe = self.memory_bank.get_spatial_pos_embed(Hp, Wp, visual_tokens.device)
        visual_tokens = visual_tokens + spatial_pe

        return visual_tokens, (Hp, Wp)

    def forward_single_step(
        self,
        x: torch.Tensor,
        update_memory: bool = True,
        is_anchor: bool = False,
        timestamp: Optional[Union[float, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Xử lý 1 khung hình đơn lẻ trong luồng streaming:
          1. READ: Point-wise Temporal Attention truy vấn BMB.
          2. PREDICT: MultiScale Decoder sinh alpha, FG, BG, Sigma (softplus).
          3. WRITE: Gated Write có cổng, token detached vào BMB.
        """
        B, C, H, W = x.shape
        visual_tokens, (Hp, Wp) = self.extract_visual_tokens(x)
        mean_rgb = x.mean(dim=[-2, -1]).detach()  # (B, 3)

        # 1. READ: Truy vấn Memory Bank kết hợp độ tương đồng ánh sáng và kích thước lưới patch
        mem_tokens, raw_mem, mem_valid, is_empty = self.memory_bank.read(
            current_device=x.device,
            query_timestamp=timestamp,
            query_mean_rgb=mean_rgb,
            memory_dropout_p=self.memory_dropout_p if self.training else 0.0,
            spatial_hw=(Hp, Wp),
        )

        cond_tokens, bg_fam = self.memory_attn(
            q_tokens=visual_tokens,
            mem_tokens=mem_tokens,
            raw_mem_tokens=raw_mem,
            valid=mem_valid,
            is_empty_memory=is_empty,
        )

        # 2. Reshape về không gian 2D cho Decoder: (B, embed_dim, Hp, Wp)
        feat_map = cond_tokens.transpose(1, 2).contiguous().view(B, self.embed_dim, Hp, Wp)

        # 3. Upsample Decoder kết hợp RGB Skip
        dec_feat = self.decoder(feat_map, target_hw=(H, W))
        dec_feat = dec_feat + self.rgb_skip(x)

        # Background Familiarity upsample lên kích thước ảnh (H, W)
        bg_fam_map = bg_fam.transpose(1, 2).contiguous().view(B, 1, Hp, Wp)
        bg_fam_upsampled = F.interpolate(bg_fam_map, size=(H, W), mode="bilinear", align_corners=False)

        # 4. Dự đoán 4 Heads
        alpha_in = torch.cat([dec_feat, bg_fam_upsampled], dim=1)
        alpha_mask = self.head_alpha(alpha_in)
        pred_fg = self.head_fg(dec_feat)
        pred_bg = self.head_bg(dec_feat)

        # Sigma: softplus trơn tru + 0.01 đảm bảo gradient không bao giờ bị triệt tiêu
        raw_sigma = self.head_sigma(dec_feat)
        sigma = F.softplus(raw_sigma) + 0.01

        # Tái tạo ảnh vật lý: I_recon = alpha * F + (1 - alpha) * B
        recon_origin = alpha_mask * pred_fg + (1.0 - alpha_mask) * pred_bg

        # 5. Tạo mask alpha_p và sigma_p cấp độ patch (Hp, Wp) cho Gated Write và valid bias
        alpha_p = F.adaptive_avg_pool2d(alpha_mask, (Hp, Wp)).flatten(1)  # (B, L)
        sigma_p = F.adaptive_avg_pool2d(sigma, (Hp, Wp)).flatten(1)        # (B, L)

        # 6. WRITE: Gated Write có cổng vào Memory Bank cho frame kế tiếp
        if update_memory:
            # Token Detach BẮT BUỘC: Cắt đứt gradient lịch sử
            new_bg_tokens = self.memory_encoder(
                pred_bg=pred_bg.detach(),
                alpha_mask=alpha_mask.detach(),
                sigma=sigma.detach(),
                target_hw=(Hp, Wp),
            )
            self.memory_bank.write(
                bg_tokens=new_bg_tokens.detach(),
                alpha_p=alpha_p.detach(),
                sigma_p=sigma_p.detach(),
                timestamp=timestamp,
                mean_rgb=mean_rgb,
                is_anchor=is_anchor,
                spatial_hw=(Hp, Wp),
            )

        return {
            "alpha_mask": alpha_mask,       # M_alpha: (B, 1, H, W)
            "pred_mask": alpha_mask,        # Alias
            "pred_fg": pred_fg,             # F: (B, 3, H, W)
            "pred_bg": pred_bg,             # B_hat: (B, 3, H, W)
            "sigma": sigma,                 # sigma: (B, 1, H, W)
            "alpha_p": alpha_p,             # (B, L)
            "recon_origin": recon_origin,   # I_recon: (B, 3, H, W)
            "bg_familiarity": bg_fam_upsampled,
            "visual_tokens_global": visual_tokens.mean(dim=1),  # (B, embed_dim)
        }

    def forward_sequence(
        self,
        frames: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        reset_memory: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Xử lý chuỗi W khung hình liên tiếp của cùng camera.
        Tự động chọn anchor frame sạch nhất và thực hiện căn chỉnh quang sai sáng/chiều.
        Args:
            frames: Tensor shape (B, W, 3, H, W).
            timestamps: Tensor timestamps shape (B, W) hoặc None.
            reset_memory: Boolean xóa bộ nhớ trước khi xử lý chuỗi.
        Returns:
            step_outputs: Danh sách W kết quả dự đoán.
        """
        B, W_len, C, H, W = frames.shape
        if reset_memory:
            self.memory_bank.reset(frames.device)

        step_outputs: List[Dict[str, torch.Tensor]] = []
        for t in range(W_len):
            cur_frame = frames[:, t]
            cur_ts = timestamps[:, t] if timestamps is not None else None
            # Frame đầu tiên t=0 khởi tạo anchor ban đầu, các frame sau tự động cập nhật per-sample nếu sạch hơn
            is_anchor = (t == 0)

            out_t = self.forward_single_step(
                cur_frame,
                update_memory=True,
                is_anchor=is_anchor,
                timestamp=cur_ts,
            )

            if t == 0:
                out_t["bg_prev_aligned"] = None
                out_t["illum_gain"] = torch.ones(B, 3, 1, 1, device=frames.device)
                out_t["illum_bias"] = torch.zeros(B, 3, 1, 1, device=frames.device)
                out_t["illum_soft_weight"] = torch.ones(B, 1, 1, 1, device=frames.device)
            else:
                cur_tok = out_t["visual_tokens_global"]
                prev_tok = step_outputs[t - 1]["visual_tokens_global"].detach()
                prev_bg = step_outputs[t - 1]["pred_bg"].detach()
                prev_frame = frames[:, t - 1].detach()
                bg_aligned, gain, bias, soft_w = self.illumination_adaptor(
                    token_t=cur_tok,
                    token_prev=prev_tok,
                    frame_t=cur_frame,
                    frame_prev=prev_frame,
                    bg_prev=prev_bg,
                )
                out_t["bg_prev_aligned"] = bg_aligned
                out_t["illum_gain"] = gain
                out_t["illum_bias"] = bias
                out_t["illum_soft_weight"] = soft_w

            step_outputs.append(out_t)

        return step_outputs

    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        reset_memory: bool = False,
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        Hàm forward đa năng:
        - 5D (B, W, 3, H, W): Chạy `forward_sequence`.
        - 4D (B, 3, H, W): Chạy `forward_single_step`.
        """
        if x.dim() == 5:
            return self.forward_sequence(x, timestamps=timestamps, reset_memory=reset_memory)
        elif x.dim() == 4:
            if reset_memory:
                self.memory_bank.reset(x.device)
            return self.forward_single_step(x, update_memory=True, timestamp=timestamps)
        else:
            raise ValueError(f"Tensor đầu vào không hợp lệ: shape={x.shape}. Cần 4D hoặc 5D.")

    def reset_memory(self, device: Optional[Union[torch.device, str, int]] = None):
        """Xóa sạch bộ nhớ của thiết bị chỉ định hoặc toàn bộ các thiết bị (Thread-Safe & Multi-GPU)."""
        self.memory_bank.reset(device)
