"""
=============================================================================
 Hướng B: Weak Supervision — End Model (DINOv3 + Causal GRU)
 Mô hình suy luận cuối cùng được huấn luyện bằng Nhãn mềm (Soft Cross-Entropy)
 Tự chủ suy luận 100% từ ảnh Camera thời gian thực mà KHÔNG CẦN bất kỳ LF nào
=============================================================================
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    """Hàm mất mát Cross-Entropy trên phân phối xác suất mềm: - sum q(y) * log p(y)."""
    def __init__(self, min_confidence_threshold: float = 0.40):
        super().__init__()
        self.min_conf = min_confidence_threshold

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes)
            soft_targets: (B, num_classes)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        # Bỏ qua các frame có độ tự tin quá thấp (max q < min_conf)
        max_q = torch.max(soft_targets, dim=-1)[0]
        valid_mask = max_q >= self.min_conf

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss_per_sample = -torch.sum(soft_targets[valid_mask] * log_probs[valid_mask], dim=-1)
        return loss_per_sample.mean()


class WeakSupervisionEndModel(nn.Module):
    """
    End Model chuẩn Meta DINOv3 + Causal Recurrent Head:
      - Backbone: DINOv3 ViT trích xuất biểu diễn ngữ nghĩa không gian sâu.
      - Causal Temporal GRU: Mô hình hóa động lực ùn tắc nhân quả (chỉ nhìn về quá khứ).
      - Classifier Head: Dự đoán phân bố xác suất p(y) trên 4 cấp độ ùn tắc.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 384,
        hidden_dim: int = 256,
        num_classes: int = 4,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Tầng thời gian nhân quả Causal GRU (1 chiều)
        self.temporal_gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,  # Bắt buộc Causal để không rò rỉ tương lai
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    def extract_cls(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, dict):
            cls_token = out.get("x_norm_clstoken", list(out.values())[0])
        elif isinstance(out, torch.Tensor):
            cls_token = out[:, 0] if out.dim() == 3 else out
        else:
            cls_token = out[0]
        return cls_token

    def forward(self, rgb_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_seq: Tensor chuỗi ảnh (B, T, 3, H, W).
        Returns:
            logits: (B, num_classes) dự đoán mức ùn tắc tại thời điểm t = T.
        """
        B, T, C, H, W = rgb_seq.shape
        flat_x = rgb_seq.view(B * T, C, H, W)
        cls_feats = self.extract_cls(flat_x)            # (B * T, embed_dim)
        cls_seq = cls_feats.view(B, T, self.embed_dim)  # (B, T, embed_dim)

        gru_out, _ = self.temporal_gru(cls_seq)         # (B, T, hidden_dim)
        last_hidden = gru_out[:, -1, :]                 # (B, hidden_dim)
        logits = self.classifier(last_hidden)           # (B, num_classes)
        return logits
