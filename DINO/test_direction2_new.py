"""
=============================================================================
 Test Suite: Direction 2 New (test_direction2_new.py)
 Kiểm thử tích hợp toàn diện quy trình 2 giai đoạn Hướng 2 Mới:
 1. Giai đoạn 1: SceneBasis (E0, Ej, ell) & Robust IRLS weights
 2. Chế độ T: solve_ell Weighted Least Squares IRLS solver
 3. Giai đoạn 2: TrafficDecompositionNet (5 heads: M, F, B, sigma, ell)
 4. Hàm mất mát SceneDecompositionLossV2 (Laplace NLL, DropLoss, Lighting loss)
=============================================================================
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_dino_root = os.path.dirname(os.path.abspath(__file__))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from direction2_scene_decomposition.scene_fit import SceneBasis, robust_weights
from direction2_scene_decomposition.solve_ell import solve_ell
from direction2_scene_decomposition.models import TrafficDecompositionNet
from direction2_scene_decomposition.losses import SceneDecompositionLossV2


class MockViTBackbone(nn.Module):
    """Backbone giả lập để test nhanh không phụ thuộc GPU."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        # Giả lập blocks
        self.blocks = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(6)])

    def forward(self, x):
        B, C, H, W = x.shape
        Hp = H // 16
        Wp = W // 16
        return torch.randn(B, Hp * Wp, self.embed_dim, device=x.device)


class TestDirection2New(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def test_01_scene_basis_and_robust_weights(self):
        """Kiểm tra SceneBasis và hàm tính trọng số robust_weights."""
        N, C, H, W = 4, 3, 256, 448
        frames = torch.rand(N, C, H, W, device=self.device)
        pi = torch.zeros(N, 16, 28, device=self.device)
        pi[:, 5:10, 10:15] = 0.9  # Vùng xe

        basis = SceneBasis(num_cams=1, num_frames_per_cam=N, J=4, device=self.device)
        basis.init_from_frames(cam_idx=0, frames=frames, pi=pi)

        # Tái tạo nền B
        indices = torch.arange(N, device=self.device)
        B_out = basis.get_background(cam_idx=0, frame_indices=indices)
        self.assertEqual(B_out.shape, (N, 3, 256, 448))

        # Tính robust weights
        w, w_res = robust_weights(frames, B_out, pi=pi)
        self.assertEqual(w.shape, (N, 1, 256, 448))
        self.assertEqual(w_res.shape, (N, 1, 256, 448))
        self.assertTrue((w >= 0.0).all() and (w <= 1.0).all())

    def test_02_solve_ell_irls(self):
        """Kiểm tra solve_ell giải mã ánh sáng bằng IRLS."""
        H, W = 256, 448
        E0 = torch.full((3, H, W), 0.4, device=self.device)
        Ej = torch.randn(4, 3, 128, 224, device=self.device) * 0.05
        # Tạo frame tổng hợp với ell thật
        ell_gt = torch.tensor([0.5, -0.2, 0.8, -0.1], device=self.device)
        Ej_up = F.interpolate(Ej, size=(H, W), mode="bilinear", align_corners=False)
        I = E0 + torch.einsum("j,jchw->chw", ell_gt, Ej_up)

        # Giải ell
        ell_est = solve_ell(I, E0, Ej, iters=5)
        self.assertEqual(ell_est.shape, (4,))
        # Sai số giải tích phải rất nhỏ
        err = torch.abs(ell_est - ell_gt).max().item()
        self.assertLess(err, 0.05, f"Sai số solve_ell lớn ({err:.4f})!")

    def test_03_traffic_decomp_net_5_heads(self):
        """Kiểm tra TrafficDecompositionNet 5 đầu ra và unfreeze_last_blocks."""
        backbone = MockViTBackbone(embed_dim=128)
        model = TrafficDecompositionNet(
            backbone=backbone,
            embed_dim=128,
            patch_size=16,
            unfreeze_last_blocks=2,
            num_light_codes=4,
        ).to(self.device)

        x = torch.rand(2, 3, 256, 448, device=self.device)
        pi = torch.zeros(2, 16, 28, device=self.device)
        out = model(x, pi=pi)

        self.assertIn("alpha_mask", out)
        self.assertIn("pred_fg", out)
        self.assertIn("pred_bg", out)
        self.assertIn("sigma", out)
        self.assertIn("pred_ell", out)
        self.assertEqual(out["alpha_mask"].shape, (2, 1, 256, 448))
        self.assertEqual(out["pred_fg"].shape, (2, 3, 256, 448))
        self.assertEqual(out["pred_bg"].shape, (2, 3, 256, 448))
        self.assertEqual(out["sigma"].shape, (2, 1, 256, 448))
        self.assertEqual(out["pred_ell"].shape, (2, 4))

    def test_04_loss_v2_backward(self):
        """Kiểm tra SceneDecompositionLossV2 tính toán loss và backward không NaN."""
        loss_fn = SceneDecompositionLossV2()
        B, C, H, W = 2, 3, 256, 448
        preds = {
            "recon_origin": torch.rand(B, C, H, W, device=self.device, requires_grad=True),
            "pred_bg": torch.rand(B, C, H, W, device=self.device, requires_grad=True),
            "pred_fg": torch.rand(B, C, H, W, device=self.device, requires_grad=True),
            "alpha_mask": torch.rand(B, 1, H, W, device=self.device, requires_grad=True),
            "sigma": torch.full((B, 1, H, W), 0.1, device=self.device, requires_grad=True),
            "log_sigma": torch.full((B, 1, H, W), -2.3, device=self.device, requires_grad=True),
            "pred_ell": torch.zeros(B, 4, device=self.device, requires_grad=True),
        }
        origin = torch.rand(B, C, H, W, device=self.device)
        bg_pseudo = torch.rand(B, C, H, W, device=self.device)
        mask_pseudo = torch.zeros(B, 1, H, W, device=self.device)
        conf = torch.ones(B, 1, H, W, device=self.device)
        ell_gt = torch.zeros(B, 4, device=self.device)
        pi = torch.zeros(B, 16, 28, device=self.device)

        total_loss, loss_dict = loss_fn(
            preds=preds,
            origin=origin,
            bg_pseudo=bg_pseudo,
            mask_pseudo=mask_pseudo,
            conf=conf,
            ell_gt=ell_gt,
            pi=pi,
        )

        total_loss.backward()
        self.assertFalse(torch.isnan(total_loss))
        self.assertIsNotNone(preds["pred_bg"].grad)
        self.assertIsNotNone(preds["alpha_mask"].grad)


if __name__ == "__main__":
    unittest.main()
