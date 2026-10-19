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

_dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from direction2_new.scene_fit import SceneBasis, robust_weights
from direction2_new.solve_ell import solve_ell
from direction2_new.models import TrafficDecompositionNet
from direction2_new.losses import SceneDecompositionLossV2


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

    def test_05_direction2_smooth_lr_scheduler_continuity(self):
        """Kiểm tra build_smooth_lr_scheduler cho Hướng 2 đảm bảo C^0 continuity tuyệt đối giữa các Session."""
        from common.gpu_utils import build_smooth_lr_scheduler

        param1 = torch.nn.Parameter(torch.zeros(10))
        base_lr = 3e-4
        eta_min = 1e-6
        total_epochs = 15
        warmup_epochs = 1

        # PHIÊN 1 (Session 1): Huấn luyện 4 epochs (Epoch 0 -> 3)
        opt1 = torch.optim.AdamW([param1], lr=base_lr)
        sched1 = build_smooth_lr_scheduler(opt1, total_epochs=total_epochs, warmup_epochs=warmup_epochs, eta_min=eta_min)

        lrs_session_1 = []
        for ep in range(4):
            lrs_session_1.append(opt1.param_groups[0]["lr"])
            opt1.step()
            sched1.step()

        # LR ở đầu Epoch 4 (sẵn sàng cho epoch tiếp theo sau khi kết thúc session 1)
        lr_end_session_1 = opt1.param_groups[0]["lr"]

        # PHIÊN 2 (Session 2): Resume từ Checkpoint sau 4 epochs (start_epoch = 4, last_epoch = 3)
        param2 = torch.nn.Parameter(torch.zeros(10))
        opt2 = torch.optim.AdamW([param2], lr=base_lr)
        sched2 = build_smooth_lr_scheduler(
            opt2,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            eta_min=eta_min,
            last_epoch=3,
        )
        lr_start_session_2 = opt2.param_groups[0]["lr"]

        # KIỂM ĐỊNH TOÁN HỌC: LR tại điểm tiếp nối của Session 2 phải bằng đúng LR kết thúc Session 1!
        self.assertAlmostEqual(lr_end_session_1, lr_start_session_2, places=7)

        # Chạy tiếp 4 epochs của Session 2 (Epoch 4 -> 7)
        lrs_session_2 = []
        for ep in range(4, 8):
            lrs_session_2.append(opt2.param_groups[0]["lr"])
            opt2.step()
            sched2.step()

        # Kiểm tra tính đơn điệu giảm của LR sau thời kỳ warmup (Epoch 1 -> 7)
        all_cosine_lrs = lrs_session_1[warmup_epochs:] + lrs_session_2
        for i in range(len(all_cosine_lrs) - 1):
            self.assertGreaterEqual(all_cosine_lrs[i], all_cosine_lrs[i + 1])

    def test_06_direction2_session_checkpoint_metadata(self):
        """Kiểm tra việc lưu và khôi phục metadata Session-based trong Checkpoint."""
        import tempfile
        from common.gpu_utils import save_checkpoint, load_checkpoint

        backbone = MockViTBackbone(embed_dim=128)
        model = TrafficDecompositionNet(backbone=backbone, embed_dim=128, patch_size=16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_session_ckpt.pth")
            save_checkpoint(
                save_path=ckpt_path,
                model=model,
                optimizer=opt,
                epoch=4,
                extra_dict={
                    "total_epochs": 15,
                    "session_epochs": 4,
                    "warmup_epochs": 1,
                    "completed_epoch": 3,
                },
                verbose=False,
            )

            loaded = load_checkpoint(ckpt_path, model=model, optimizer=opt, device="cpu", verbose=False)
            self.assertEqual(loaded.get("total_epochs"), 15)
            self.assertEqual(loaded.get("session_epochs"), 4)
            self.assertEqual(loaded.get("warmup_epochs"), 1)
            self.assertEqual(loaded.get("completed_epoch"), 3)
            self.assertEqual(loaded.get("epoch"), 4)

    def test_07_direction2_cli_specified_args(self):
        """Kiểm tra get_cli_specified_args nhận diện đúng các đối số của Hướng 2."""
        from common.gpu_utils import get_cli_specified_args

        test_argv = [
            "--resume", "checkpoints/direction2_scene_decomp/last_checkpoint.pth",
            "--session_epochs", "4",
            "--warmup_epochs", "1",
            "--origin_dir", "data/output",
        ]
        specified = get_cli_specified_args(test_argv)
        self.assertIn("resume", specified)
        self.assertIn("session_epochs", specified)
        self.assertIn("warmup_epochs", specified)
        self.assertIn("origin_dir", specified)
        self.assertNotIn("epochs", specified)
        self.assertNotIn("lr", specified)

    def test_08_smooth_sigma_gradient_continuity(self):
        """Kiểm tra độ bất định sigma kích hoạt trơn tru C^inf, không bị triệt tiêu gradient (Vanishing Gradient)."""
        backbone = MockViTBackbone(embed_dim=128)
        model = TrafficDecompositionNet(backbone=backbone, embed_dim=128, patch_size=16).to(self.device)

        # Kiểm tra forward qua toàn bộ mạng
        x = torch.rand(2, 3, 256, 448, device=self.device)
        out = model(x)
        sigma = out["sigma"]
        log_sigma = out["log_sigma"]

        # Kiểm tra khoảng giá trị lý thuyết: sigma in (0.01, 0.50)
        self.assertTrue((sigma >= 0.01).all() and (sigma <= 0.50).all())
        self.assertTrue((log_sigma >= -4.606).all() and (log_sigma <= -0.693).all())

        # Kiểm tra đạo hàm trực tiếp: Ở dải hoạt động rộng [-8.0, +8.0], đạo hàm của sigma theo raw_sigma không bị triệt tiêu
        raw_s = torch.tensor([-8.0, -4.0, 0.0, 4.0, 8.0], requires_grad=True)
        sig = 0.01 + 0.49 * torch.sigmoid(raw_s)
        loss = sig.sum()
        loss.backward()

        self.assertIsNotNone(raw_s.grad)
        self.assertFalse(torch.isnan(raw_s.grad).any())
        self.assertTrue((raw_s.grad > 0.0).all(), "Đạo hàm của sigma theo raw_s phải luôn dương nghiêm ngặt trong dải hoạt động!")

    def test_09_semantic_tam_guidance_fusion(self):
        """Kiểm tra Semantic TAM Guidance (pi) hợp nhất thành công với m_guide và kích hoạt vehicle_shield."""
        from direction2_new.models import compute_illumination_invariant_guide

        H, W = 256, 448
        # Giả lập tình huống xe máy trùng màu nhựa đường: origin và prior có RGB giống hệt nhau
        origin = torch.full((1, 3, H, W), 0.35, device=self.device)
        prior = torch.full((1, 3, H, W), 0.35, device=self.device)

        # 1. Khi không có pi: m_guide sẽ bằng 0 do sai khác pixel RGB = 0
        m_guide_no_pi, _, _ = compute_illumination_invariant_guide(origin, prior, pi=None)
        self.assertLess(m_guide_no_pi.max().item(), 0.10, "Khi không có pi và xe trùng màu đường, m_guide pixel bị mù!")

        # 2. Khi có Semantic TAM pi: DINOv3 ViT phát hiện patch xe tại góc (pi = 0.95)
        pi = torch.zeros(1, 16, 28, device=self.device)
        pi[:, 5:10, 10:15] = 0.95

        m_guide_with_pi, _, _ = compute_illumination_invariant_guide(origin, prior, pi=pi)
        # Tại vùng xe, m_guide_with_pi phải được kích hoạt mạnh mẽ >= 0.85
        self.assertGreaterEqual(m_guide_with_pi.max().item(), 0.85, "TAM pi phải cứu nguy và kích hoạt m_guide khi xe trùng màu đường!")

    def test_10_smooth_binarization_ramp(self):
        """Kiểm tra lịch trình làm mịn lực phạt nhị phân hóa (Smooth Annealed Binarization Ramp)."""
        from direction2_new.losses import NoiseAwareDecompositionLoss

        loss_fn = NoiseAwareDecompositionLoss(lambda_bin=0.05, warmup_bin_epoch=10)
        B, C, H, W = 2, 3, 256, 448
        preds = {
            "recon_origin": torch.rand(B, C, H, W, device=self.device),
            "pred_bg": torch.rand(B, C, H, W, device=self.device),
            "pred_fg": torch.rand(B, C, H, W, device=self.device),
            "alpha_mask": torch.full((B, 1, H, W), 0.5, device=self.device),  # Độ bất định cực đại 0.5
            "sigma": torch.full((B, 1, H, W), 0.1, device=self.device),
            "log_sigma": torch.full((B, 1, H, W), -2.3, device=self.device),
        }
        origin = torch.rand(B, C, H, W, device=self.device)
        prior_bg = torch.rand(B, C, H, W, device=self.device)

        # Epoch 9 (< 10): bin_ramp_w = 0.0, loss_bin = 0.0
        _, dict_ep9 = loss_fn(preds, origin, prior_bg, epoch=9, warmup_bin_epoch=10)
        self.assertEqual(dict_ep9["bin_ramp_w"], 0.0)
        self.assertEqual(dict_ep9["loss_bin"], 0.0)

        # Epoch 10 (bắt đầu kích hoạt): ramp_weight = 1/4 = 0.25
        _, dict_ep10 = loss_fn(preds, origin, prior_bg, epoch=10, warmup_bin_epoch=10)
        self.assertAlmostEqual(dict_ep10["bin_ramp_w"], 0.25, places=5)
        self.assertGreater(dict_ep10["loss_bin"], 0.0)

        # Epoch 11: ramp_weight = 2/4 = 0.50
        _, dict_ep11 = loss_fn(preds, origin, prior_bg, epoch=11, warmup_bin_epoch=10)
        self.assertAlmostEqual(dict_ep11["bin_ramp_w"], 0.50, places=5)

        # Epoch 13: ramp_weight = 4/4 = 1.0 (đạt 100% êm đềm, không sốc gradient)
        _, dict_ep13 = loss_fn(preds, origin, prior_bg, epoch=13, warmup_bin_epoch=10)
        self.assertAlmostEqual(dict_ep13["bin_ramp_w"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

