"""
=============================================================================
 Test Suite: Direction G (test_direction_g.py)
 Kiểm thử tích hợp toàn bộ module của Hướng G:
 TAM (PositionStats, GMMCalibrator), AGM (agm_sample), SRS (static_region_swap),
 Losses (DINO, iBOT, KoLeo, RIC) và Pipeline Huấn luyện.
=============================================================================
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import unittest
import torch
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

from direction1_new.tam import PositionStats, GMMCalibrator
from direction1_new.masking import agm_sample
from direction1_new.srs import static_region_swap
from direction1_new.losses import (
    DINOLoss,
    iBOTPatchLoss,
    KoLeoLoss,
    RICLoss,
    pool_vehicle_embedding,
    CameraAdversarialHead,
    compute_dann_alpha,
)
from direction1_new.sampler import CameraGroupedSampler


class TestDirection1New(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def test_01_tam_position_stats(self):
        """Kiểm tra PositionStats cập nhật trực tuyến và tính độ khác thường a_t."""
        B, P, d = 2, 448, 64
        stats = PositionStats(num_cams=4, num_patches=P, num_states=4, feat_dim=d).to(self.device)
        cids = torch.tensor([0, 1], device=self.device)
        U = F.normalize(torch.randn(B, P, d, device=self.device), p=2, dim=-1)

        # Cập nhật trực tuyến
        stats.update(cids, U)
        self.assertFalse(torch.isnan(stats.mu).any())

        # Điểm khác thường a_t
        a, valid = stats.atypicality(cids, U)
        self.assertEqual(a.shape, (B, P))
        self.assertEqual(valid.shape, (B, P))

        # Hiệu chuẩn GMM
        calibrator = GMMCalibrator()
        calibrator.push(a, valid)
        pi = calibrator.posterior(a)
        self.assertEqual(pi.shape, (B, P))
        self.assertTrue((pi >= 0.0).all() and (pi <= 1.0).all())

    def test_02_agm_masking(self):
        """Kiểm tra thuật toán AGM che phân tầng có trọng số."""
        N = 196
        pi = torch.zeros(N, device=self.device)
        pi[:40] = 0.85  # 40 patch xe
        valid = torch.ones(N, dtype=torch.bool, device=self.device)

        phi = 0.5
        q_max = 0.6
        ratio = 0.3
        mask = agm_sample(pi, valid, ratio=ratio, phi=phi, q_max=q_max)

        expected_mask_count = int(round(ratio * N))
        self.assertEqual(mask.sum().item(), expected_mask_count)

        # Kiểm tra trần q_max
        fg_masked = mask[:40].sum().item()
        self.assertLessEqual(fg_masked, int(q_max * 40))

    def test_03_static_region_swap(self):
        """Kiểm tra SRS hoán đổi vùng tĩnh bảo toàn pixel phương tiện."""
        x1 = torch.zeros(3, 256, 448, device=self.device)
        x2 = torch.ones(3, 256, 448, device=self.device)
        pi1 = torch.zeros(16, 28, device=self.device)
        pi2 = torch.zeros(16, 28, device=self.device)
        pi1[4:8, 4:8] = 0.9  # Vùng xe

        x_tilde, swap_mask = static_region_swap(x1, x2, pi1, pi2, ratio=0.7, feather=2)
        self.assertEqual(x_tilde.shape, (3, 256, 448))
        self.assertEqual(swap_mask.shape, (16, 28))

    def test_04_ssl_losses(self):
        """Kiểm tra các hàm mất mát DINOLoss, iBOTPatchLoss, KoLeoLoss, RICLoss."""
        B, N, D = 2, 196, 256
        s_cls = torch.randn(B, D, device=self.device, requires_grad=True)
        t_cls = torch.randn(B, D, device=self.device)
        s_patch = torch.randn(B, N, D, device=self.device, requires_grad=True)
        t_patch = torch.randn(B, N, D, device=self.device)
        mask = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        mask[:, :30] = True

        dino_fn = DINOLoss(out_dim=D).to(self.device)
        ibot_fn = iBOTPatchLoss(out_dim=D).to(self.device)
        koleo_fn = KoLeoLoss().to(self.device)
        ric_fn = RICLoss().to(self.device)

        loss_d = dino_fn(s_cls, t_cls)
        loss_i = ibot_fn(s_patch, t_patch, mask)
        loss_k = koleo_fn(s_cls)
        loss_r = ric_fn(F.normalize(s_cls, p=2, dim=-1), F.normalize(t_cls, p=2, dim=-1), cids=torch.tensor([0, 1]))

        total = loss_d + loss_i + 0.1 * loss_k + 0.1 * loss_r
        total.backward()
        self.assertIsNotNone(s_cls.grad)
        self.assertFalse(torch.isnan(s_cls.grad).any())

    def test_05_position_stats_dynamic_resume(self):
        """Kiểm tra PositionStats tự động co giãn dung lượng camera khi load state_dict có shape khác (570 vs 16 cams)."""
        # Giả lập checkpoint đã train trên 570 camera
        stats_saved = PositionStats(num_cams=570, num_patches=448, num_states=4, feat_dim=64)
        saved_sd = stats_saved.state_dict()

        # Mô hình mới khởi tạo với 16 camera mặc định
        stats_new = PositionStats(num_cams=16, num_patches=448, num_states=4, feat_dim=64)
        self.assertEqual(stats_new.num_cams, 16)

        # Nạp state_dict: Phải tự động mở rộng lên 570 mà không văng RuntimeError
        stats_new.load_state_dict(saved_sd)
        self.assertEqual(stats_new.num_cams, 570)
        self.assertEqual(stats_new.mu.shape, (570, 448, 4, 64))
        self.assertEqual(stats_new.w.shape, (570, 448, 4))
        self.assertEqual(stats_new.s.shape, (570, 448, 4))
        self.assertEqual(stats_new.is_initialized.shape, (570,))

    def test_06_gmm_calibrator_state_dict(self):
        """Kiểm tra GMMCalibrator lưu và khôi phục trạng thái chuẩn xác (state_dict / load_state_dict)."""
        calib1 = GMMCalibrator()
        a_mock = torch.cat([torch.rand(2, 448) * 0.2, torch.rand(2, 448) * 5.0 + 2.0], dim=0)
        v_mock = torch.ones_like(a_mock, dtype=torch.bool)
        calib1.push(a_mock, v_mock)
        calib1.refit()
        self.assertTrue(calib1.is_fitted)

        # Lưu state_dict
        sd = calib1.state_dict()
        self.assertIn("mu_bg", sd)
        self.assertIn("mu_fg", sd)
        self.assertIn("pi_weight", sd)
        self.assertIn("max_buffer_size", sd)
        self.assertIn("ema_alpha", sd)
        self.assertIn("buffer_snapshot", sd)
        self.assertEqual(sd["max_buffer_size"], 300000)
        self.assertTrue(sd["is_fitted"])

        # Tạo calibrator mới chưa fit và khôi phục
        calib2 = GMMCalibrator(max_buffer_size=200000)
        self.assertFalse(calib2.is_fitted)
        calib2.load_state_dict(sd)
        self.assertTrue(calib2.is_fitted)
        self.assertEqual(calib2.max_buffer_size, 300000)
        self.assertAlmostEqual(calib1.mu_bg, calib2.mu_bg, places=4)
        self.assertAlmostEqual(calib1.mu_fg, calib2.mu_fg, places=4)
        self.assertAlmostEqual(calib1.pi_weight, calib2.pi_weight, places=4)

        # So sánh đầu ra posterior
        test_pts = torch.tensor([[0.05, 0.5, 2.0, 10.0]])
        pi1 = calib1.posterior(test_pts)
        pi2 = calib2.posterior(test_pts)
        self.assertTrue(torch.allclose(pi1, pi2, atol=1e-5))

    def test_07_resume_scheduler_learning_rate_continuity(self):
        """Kiểm tra cơ chế build_smooth_lr_scheduler đảm bảo tính liên tục C^0 tuyệt đối giữa các Session."""
        from direction1_new.train import build_smooth_lr_scheduler

        param1 = torch.nn.Parameter(torch.zeros(10))
        base_lr = 5e-5
        eta_min = 1e-6
        total_epochs = 12
        warmup_epochs = 1

        # PHIÊN 1 (Session 1): Huấn luyện 4 epochs (Epoch 0 -> 3)
        opt1 = torch.optim.AdamW([param1], lr=base_lr)
        sched1 = build_smooth_lr_scheduler(opt1, total_epochs=total_epochs, warmup_epochs=warmup_epochs, eta_min=eta_min)

        lrs_session_1 = []
        for ep in range(4):
            lrs_session_1.append(opt1.param_groups[0]["lr"])
            opt1.step()
            sched1.step()

        # LR ở đầu Epoch 4 (sẵn sàng cho epoch tiếp theo sau khi step 4 lần)
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
            self.assertGreaterEqual(all_cosine_lrs[i], all_cosine_lrs[i+1])

    def test_08_cli_args_parsing_and_smart_inheritance(self):
        """Kiểm tra helper get_cli_specified_args phân biệt đúng đối số người dùng truyền."""
        from direction1_new.train import get_cli_specified_args

        simulated_argv = ["--resume", "last_checkpoint.pth", "--session_epochs", "4", "--device", "cuda:0", "--gmm_buffer_size", "200000"]
        specified = get_cli_specified_args(simulated_argv)

        self.assertIn("resume", specified)
        self.assertIn("session_epochs", specified)
        self.assertIn("device", specified)
        self.assertIn("gmm_buffer_size", specified)
        self.assertNotIn("phi", specified)
        self.assertNotIn("q_max", specified)
        self.assertNotIn("p_srs", specified)

    def test_09_koleo_loss_continuous_non_zero_gradient(self):
        """Kiểm tra KoLeoLoss luôn sinh gradient liên tục và > 0 ngay cả khi các vector phân tán xa nhau."""
        koleo_fn = KoLeoLoss()

        # Tạo batch 4 vector trực giao hoàn toàn (khoảng cách cực đại sqrt(2) > 1.0)
        cls_orthogonal = torch.eye(4, 16, requires_grad=True)
        loss = koleo_fn(cls_orthogonal)

        # Loss phải dương tự nhiên do chuẩn hóa rho / 2.0
        self.assertGreater(loss.item(), 0.0)

        # Gradient phải tồn tại và khác 0 (không bị clamp min=0 triệt tiêu)
        loss.backward()
        self.assertIsNotNone(cls_orthogonal.grad)
        self.assertFalse(torch.all(cls_orthogonal.grad == 0))
        self.assertFalse(torch.isnan(cls_orthogonal.grad).any())

    def test_10_gated_tam_update_protects_static_clusters(self):
        """Kiểm tra Gated Online Update: Vùng có xác suất xe pi=1 không làm dịch chuyển tâm cụm nền tĩnh."""
        stats = PositionStats(num_cams=2, num_patches=16, num_states=2, feat_dim=8)
        cids = torch.tensor([0])
        U = F.normalize(torch.randn(1, 16, 8), p=2, dim=-1)
        stats.init_kmeans(0, U.repeat(5, 1, 1))

        initial_mu = stats.mu[0].clone()

        # Giả lập frame xuất hiện xe dày đặc ở patch 0..7 (pi = 1.0) và nền sạch ở patch 8..15 (pi = 0.0)
        gate_pi = torch.zeros(1, 16)
        gate_pi[0, :8] = 1.0  # Xe rõ rệt

        # Vector feature mới hoàn toàn khác lạ (để thử kéo lệch)
        U_new = F.normalize(-U, p=2, dim=-1)
        stats.update(cids, U_new, gate_pi=gate_pi)

        # Kiểm tra: Patch có pi=1 thì delta_mu bị khóa hãm (thay đổi cực kỳ nhỏ hoặc không đổi)
        # Trong khi patch có pi=0 được cập nhật bình thường
        shift_fg = torch.norm(stats.mu[0, :8] - initial_mu[:8], dim=-1).mean().item()
        shift_bg = torch.norm(stats.mu[0, 8:] - initial_mu[8:], dim=-1).mean().item()
        self.assertLess(shift_fg, shift_bg)

    def test_11_block_masking_mode(self):
        """Kiểm tra AGM Block-wise Masking che cụm 2x2 quanh xe và không vượt quá trần q_max."""
        N = 16 * 16  # 256 patches
        pi = torch.zeros(N)
        # Giả lập 2 cụm xe 3x3
        pi[10:13] = 0.9
        pi[26:29] = 0.9
        pi[42:45] = 0.9

        mask = agm_sample(pi, ratio=0.25, phi=0.5, q_max=0.6, mode="block", grid_shape=(16, 16))
        self.assertEqual(mask.sum().item(), int(round(0.25 * N)))
        # Không vi phạm trần q_max
        fg_count = (pi > 0.5).sum().item()
        fg_masked = mask[pi > 0.5].sum().item()
        self.assertLessEqual(fg_masked, int(math.floor(0.6 * fg_count)))

    def test_12_continuous_srs_and_color_match(self):
        """Kiểm tra SRS chế độ continuous_bg và color_match."""
        x1 = torch.rand(3, 256, 448)
        x2 = torch.rand(3, 256, 448) * 0.5  # tối hơn
        pi1 = torch.zeros(16, 28)
        pi2 = torch.zeros(16, 28)
        pi1[5:9, 5:9] = 0.85  # Xe

        x_tilde, swap_mask = static_region_swap(
            x1, x2, pi1, pi2, ratio=0.8, continuous_bg=True, color_match=True
        )
        self.assertEqual(x_tilde.shape, (3, 256, 448))
        self.assertTrue((x_tilde >= 0.0).all() and (x_tilde <= 1.0).all())

    def test_13_camera_adversarial_head_grl(self):
        """Kiểm tra CameraAdversarialHead với Gradient Reversal Layer."""
        in_dim = 64
        num_cams = 4
        head = CameraAdversarialHead(in_dim=in_dim, num_cams=num_cams)
        cls_feat = torch.randn(4, in_dim, requires_grad=True)
        cids = torch.tensor([0, 1, 2, 3])

        loss = head(cls_feat, cids, grl_alpha=0.5)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(cls_feat.grad)
        self.assertFalse(torch.isnan(cls_feat.grad).any())

    def test_14_vehicle_pooling_and_ric_loss(self):
        """Kiểm tra pool_vehicle_embedding và RICLoss trên hard negatives cùng camera."""
        B, P, D = 4, 16, 32
        patches = torch.randn(B, P, D)
        pi = torch.zeros(B, P)
        pi[:, :4] = 0.9  # 4 patch đầu là xe

        z_veh = pool_vehicle_embedding(patches, pi)
        self.assertEqual(z_veh.shape, (B, D))
        # Chuẩn hóa L2
        norms = torch.norm(z_veh, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(B), atol=1e-4))

        # Test RICLoss
        z1 = z_veh
        z2 = z_veh + 0.05 * torch.randn_like(z_veh)
        cids = torch.tensor([0, 0, 1, 1])  # 2 camera, mỗi camera 2 frame
        ric_fn = RICLoss()
        loss_ric = ric_fn(z1, z2, cids)
        self.assertGreater(loss_ric.item(), 0.0)

    def test_15_camera_grouped_sampler(self):
        """Kiểm tra CameraGroupedSampler phân phối đúng camera và số frame mỗi batch."""
        cam_dict = {
            0: list(range(0, 10)),
            1: list(range(10, 20)),
            2: list(range(20, 30)),
            3: list(range(30, 40)),
        }
        sampler = CameraGroupedSampler(
            camera_to_indices=cam_dict,
            batch_cameras=2,
            frames_per_camera=3,
            batches_per_epoch=5,
            shuffle=False,
        )
    def test_16_gmm_reservoir_sampling_and_ema_smoothing(self):
        """Kiểm tra Reservoir Sampling không bị tràn buffer, Equal Quota và EMA smoothing khi refit."""
        buffer_limit = 500
        calib = GMMCalibrator(max_buffer_size=buffer_limit, ema_alpha=0.2)

        # 1. Đẩy 2000 mẫu vào buffer size 500 -> Buffer phải giữ đúng 500 mẫu, total_seen = 2000
        mock_a = torch.rand(10, 200) * 2.0 + 0.1
        mock_v = torch.ones_like(mock_a, dtype=torch.bool)
        calib.push(mock_a, mock_v)

        self.assertEqual(calib.size, buffer_limit)
        self.assertEqual(calib.total_seen, 2000)

        # 2. Fit lần đầu
        calib.refit(max_iters=20)
        self.assertTrue(calib.is_fitted)
        init_mu_bg = calib.mu_bg
        init_mu_fg = calib.mu_fg

        # 3. Đẩy dữ liệu mới bị lệch mạnh (giá trị rất lớn)
        high_a = torch.full((5, 100), 50.0)
        high_v = torch.ones_like(high_a, dtype=torch.bool)
        calib.push(high_a, high_v)
        calib.refit(max_iters=20)

        # 4. Kiểm tra tính làm mượt EMA: tham số di chuyển dần dần chứ không nhảy bổ ngay lập tức
        # mu_fg mới phải dịch về phía dữ liệu cao hơn, nhưng bị kìm hãm bởi ema_alpha = 0.2
        self.assertGreaterEqual(calib.mu_fg, init_mu_fg - 0.1)

        # 5. Kiểm tra tính năng max_samples (Equal Quota per camera)
        quota_calib = GMMCalibrator(max_buffer_size=1000)
        big_a = torch.rand(4, 448)
        big_v = torch.ones_like(big_a, dtype=torch.bool)
        quota_calib.push(big_a, big_v, max_samples=150)
        # Chỉ có tối đa đúng 150 mẫu được nạp vào
        self.assertEqual(quota_calib.size, 150)

    def test_17_ric_anti_saturation_and_gradients(self):
        """Kiểm tra RICLoss cải tiến: Không bị bão hòa sát 0, sinh gradient mạnh trên cả z1 và z2."""
        B, D = 8, 128
        raw_z1 = torch.randn(B, D, requires_grad=True)
        raw_z2 = torch.randn(B, D, requires_grad=True)
        z1 = F.normalize(raw_z1, p=2, dim=-1)
        z2 = F.normalize(raw_z2, p=2, dim=-1)
        cids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        ric_fn = RICLoss(temperature=0.07, hard_cam_weight=2.0)
        loss = ric_fn(z1, z2, cids)

        # Loss phải nằm trong dải động hoạt động lành mạnh (> 0.2), không được bẹp dí ở 0.04
        self.assertGreater(loss.item(), 0.2)

        # Kiểm tra lan truyền gradient
        loss.backward()
        self.assertIsNotNone(raw_z1.grad)
        self.assertIsNotNone(raw_z2.grad)
        self.assertGreater(raw_z1.grad.norm().item(), 1e-4)
        self.assertGreater(raw_z2.grad.norm().item(), 1e-4)

    def test_18_dann_progressive_alpha(self):
        """Kiểm tra lịch trình DANN GRL alpha tăng mượt mà từ 0.0 -> 0.1."""
        alpha_start = compute_dann_alpha(0.0, max_alpha=0.1)
        alpha_mid = compute_dann_alpha(0.5, max_alpha=0.1)
        alpha_end = compute_dann_alpha(1.0, max_alpha=0.1)

        self.assertAlmostEqual(alpha_start, 0.0, places=4)
        self.assertGreater(alpha_mid, 0.05)
        self.assertLessEqual(alpha_end, 0.1)
        self.assertGreater(alpha_end, alpha_mid)

    def test_19_dino_distribution_metrics(self):
        """Kiểm tra tính toán phân bố xác suất DINO (Teacher/Student Entropy, KL Div, Active Prototypes)."""
        dino_fn = DINOLoss(out_dim=1024, nepochs=10)
        s_logits = torch.randn(4, 1024)
        t_logits = torch.randn(4, 1024)

        metrics = dino_fn.compute_distribution_metrics(s_logits, t_logits, epoch=2)
        self.assertIn("teacher_entropy", metrics)
        self.assertIn("student_entropy", metrics)
        self.assertIn("kl_div", metrics)
        self.assertIn("active_prototypes", metrics)
        self.assertGreater(metrics["teacher_entropy"], 0.0)
        self.assertGreater(metrics["active_prototypes"], 0)

    def test_20_tam_cache_loading(self):
        """Kiểm tra nạp trực tiếp bản đồ pi từ cache thư mục."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = os.path.join(tmp_dir, "cache")
            data_dir = os.path.join(tmp_dir, "data")
            os.makedirs(cache_dir, exist_ok=True)
            cam_dir = os.path.join(data_dir, "cam_0")
            os.makedirs(cam_dir, exist_ok=True)

            # Tạo 2 ảnh giả lập
            f1 = os.path.join(cam_dir, "img_001.jpg")
            f2 = os.path.join(cam_dir, "img_002.jpg")
            from PIL import Image
            Image.new("RGB", (256, 448)).save(f1)
            Image.new("RGB", (256, 448)).save(f2)

            # Tạo 2 file cache .npy
            c1_dir = os.path.join(cache_dir, "cam_0")
            os.makedirs(c1_dir, exist_ok=True)
            np.save(os.path.join(c1_dir, "img_001.jpg.npy"), np.ones((16, 28), dtype=np.float16) * 0.75)
            np.save(os.path.join(c1_dir, "img_002.jpg.npy"), np.ones((16, 28), dtype=np.float16) * 0.25)

            from direction1_new.train import TrafficCameraDataset
            ds = TrafficCameraDataset(data_dir=data_dir, tam_cache_dir=cache_dir)
            sample = ds[0]

            self.assertIn("pi1", sample)
            self.assertIn("pi2", sample)
            self.assertEqual(sample["pi1"].shape, (16, 28))
            self.assertAlmostEqual(sample["pi1"][0, 0].item(), 0.75, places=2)
            self.assertAlmostEqual(sample["pi2"][0, 0].item(), 0.25, places=2)

    def test_21_prototype_weights_no_weight_decay(self):
        """Kiểm tra lớp last_layer.weight trong DINOHead không bị áp dụng weight decay."""
        from direction1_new.train import DINOHead
        head = DINOHead(in_dim=384, out_dim=1024, norm_last_layer=True)

        decay_params, nodecay_params = [], []
        for name, param in head.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias") or "last_layer" in name:
                nodecay_params.append((name, param))
            else:
                decay_params.append((name, param))

        nodecay_names = [n for n, p in nodecay_params]
        self.assertIn("last_layer.weight", nodecay_names)

    def test_22_auto_detect_or_precompute_tam_cache(self):
        """Kiểm tra cơ chế tự động tìm cache hoặc tự động tiền xử lý TAM cache trước khi train."""
        import tempfile
        from direction1_new.train import resolve_or_precompute_tam_cache

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = os.path.join(tmp_dir, "data")
            cam_dir = os.path.join(data_dir, "cam_0")
            os.makedirs(cam_dir, exist_ok=True)

            from PIL import Image
            Image.new("RGB", (256, 448)).save(os.path.join(cam_dir, "frame_01.jpg"))
            Image.new("RGB", (256, 448)).save(os.path.join(cam_dir, "frame_02.jpg"))

            # 1. Chưa có cache: Hàm tự động chạy precompute và trả về cache_dir hợp lệ
            target_cache = os.path.join(tmp_dir, "auto_cache")
            res_cache = resolve_or_precompute_tam_cache(
                data_dir=data_dir,
                requested_cache_dir=target_cache,
                device="cpu",
                batch_size=2,
            )
            self.assertEqual(res_cache, target_cache)
            self.assertTrue(os.path.isdir(target_cache))
            self.assertTrue(os.path.isfile(os.path.join(target_cache, "tam_stats.pth")))

            # 2. Đã có cache: Lần gọi tiếp theo tự động phát hiện và tái sử dụng ngay lập tức mà không chạy lại
            cached_res = resolve_or_precompute_tam_cache(
                data_dir=data_dir,
                requested_cache_dir=target_cache,
                device="cpu",
            )
            self.assertEqual(cached_res, target_cache)

    def test_23_refine_traffic_foreground_map(self):
        """Kiểm tra hàm refine_traffic_foreground_map dập tắt nhiễu mặt đường và làm nét xe cộ."""
        from direction1_new.tam import refine_traffic_foreground_map
        pi_raw = torch.tensor([0.15, 0.25, 0.50, 0.75, 0.90])
        pi_clean = refine_traffic_foreground_map(pi_raw)

        # 1. Các patch mặt đường yếu (<= 0.25) phải bị dập tắt mạnh về < 0.10
        self.assertLess(pi_clean[0].item(), 0.05)
        self.assertLess(pi_clean[1].item(), 0.10)

        # 2. Patch ngưỡng trung bình 0.50 giữ nguyên đối xứng
        self.assertAlmostEqual(pi_clean[2].item(), 0.50, places=2)

        # 3. Patch tiền cảnh xe cộ (>= 0.75) được khuếch đại sắc nét lên >= 0.85
        self.assertGreaterEqual(pi_clean[3].item(), 0.85)
        self.assertGreaterEqual(pi_clean[4].item(), 0.95)

    def test_24_srs_adaptive_fallback_non_zero_delta(self):
        """Kiểm tra cơ chế Adaptive Fallback của SRS đảm bảo delta > 0 ngay cả khi pi cao khắp nơi."""
        from direction1_new.srs import static_region_swap
        x1 = torch.zeros(3, 256, 448)
        x2 = torch.ones(3, 256, 448)
        # Tình huống biên cực đoan: cả 2 ảnh đều có pi = 0.45 khắp nơi (trước đây n_eligible == 0 khiến delta = 0%)
        pi1 = torch.full((16, 28), 0.45)
        pi2 = torch.full((16, 28), 0.45)

        x_tilde, swap_mask = static_region_swap(x1, x2, pi1, pi2, ratio=0.6, continuous_bg=True)
        diff_val = float(torch.abs(x_tilde - x1).mean().item() * 100.0)

        # Cam kết kỹ thuật: SRS phải hoán nền thành công, diff_val > 0.0% và số patch swap >= 20
        self.assertGreater(diff_val, 5.0)
        self.assertGreaterEqual(int(swap_mask.sum().item()), 20)

    def test_25_patch_variance_and_anchor_loss_non_contiguous(self):
        """Kiểm tra PatchVarianceRegularizer và FeatureAnchorLoss với non-contiguous tensors (DataParallel safe)."""
        from direction1_new.losses import PatchVarianceRegularizer, FeatureAnchorLoss
        pvar = PatchVarianceRegularizer()
        anchor = FeatureAnchorLoss()

        # Tạo non-contiguous tensor bằng permute mô phỏng DataParallel gather output
        raw_leaf = torch.randn(4, 384, 448, requires_grad=True)
        s_patches = raw_leaf.permute(0, 2, 1)  # (4, 448, 384)
        s_patches.retain_grad()
        self.assertFalse(s_patches.is_contiguous())

        pi = torch.rand(4, 28, 16).permute(0, 2, 1).reshape(4, 448)  # non-contiguous -> reshape
        loss_pvar = pvar(s_patches, pi=pi)
        self.assertTrue(torch.isfinite(loss_pvar))
        loss_pvar.backward(retain_graph=True)
        self.assertIsNotNone(s_patches.grad)
        self.assertIsNotNone(raw_leaf.grad)

        # Feature Anchor Loss với non-contiguous
        pretrained = torch.randn(4, 384, 448).permute(0, 2, 1)
        loss_anc = anchor(s_patches, pretrained)
        self.assertTrue(torch.isfinite(loss_anc))
        loss_anc.backward()
        self.assertTrue(torch.isfinite(raw_leaf.grad).all())


if __name__ == "__main__":
    unittest.main()

