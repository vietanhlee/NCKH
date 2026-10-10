"""
=============================================================================
 Comprehensive Smoke Test & Verification Suite for DINO Traffic Suite (Core Directions)
 Kiểm thử toàn diện các hướng nghiên cứu cốt lõi (H1 New, H2 New, H1, H2) và tầng Common Utilities
 Chạy trên dữ liệu mô phỏng (Synthetic Dummy Data) để xác thực 100% không lỗi runtime
=============================================================================
"""

import os
import sys

# Đảm bảo mã hóa UTF-8 cho Windows Terminal / PowerShell và chống xung đột OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import shutil
import tempfile
import traceback
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thêm path
dino_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dino_dir not in sys.path:
    sys.path.insert(0, dino_dir)

print("=" * 80)
print(" [*] STARTING VERIFICATION SUITE - DINO CORE TRAFFIC DIRECTIONS")
print("=" * 80)

# Tạo thư mục tạm chứa dữ liệu giả lập chuẩn
temp_root = tempfile.mkdtemp(prefix="dino_test_")
print(f"📁 Thư mục tạm kiểm thử: {temp_root}")

try:
    # -------------------------------------------------------------
    # 0. TẠO CẤU TRÚC DỮ LIỆU GIẢ ĐỊNH (MOCK DATA)
    # -------------------------------------------------------------
    bg_dir = os.path.join(temp_root, "traffic_backgrounds")
    origin_dir = os.path.join(temp_root, "output")
    os.makedirs(os.path.join(bg_dir, "route_1"), exist_ok=True)
    os.makedirs(os.path.join(bg_dir, "route_2"), exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)

    # Tạo ảnh dummy background: 224x224
    dummy_bg_1_14 = Image.fromarray(np.full((224, 224, 3), 100, dtype=np.uint8))
    dummy_bg_1_14.save(os.path.join(bg_dir, "route_1", "background_slot_14h.jpg"))

    dummy_bg_1_15 = Image.fromarray(np.full((224, 224, 3), 110, dtype=np.uint8))
    dummy_bg_1_15.save(os.path.join(bg_dir, "route_1", "background_slot_15h.jpg"))

    dummy_bg_2_00 = Image.fromarray(np.full((224, 224, 3), 50, dtype=np.uint8))
    dummy_bg_2_00.save(os.path.join(bg_dir, "route_2", "background_slot_00h.jpg"))

    # Tạo ảnh origin:
    arr_orig_1 = np.full((224, 224, 3), 100, dtype=np.uint8)
    arr_orig_1[50:120, 50:120, :] = 255  # "Xe cộ" màu trắng nổi bật
    dummy_orig_1 = Image.fromarray(arr_orig_1)
    dummy_orig_1.save(os.path.join(origin_dir, "1_1755698811.jpg"))

    arr_orig_2 = np.full((224, 224, 3), 110, dtype=np.uint8)
    arr_orig_2[80:150, 100:180, :] = 220
    dummy_orig_2 = Image.fromarray(arr_orig_2)
    dummy_orig_2.save(os.path.join(origin_dir, "1_1755702400.jpg"))

    arr_orig_3 = np.full((224, 224, 3), 50, dtype=np.uint8)
    arr_orig_3[30:90, 40:110, :] = 200
    dummy_orig_3 = Image.fromarray(arr_orig_3)
    dummy_orig_3.save(os.path.join(origin_dir, "2_1755600000.jpg"))

    # CSV nhãn dummy
    csv_file = os.path.join(temp_root, "dummy_labels.csv")
    import pandas as pd
    df_dummy = pd.DataFrame([
        {"filename": "1_1755698811.jpg", "xe_may": 12, "o_to": 3, "tong": 15},
        {"filename": "1_1755702400.jpg", "xe_may": 25, "o_to": 5, "tong": 30},
        {"filename": "2_1755600000.jpg", "xe_may": 8, "o_to": 1, "tong": 9},
    ])
    df_dummy.to_csv(csv_file, index=False)
    print("✅ [Setup] Đã khởi tạo dữ liệu mô phỏng thành công.")

    # -------------------------------------------------------------
    # 1. TEST COMMON UTILITIES
    # -------------------------------------------------------------
    print("\n--- [TEST 1] Common Utilities ---")
    from common.matcher import TrafficPairMatcher
    matcher = TrafficPairMatcher(bg_dir=bg_dir, origin_dir=origin_dir, match_strategy="route_hourly")
    pairs = matcher.discover_pairs()
    assert len(pairs) >= 2, f"Kỳ vọng >= 2 cặp ảnh, thực tế: {len(pairs)}"
    print(f"   + TrafficPairMatcher phát hiện: {len(pairs)} cặp")

    from common.subtraction import BackgroundSubtractor
    subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=5)
    delta_norm, delta_raw = subtractor.compute_delta(np.array(dummy_orig_1), np.array(dummy_bg_1_14))
    assert delta_norm.shape == (224, 224), f"Shape delta sai: {delta_norm.shape}"
    bin_mask = subtractor.extract_binary_mask(delta_norm, method="otsu")
    assert bin_mask.shape == (224, 224) and bin_mask.dtype == np.uint8, "Mask sai định dạng"

    patch_weights = subtractor.compute_patch_mask_weights(delta_norm, patch_size=16)
    assert patch_weights.shape == (14, 14), f"Patch weights shape sai: {patch_weights.shape}"
    print(f"   + BackgroundSubtractor: Delta={delta_norm.shape}, PatchWeights={patch_weights.shape}")
    print("   ✅ [Common] Matcher & Subtractor pass hoàn hảo!")

    class MockPatchViT(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.embed_dim = embed_dim
            self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        def forward(self, x):
            feat = self.conv(x)  # (B, embed_dim, H_p, W_p)
            b, c, h, w = feat.shape
            tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            cls_t = tokens.mean(dim=1)
            return {
                "x_norm_clstoken": cls_t,
                "x_norm_patchtokens": tokens,
            }
        def get_intermediate_layers(self, x, n=1, return_class_token=True):
            feat = self.conv(x)
            b, c, h, w = feat.shape
            tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
            cls_t = tokens.mean(dim=1)
            return [(tokens, cls_t)]

    mock_vit = MockPatchViT(embed_dim=64)

    # -------------------------------------------------------------
    # 2. TEST DIRECTION 1: BG-GUIDED DINO
    # -------------------------------------------------------------
    print("\n--- [TEST 2] Direction 1: BG-Guided DINO SSL ---")
    from direction1_bg_guided_dino.dataset import BGGuidedDINODataset
    from direction1_bg_guided_dino.models import DINOHead, BGGuidedDINOModel
    from direction1_bg_guided_dino.losses import BGGuidedDINOLoss

    ds1 = BGGuidedDINODataset(
        bg_dir=bg_dir,
        origin_dir=origin_dir,
        match_strategy="route_hourly",
        patch_size=16,
        size_global=224,
        size_local=96,
        local_crops_number=2,
    )
    sample1 = ds1[0]
    crops = sample1["crops"]
    assert len(crops) == 4, f"Kỳ vọng 2 global + 2 local = 4 crops, thực tế: {len(crops)}"
    assert crops[0].shape == (3, 224, 224) and crops[2].shape == (3, 96, 96), "Shape crops sai"

    class TinyBackbone(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
            self.embed_dim = embed_dim
        def forward(self, x):
            x = self.conv(x)
            return x.mean(dim=(2, 3))

    mock_bb = TinyBackbone(embed_dim=64)
    dino_model = BGGuidedDINOModel(student_backbone=mock_bb, embed_dim=64, out_dim=512)

    batch_crops = [torch.stack([c, c]) for c in crops]
    student_out = dino_model.forward_student(batch_crops)
    teacher_out = dino_model.forward_teacher(batch_crops[:2])

    loss_fn1 = BGGuidedDINOLoss(out_dim=512, ncrops=4, nepochs=2)
    loss1 = loss_fn1(student_out, teacher_out, epoch=0)
    assert not torch.isnan(loss1) and loss1.item() > 0, "Loss DINO bất thường"
    loss1.backward()
    dino_model.update_teacher(0.99)
    print(f"   + DINO Multi-crop Loss: {loss1.item():.4f}")
    print("   ✅ [Direction 1] BG-Guided DINO pass hoàn hảo!")

    # -------------------------------------------------------------
    # 3. TEST DIRECTION 2: SCENE DECOMPOSITION
    # -------------------------------------------------------------
    print("\n--- [TEST 3] Direction 2: Scene Decomposition ---")
    from direction2_scene_decomposition.dataset import DecompositionDataset
    from direction2_scene_decomposition.models import TrafficDecompositionNet
    from direction2_scene_decomposition.losses import DecompositionLoss

    ds2 = DecompositionDataset(bg_dir=bg_dir, origin_dir=origin_dir, match_strategy="route_hourly", img_size=128, is_train=True)
    sample2 = ds2[0]
    assert sample2["origin"].shape == (3, 128, 128) and sample2["bg"].shape == (3, 128, 128)

    decomp_net = TrafficDecompositionNet(backbone=mock_vit, embed_dim=64, patch_size=16, freeze_backbone=True)
    batch_orig = torch.stack([sample2["origin"], sample2["origin"]])
    batch_bg = torch.stack([sample2["bg"], sample2["bg"]])
    preds2 = decomp_net(batch_orig)

    assert preds2["pred_bg"].shape == (2, 3, 128, 128)
    assert preds2["pred_fg"].shape == (2, 3, 128, 128)
    assert preds2["pred_mask"].shape == (2, 1, 128, 128)
    assert preds2["recon_origin"].shape == (2, 3, 128, 128)

    loss_fn2 = DecompositionLoss()
    l2, l2_dict = loss_fn2(preds2, {"origin": batch_orig, "bg": batch_bg})
    assert not torch.isnan(l2), "Loss Decomposition bị NaN"
    l2.backward()
    print(f"   + Decomposition Loss: {l2.item():.4f} (Recon: {l2_dict['loss_recon']:.4f})")
    print("   ✅ [Direction 2] Scene Decomposition pass hoàn hảo!")



    # -------------------------------------------------------------
    # 6. TEST MULTI-GPU SMART SAVE & LOAD CHECKPOINTING
    # -------------------------------------------------------------
    print("\n--- [TEST 6] Multi-GPU Smart Checkpointing Interoperability ---")
    from common.gpu_utils import save_checkpoint, load_checkpoint, clean_state_dict, smart_load_state_dict

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyNet()
    ckpt_save_test = os.path.join(temp_root, "test_smart_save.pth")

    class MockDataParallelWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
        def forward(self, x):
            return self.module(x)

    mock_dp_model = MockDataParallelWrapper(dummy_model)
    opt_dummy = torch.optim.AdamW(mock_dp_model.parameters(), lr=1e-3)
    sch_dummy = torch.optim.lr_scheduler.CosineAnnealingLR(opt_dummy, T_max=20)
    
    # Giả lập 5 bước cập nhật
    for _ in range(5):
        loss_dum = mock_dp_model(torch.randn(4, 10)).sum()
        opt_dummy.zero_grad()
        loss_dum.backward()
        opt_dummy.step()
        sch_dummy.step()

    save_checkpoint(
        save_path=ckpt_save_test,
        model=mock_dp_model,
        optimizer=opt_dummy,
        scheduler=sch_dummy,
        epoch=5,
        metrics={"test_metric": 0.99},
        verbose=False,
    )
    assert os.path.isfile(ckpt_save_test), "Lưu file checkpoint thất bại"

    saved_raw = torch.load(ckpt_save_test, map_location="cpu")
    for k in saved_raw["model_state"].keys():
        assert not k.startswith("module."), f"Lỗi: Checkpoint vẫn còn tiền tố 'module.': {k}"
    print("   + Checkpoint dọn sạch 100% tiền tố 'module.' -> PASSED!")

    target_clean_model = DummyNet()
    target_opt = torch.optim.AdamW(target_clean_model.parameters(), lr=1e-3)
    target_sch = torch.optim.lr_scheduler.CosineAnnealingLR(target_opt, T_max=20)

    loaded_ckpt = load_checkpoint(
        ckpt_save_test,
        model=target_clean_model,
        optimizer=target_opt,
        scheduler=target_sch,
        device="cpu",
        verbose=False,
    )
    assert loaded_ckpt["epoch"] == 5, f"Kỳ vọng epoch=5, thực tế: {loaded_ckpt.get('epoch')}"
    assert target_sch.last_epoch == 5, f"Kỳ vọng scheduler last_epoch=5, thực tế: {target_sch.last_epoch}"
    print("   + Khôi phục trọn vẹn Model, Optimizer, Scheduler và Epoch=5 -> PASSED!")

    # Kiểm tra Smart Inherit Checkpoint Args
    from common.gpu_utils import smart_inherit_checkpoint_args, resolve_checkpoint_path
    class DummyArgs:
        def __init__(self):
            self.lr = 1e-4
            self.batch_size = 16
            self.epochs = 10
            self.lambda_prior = 1.0

    d_args = DummyArgs()
    ckpt_args = {"lr": 3e-4, "batch_size": 32, "epochs": 20, "lambda_prior": 2.5}
    # Giả lập người dùng gõ --lr trên CLI, còn batch_size và lambda_prior không gõ
    smart_inherit_checkpoint_args(d_args, ckpt_args, sync_keys=["lr", "batch_size", "lambda_prior"], argv_list=["--lr", "1e-4"])
    assert d_args.lr == 1e-4, "Tham số CLI gõ tường minh phải được ưu tiên!"
    assert d_args.batch_size == 32, "Tham số không gõ trên CLI phải kế thừa từ checkpoint!"
    assert d_args.lambda_prior == 2.5, "lambda_prior phải kế thừa từ checkpoint!"
    print("   + Kế thừa Smart Hyperparameters thông minh -> PASSED!")
    print("   ✅ [Checkpointing] Multi-GPU Smart Save, Load & Full Resume pass hoàn hảo!")

    # -------------------------------------------------------------
    # 7. TEST COMMON ADVANCED UTILITIES: RELIABILITY, BDB, CORRUPT
    # -------------------------------------------------------------
    print("\n--- [TEST 7] Common Advanced Utilities: Reliability, BDB & FCS ---")
    from common.reliability import StaticRegionReliabilityEstimator, CameraAlignmentChecker
    from common.degradation import BackgroundDegradationBenchmark
    from common.corrupt import FrameCorruptionSuite

    # Reliability test
    rel_estimator = StaticRegionReliabilityEstimator(temporal_variance_threshold=0.01)
    dummy_bg_tensor = torch.rand(3, 128, 128)
    dummy_static_mask = torch.ones(128, 128)
    dummy_static_mask[40:90, 40:90] = 0.0  # Vùng đường ở giữa
    dummy_seq_tensor = torch.rand(4, 3, 128, 128)
    r_score = rel_estimator.compute_reliability_score(dummy_seq_tensor, dummy_bg_tensor, dummy_static_mask)
    assert 0.0 <= r_score <= 1.0, f"r_score ngoài khoảng [0, 1]: {r_score}"
    print(f"   + Estimated Static Reliability r_i: {r_score:.4f}")

    align_checker = CameraAlignmentChecker(shift_threshold_px=4.0)
    dy, dx, is_aligned = align_checker.estimate_camera_shift(dummy_bg_tensor, dummy_bg_tensor, dummy_static_mask)
    assert abs(dy) < 1.0 and abs(dx) < 1.0 and is_aligned
    print(f"   + Camera Shift: dy={dy:.2f}px, dx={dx:.2f}px, is_aligned={is_aligned}")

    # BDB test
    bdb = BackgroundDegradationBenchmark()
    corrupted_ghost = bdb.apply_degradation(dummy_bg_tensor, "ghost_injection", severity=3)
    assert corrupted_ghost.shape == dummy_bg_tensor.shape
    print("   + BDB Ghost Injection (Severity 3): OK")

    # FCS test
    fcs = FrameCorruptionSuite()
    corrupted_rain = fcs.apply_corruption(dummy_bg_tensor, "rain_streaks", severity=4)
    assert corrupted_rain.shape == dummy_bg_tensor.shape
    print("   + FCS Rain Streaks (Severity 4): OK")
    print("   ✅ [Common Advanced] Reliability, BDB & FCS pass hoàn hảo!")

    # -------------------------------------------------------------
    # 8. TEST DIRECTION 3: CONTEXT-AWARE WEAK SUPERVISION
    # -------------------------------------------------------------
    print("\n--- [TEST 8] Direction 3: Context-Aware Weak Supervision ---")
    from direction3_weak_supervision.context import TrafficContextClassifier
    from direction3_weak_supervision.label_model import ContextAwareMarkovLabelModel
    from direction3_weak_supervision.end_model import WeakSupervisionEndModel, SoftCrossEntropyLoss

    ctx_classifier = TrafficContextClassifier()
    ctx_id = ctx_classifier.get_context_id(hour=8, is_night=False, is_rain=False, is_major_artery=True, reliability_score=0.85)
    assert 0 <= ctx_id < 54
    print(f"   + Traffic Context ID: {ctx_id}")

    # Label model EM test
    T_b = 30
    sim_lfs = np.random.randint(0, 4, size=(T_b, 5))
    sim_ctxs = np.full(T_b, ctx_id, dtype=int)
    label_model = ContextAwareMarkovLabelModel(num_classes=4, num_lfs=5, num_contexts=54)
    label_model.fit_em([sim_lfs], [sim_ctxs], max_iters=5, verbose=False)
    soft_labels = label_model.predict_soft_labels(sim_lfs, sim_ctxs)
    assert soft_labels.shape == (T_b, 4)
    assert np.allclose(soft_labels.sum(axis=1), 1.0)
    print(f"   + Markov Label Model Soft Label Sample: {soft_labels[0].tolist()}")

    # End Model forward + loss test
    mock_vit_b = MockPatchViT(embed_dim=64)
    end_model = WeakSupervisionEndModel(backbone=mock_vit_b, embed_dim=64, hidden_dim=32, num_classes=4)
    dummy_seq_b = torch.randn(2, 4, 3, 128, 128)
    logits_b = end_model(dummy_seq_b)
    assert logits_b.shape == (2, 4)
    soft_loss_fn = SoftCrossEntropyLoss()
    target_soft_b = torch.tensor([[0.1, 0.7, 0.1, 0.1], [0.05, 0.1, 0.8, 0.05]])
    loss_b = soft_loss_fn(logits_b, target_soft_b)
    assert not torch.isnan(loss_b) and loss_b.item() >= 0
    print(f"   + End Model Logits: {logits_b[0].tolist()} | Soft CE Loss: {loss_b.item():.4f}")
    print("   ✅ [Direction 3] Weak Supervision pass hoàn hảo!")

    # -------------------------------------------------------------
    # 9. TEST DIRECTION 4: ANOMALY DETECTION & PERSISTENCE FILTERING
    # -------------------------------------------------------------
    print("\n--- [TEST 9] Direction 4: Anomaly Detection & Persistence Filtering ---")
    from direction4_anomaly_detection.features import DINOv3PatchFeatureExtractor
    from direction4_anomaly_detection.pooling import TemporalFeaturePooler
    from direction4_anomaly_detection.bank import NormalMemoryBank
    from direction4_anomaly_detection.score import AnomalyScorer
    from direction4_anomaly_detection.camera_fault import CameraFaultClassifier
    from direction4_anomaly_detection.events import PersistenceEventTracker

    extractor_c = DINOv3PatchFeatureExtractor(backbone=MockPatchViT(embed_dim=64), feature_dim=64, proj_dim=32, patch_size=16)
    p_tokens, h_p, w_p = extractor_c.extract_patch_tokens(torch.randn(2, 3, 128, 128))
    assert p_tokens.shape == (2, 64, 32)
    print(f"   + DINOv3 Extracted Patches: {p_tokens.shape}")

    pooler_c = TemporalFeaturePooler(window_size=3)
    pooled_c = pooler_c.update(p_tokens)
    assert pooled_c.shape == p_tokens.shape

    bank_c = NormalMemoryBank("cam_test", "morning", feature_dim=32)
    bank_c.fit_coreset(torch.randn(100, 32), subsampling_ratio=0.20)
    assert bank_c.bank is not None and len(bank_c.bank) >= 20

    scorer_c = AnomalyScorer(k_nearest=1, top_k_ratio=0.10)
    p_scores_c = scorer_c.compute_patch_scores(pooled_c, bank_c.bank)
    assert p_scores_c.shape == (2, 64)
    road_mask_patch = torch.ones(2, 64)
    road_mask_patch[:, 32:] = 0.0
    r_score_c, s_score_c = scorer_c.aggregate_frame_score(p_scores_c, road_mask_patch)
    assert r_score_c.shape == (2,) and s_score_c.shape == (2,)

    cam_clf = CameraFaultClassifier()
    diag = cam_clf.classify(r_score_c[0].item(), s_score_c[0].item(), 0.5, 0.5)
    assert "event_type" in diag
    print(f"   + Anomaly Diagnosis: {diag['event_type']} (Road: {r_score_c[0].item():.3f}, Static: {s_score_c[0].item():.3f})")

    tracker_c = PersistenceEventTracker(min_consecutive_windows=2)
    tracker_c.calibrate_threshold(np.array([0.1, 0.15, 0.2, 0.25, 0.3]))
    alert_1 = tracker_c.update(step_idx=0, score=0.9)
    alert_2 = tracker_c.update(step_idx=1, score=0.95)
    assert alert_2 is not None and alert_2["status"] == "CONFIRMED"
    print("   + Persistence Tracker Triggered Alert: CONFIRMED")
    print("   ✅ [Direction 4] Anomaly Detection pass hoàn hảo!")
    print("\n--- [TEST 12] Direction 1 New: Vehicle-Centric SSL Pretraining (TAM + AGM + SRS) ---")
    from direction1_new.tam import PositionStats, GMMCalibrator
    from direction1_new.masking import agm_sample
    from direction1_new.srs import static_region_swap

    pos_stats_g = PositionStats(num_cams=2, num_patches=64, num_states=4, feat_dim=32)
    cids_g = torch.tensor([0, 1])
    u_dummy = F.normalize(torch.randn(2, 64, 32), p=2, dim=-1)
    pos_stats_g.update(cids_g, u_dummy)
    a_g, valid_g = pos_stats_g.atypicality(cids_g, u_dummy)
    assert a_g.shape == (2, 64)

    calib_g = GMMCalibrator()
    calib_g.push(a_g, valid_g)
    pi_g = calib_g.posterior(a_g)
    assert pi_g.shape == (2, 64)

    agm_mask_g = agm_sample(pi_g[0], valid_g[0], ratio=0.35, phi=0.5, q_max=0.6)
    assert agm_mask_g.shape == (64,)

    x1_g = torch.zeros(3, 128, 128)
    x2_g = torch.ones(3, 128, 128)
    pi1_g = torch.zeros(8, 8)
    pi2_g = torch.zeros(8, 8)
    x_srs_g, _ = static_region_swap(x1_g, x2_g, pi1_g, pi2_g, ratio=0.5, feather=2)
    assert x_srs_g.shape == (3, 128, 128)
    print("   ✅ [Direction 1 New] TAM, AGM, SRS pass hoàn hảo!")

    # -------------------------------------------------------------
    # 13. TEST DIRECTION 2 NEW: PRIOR-FREE SCENE DECOMPOSITION
    # -------------------------------------------------------------
    print("\n--- [TEST 13] Direction 2 New: Prior-Free Scene Decomposition (SceneBasis + TrafficDecompositionNet) ---")
    from direction2_new.scene_fit import SceneBasis, robust_weights
    from direction2_new.solve_ell import solve_ell
    from direction2_new.losses import SceneDecompositionLossV2

    basis = SceneBasis(num_cams=1, num_frames_per_cam=2, J=4, device=torch.device("cpu"))
    b_recon = basis.get_background(cam_idx=0, frame_indices=torch.tensor([0, 1]))
    assert b_recon.shape == (2, 3, 256, 448)

    # Test solve_ell
    frame_t = torch.randn(3, 256, 448)
    E0_t = basis.E0[0]
    Ej_t = basis.Ej[0]
    w_t = torch.ones(256, 448)
    ell_opt = solve_ell(frame_t, E0_t, Ej_t, w_t, iters=5)
    assert ell_opt.shape == (4,)

    loss_v2 = SceneDecompositionLossV2()
    pred_dict_2 = {
        "recon_origin": torch.rand(2, 3, 256, 448),
        "pred_bg": torch.rand(2, 3, 256, 448),
        "pred_fg": torch.rand(2, 3, 256, 448),
        "alpha_mask": torch.rand(2, 1, 256, 448),
        "sigma": torch.full((2, 1, 256, 448), 0.1),
        "log_sigma": torch.full((2, 1, 256, 448), -2.3),
        "pred_ell": torch.zeros(2, 4),
    }
    origin_2 = torch.rand(2, 3, 256, 448)
    bg_pseudo_2 = torch.rand(2, 3, 256, 448)
    mask_pseudo_2 = torch.zeros(2, 1, 256, 448)
    conf_2 = torch.ones(2, 1, 256, 448)
    ell_gt_2 = torch.zeros(2, 4)
    pi_2 = torch.zeros(2, 16, 28)

    total_loss_2, loss_dict_2 = loss_v2(
        preds=pred_dict_2,
        origin=origin_2,
        bg_pseudo=bg_pseudo_2,
        mask_pseudo=mask_pseudo_2,
        conf=conf_2,
        ell_gt=ell_gt_2,
        pi=pi_2,
    )
    assert not torch.isnan(total_loss_2)
    print("   ✅ [Direction 2 New] SceneBasis, solve_ell, LossV2 pass hoàn hảo!")

    print("\n" + "=" * 80)
    print(" 🎉 TOÀN BỘ CÁC HƯỚNG TRỌNG TÂM (H1 NEW, H2 NEW, H1 CŨ, H2 CŨ, COMMON) ĐỀU VƯỢT QUA TEST 100%!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ [LỖI RUNTIME] Kiểm thử thất bại: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)
