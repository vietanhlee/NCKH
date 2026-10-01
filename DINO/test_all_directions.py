"""
=============================================================================
 Comprehensive Smoke Test & Verification Suite for DINO Traffic Suite
 Kiểm thử toàn diện 4 hướng nghiên cứu và tầng Common Utilities
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

# Thêm path
dino_dir = os.path.dirname(os.path.abspath(__file__))
if dino_dir not in sys.path:
    sys.path.insert(0, dino_dir)

print("=" * 80)
print(" [*] STARTING COMPREHENSIVE VERIFICATION SUITE")
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
    # Route 1: slot 14h và slot 15h
    dummy_bg_1_14 = Image.fromarray(np.full((224, 224, 3), 100, dtype=np.uint8))
    dummy_bg_1_14.save(os.path.join(bg_dir, "route_1", "background_slot_14h.jpg"))

    dummy_bg_1_15 = Image.fromarray(np.full((224, 224, 3), 110, dtype=np.uint8))
    dummy_bg_1_15.save(os.path.join(bg_dir, "route_1", "background_slot_15h.jpg"))

    # Route 2: slot 00h
    dummy_bg_2_00 = Image.fromarray(np.full((224, 224, 3), 50, dtype=np.uint8))
    dummy_bg_2_00.save(os.path.join(bg_dir, "route_2", "background_slot_00h.jpg"))

    # Tạo ảnh origin:
    # Origin 1: 1_1755698811.jpg (timestamp giả định)
    # Giả sử vẽ thêm một hình chữ nhật tượng trưng xe cộ
    arr_orig_1 = np.full((224, 224, 3), 100, dtype=np.uint8)
    arr_orig_1[50:120, 50:120, :] = 255  # "Xe cộ" màu trắng nổi bật
    dummy_orig_1 = Image.fromarray(arr_orig_1)
    dummy_orig_1.save(os.path.join(origin_dir, "1_1755698811.jpg"))

    # Origin 2: 1_1755702400.jpg
    arr_orig_2 = np.full((224, 224, 3), 110, dtype=np.uint8)
    arr_orig_2[80:150, 100:180, :] = 220
    dummy_orig_2 = Image.fromarray(arr_orig_2)
    dummy_orig_2.save(os.path.join(origin_dir, "1_1755702400.jpg"))

    # Origin 3: 2_1755600000.jpg
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
    print(f"   + TrafficPairMatcher tìm thấy {len(pairs)} cặp ảnh.")
    assert len(pairs) >= 2, f"Kỳ vọng ít nhất 2 cặp ảnh, thực tế: {len(pairs)}"
    for p in pairs:
        print(f"     - Origin: {p['origin_name']} -> BG: {os.path.basename(p['bg_path'])} (Slot: {p['bg_hour']}h)")

    from common.subtraction import BackgroundSubtractor
    subtractor = BackgroundSubtractor(color_space="lab", blur_kernel=3)
    delta_norm, delta_color = subtractor.compute_delta(dummy_orig_1, dummy_bg_1_14)
    assert delta_norm.shape == (224, 224), f"Delta shape sai: {delta_norm.shape}"
    assert 0.0 <= delta_norm.min() and delta_norm.max() <= 1.0, "Delta norm vượt dải [0, 1]"

    bin_mask = subtractor.extract_binary_mask(delta_norm, method="otsu")
    assert bin_mask.shape == (224, 224), f"Mask shape sai: {bin_mask.shape}"

    patch_probs = subtractor.compute_patch_mask_weights(delta_norm, img_size=224, patch_size=16, alpha=0.75)
    assert patch_probs.shape == (14, 14), f"Patch probs shape sai: {patch_probs.shape}"
    assert np.isclose(patch_probs.sum(), 1.0), f"Tổng xác suất patch phải bằng 1: {patch_probs.sum()}"
    print("   ✅ [Common] Matcher & Subtractor pass hoàn hảo!")

    # -------------------------------------------------------------
    # 2. TEST DIRECTION 1: BG-GUIDED DINO
    # -------------------------------------------------------------
    print("\n--- [TEST 2] Direction 1: BG-Guided DINO ---")
    from direction1_bg_guided_dino.dataset import BGGuidedDINODataset
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
    assert len(crops) == 4, f"Kỳ vọng 4 crops (2 global + 2 local), thực tế: {len(crops)}"
    assert sample1["fg_mask"].shape == (14 * 14,), f"Mask shape sai: {sample1['fg_mask'].shape}"
    print(f"   + Dataset sample 1 crops: 2 Global {crops[0].shape}, 2 Local {crops[2].shape}")

    from direction1_bg_guided_dino.models import DINOHead, BGGuidedDINOModel
    from direction1_bg_guided_dino.losses import BGGuidedDINOLoss

    # Tạo mock backbone nhỏ để test forward/backward
    class TinyBackbone(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
            self.embed_dim = embed_dim
        def forward(self, x):
            x = self.conv(x)
            return x.mean(dim=(2, 3)) # (B, embed_dim)

    mock_bb = TinyBackbone(embed_dim=64)
    dino_model = BGGuidedDINOModel(student_backbone=mock_bb, embed_dim=64, out_dim=512)

    # Test forward
    batch_crops = [torch.stack([c, c]) for c in crops] # batch size = 2
    student_out = dino_model.forward_student(batch_crops)
    assert student_out.shape == (2 * 4, 512), f"Student output shape sai: {student_out.shape}"

    teacher_out = dino_model.forward_teacher(batch_crops[:2])
    assert teacher_out.shape == (2 * 2, 512), f"Teacher output shape sai: {teacher_out.shape}"

    loss_fn = BGGuidedDINOLoss(out_dim=512, ncrops=4, nepochs=2)
    loss = loss_fn(student_out, teacher_out, epoch=0)
    assert not torch.isnan(loss) and loss.item() > 0, f"Loss bất thường: {loss.item()}"

    loss.backward()
    dino_model.update_teacher(0.99)
    print(f"   + Forward-Backward DINO Loss: {loss.item():.4f}")
    print("   ✅ [Direction 1] BG-Guided DINO pass hoàn hảo!")

    # -------------------------------------------------------------
    # 3. TEST DIRECTION 2: ZERO-SHOT SEGMENTATION
    # -------------------------------------------------------------
    print("\n--- [TEST 3] Direction 2: Zero-Shot Segmentation ---")
    from direction2_zero_shot_segmentation.pca_extractor import DINOPCAExtractor
    from direction2_zero_shot_segmentation.fusion import MaskFusionEngine
    from direction2_zero_shot_segmentation.segmentor import LightweightSegDecoder, VehicleSegmentor, DiceLoss

    # Mock ViT backbone có patch tokens
    class MockPatchViT(nn.Module):
        def __init__(self, embed_dim=64):
            super().__init__()
            self.embed_dim = embed_dim
            self.conv = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        def forward(self, x):
            return torch.zeros(x.shape[0], self.embed_dim)
        def get_intermediate_layers(self, x, n=1, return_class_token=True):
            feat = self.conv(x) # (B, embed_dim, 14, 14)
            b, c, h, w = feat.shape
            tokens = feat.permute(0, 2, 3, 1).reshape(b, h*w, c)
            cls_t = torch.zeros(b, c)
            return [(tokens, cls_t)]

    mock_patch_vit = MockPatchViT(embed_dim=64)
    pca_ext = DINOPCAExtractor(backbone=mock_patch_vit, patch_size=16, img_size=224, device="cpu")
    pca_rgb, pc1_mask = pca_ext.compute_pca_maps(dummy_orig_1)
    assert pca_rgb.shape == (14, 14, 3), f"PCA RGB shape sai: {pca_rgb.shape}"
    assert pc1_mask.shape == (14, 14), f"PC1 mask shape sai: {pc1_mask.shape}"

    fusion = MaskFusionEngine()
    fused_mask = fusion.fuse(bin_mask, pc1_mask, dummy_orig_1)
    assert fused_mask.shape == (224, 224), f"Fused mask shape sai: {fused_mask.shape}"

    seg_model = VehicleSegmentor(backbone=mock_patch_vit, embed_dim=64, patch_size=16)
    dummy_x = torch.randn(2, 3, 224, 224)
    pred_logits = seg_model(dummy_x)
    assert pred_logits.shape == (2, 1, 224, 224), f"Seg logits shape sai: {pred_logits.shape}"

    dice_fn = DiceLoss()
    d_loss = dice_fn(pred_logits, torch.zeros_like(pred_logits))
    assert not torch.isnan(d_loss), "Dice loss bị NaN"
    print(f"   + Fused Mask shape: {fused_mask.shape}, Dice Loss: {d_loss.item():.4f}")
    print("   ✅ [Direction 2] Zero-Shot Segmentation pass hoàn hảo!")

    # -------------------------------------------------------------
    # 4. TEST DIRECTION 3: SCENE DECOMPOSITION
    # -------------------------------------------------------------
    print("\n--- [TEST 4] Direction 3: Scene Decomposition ---")
    from direction3_scene_decomposition.dataset import DecompositionDataset
    from direction3_scene_decomposition.models import TrafficDecompositionNet
    from direction3_scene_decomposition.losses import DecompositionLoss

    ds3 = DecompositionDataset(bg_dir=bg_dir, origin_dir=origin_dir, match_strategy="route_hourly", img_size=128, is_train=True)
    sample3 = ds3[0]
    assert sample3["origin"].shape == (3, 128, 128), f"Sample3 origin shape sai: {sample3['origin'].shape}"
    assert sample3["bg"].shape == (3, 128, 128), f"Sample3 bg shape sai: {sample3['bg'].shape}"

    decomp_net = TrafficDecompositionNet(backbone=mock_patch_vit, embed_dim=64, patch_size=16, freeze_backbone=True)
    # Forward với batch 2 ảnh 128x128
    batch_orig = torch.stack([sample3["origin"], sample3["origin"]])
    batch_bg = torch.stack([sample3["bg"], sample3["bg"]])
    preds3 = decomp_net(batch_orig)

    assert preds3["pred_bg"].shape == (2, 3, 128, 128), f"Pred bg shape sai: {preds3['pred_bg'].shape}"
    assert preds3["pred_fg"].shape == (2, 3, 128, 128), f"Pred fg shape sai: {preds3['pred_fg'].shape}"
    assert preds3["pred_mask"].shape == (2, 1, 128, 128), f"Pred mask shape sai: {preds3['pred_mask'].shape}"
    assert preds3["recon_origin"].shape == (2, 3, 128, 128), f"Recon origin shape sai: {preds3['recon_origin'].shape}"

    loss_fn3 = DecompositionLoss()
    l3, l3_dict = loss_fn3(preds3, {"origin": batch_orig, "bg": batch_bg})
    assert not torch.isnan(l3) and l3.item() > 0, f"Decomp loss bất thường: {l3.item()}"
    l3.backward()
    print(f"   + Decomposition Total Loss: {l3.item():.4f} (Recon: {l3_dict['loss_recon']:.4f}, BG: {l3_dict['loss_bg']:.4f})")
    print("   ✅ [Direction 3] Scene Decomposition pass hoàn hảo!")

    # -------------------------------------------------------------
    # 5. TEST DIRECTION 4: FOREGROUND-ENHANCED COUNTING
    # -------------------------------------------------------------
    print("\n--- [TEST 5] Direction 4: Foreground-Enhanced Counting ---")
    from direction4_foreground_enhanced_counting.dataset import FGCountingDataset
    from direction4_foreground_enhanced_counting.models import DINOv3FGCountingModel, adapt_patch_embed_to_4ch

    ds4 = FGCountingDataset(
        csv_file=csv_file,
        origin_dir=origin_dir,
        bg_dir=bg_dir,
        match_strategy="route_hourly",
        img_size=224,
        is_train=True,
    )
    sample4 = ds4[0]
    assert sample4["input_4ch"].shape == (4, 224, 224), f"Input 4ch shape sai: {sample4['input_4ch'].shape}"
    assert sample4["counts"].shape == (3,), f"Counts shape sai: {sample4['counts'].shape}"
    print(f"   + Dataset sample 4: input_4ch={sample4['input_4ch'].shape}, counts={sample4['counts'].tolist()}")

    # Test adapt patch embed 4ch
    conv_3ch = nn.Conv2d(3, 64, kernel_size=16, stride=16)
    conv_4ch = adapt_patch_embed_to_4ch(conv_3ch)
    assert conv_4ch.in_channels == 4, f"adapt_patch_embed_to_4ch thất bại: in_c={conv_4ch.in_channels}"

    class MockViT4Ch(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 64, kernel_size=16, stride=16)
        def forward(self, x):
            x = self.patch_embed(x)
            return x.mean(dim=(2, 3))

    # Test model mode 4channel
    mock_vit_4ch = MockViT4Ch()
    counting_model = DINOv3FGCountingModel(backbone=mock_vit_4ch, embed_dim=64, mode="4channel")
    dummy_input_4ch = torch.randn(2, 4, 224, 224)
    pred_counts = counting_model(dummy_input_4ch)
    assert pred_counts.shape == (2, 3), f"Pred counts shape sai: {pred_counts.shape}"
    assert (pred_counts >= 0).all(), "Dự đoán số lượng xe phải >= 0 (ReLU)"

    # Test model mode spatial_attention
    mock_vit_3ch = MockViT4Ch()
    counting_model_attn = DINOv3FGCountingModel(backbone=mock_vit_3ch, embed_dim=64, mode="spatial_attention")
    dummy_input_3ch = torch.randn(2, 3, 224, 224)
    dummy_delta = torch.randn(2, 1, 224, 224)
    pred_counts_attn = counting_model_attn(dummy_input_3ch, delta=dummy_delta)
    assert pred_counts_attn.shape == (2, 3), f"Pred counts attn shape sai: {pred_counts_attn.shape}"

    crit = nn.SmoothL1Loss()
    loss4 = crit(pred_counts, torch.tensor([[10., 2., 12.], [20., 4., 24.]]))
    loss4.backward()
    print(f"   + Counting Model Output: {pred_counts[0].tolist()}, Loss: {loss4.item():.4f}")
    print("   ✅ [Direction 4] Foreground-Enhanced Counting pass hoàn hảo!")

    # -------------------------------------------------------------
    # 6. TEST MULTI-GPU SMART SAVE & LOAD CHECKPOINTING
    # -------------------------------------------------------------
    print("\n--- [TEST 6] Multi-GPU Smart Checkpointing Interoperability ---")
    from common.gpu_utils import save_checkpoint, load_checkpoint, clean_state_dict, smart_load_state_dict

    # 1. Tạo model giả lập
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        def forward(self, x):
            return self.linear(x)

    dummy_model = DummyNet()
    ckpt_save_test = os.path.join(temp_root, "test_smart_save.pth")

    # Giả lập model đang bọc trong container có tiền tố 'module.'
    class MockDataParallelWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
        def forward(self, x):
            return self.module(x)

    mock_dp_model = MockDataParallelWrapper(dummy_model)
    # Lưu từ wrapper mock DP
    save_checkpoint(
        save_path=ckpt_save_test,
        model=mock_dp_model,
        epoch=1,
        metrics={"test_metric": 0.99},
        verbose=False,
    )
    assert os.path.isfile(ckpt_save_test), "Lưu file checkpoint thất bại"

    # Kiểm tra xem file đã lưu có thực sự SẠCH tiền tố 'module.' không
    saved_raw = torch.load(ckpt_save_test, map_location="cpu")
    for k in saved_raw["model_state"].keys():
        assert not k.startswith("module."), f"Lỗi: Checkpoint vẫn còn tiền tố 'module.': {k}"
    print("   + Test 6.1: save_checkpoint dọn sạch 100% tiền tố 'module.' -> PASSED!")

    # 2. Test nạp checkpoint có chứa tiền tố 'module.' (giả sử do ai đó lưu thủ công)
    #    vào một single-GPU model không có 'module.'
    dirty_state_dict = {f"module.{k}": v for k, v in dummy_model.state_dict().items()}
    target_clean_model = DummyNet()
    missing, unexpected = smart_load_state_dict(target_clean_model, dirty_state_dict, strict=True, verbose=False)
    assert len(missing) == 0 and len(unexpected) == 0, f"smart_load_state_dict thất bại khi nạp dirty state: missing={missing}, unexpected={unexpected}"
    print("   + Test 6.2: Nạp checkpoint có 'module.' vào Single-GPU Model sạch -> PASSED!")

    # 3. Test nạp checkpoint sạch vào một model ĐANG BỌC DataParallel (có 'module.')
    clean_state_dict_sample = dummy_model.state_dict()
    target_dp_model = MockDataParallelWrapper(DummyNet())
    missing_dp, unexpected_dp = smart_load_state_dict(target_dp_model, clean_state_dict_sample, strict=True, verbose=False)
    assert len(missing_dp) == 0 and len(unexpected_dp) == 0, f"smart_load_state_dict thất bại khi nạp clean state vào DP model: missing={missing_dp}"
    print("   + Test 6.3: Nạp checkpoint sạch vào Model bọc DataParallel -> PASSED!")

    # 4. Test hàm load_checkpoint trọn gói
    loaded_dict = load_checkpoint(ckpt_save_test, model=target_clean_model, device="cpu", verbose=False)
    assert loaded_dict["epoch"] == 1, "load_checkpoint metadata không khớp"
    print("   + Test 6.4: load_checkpoint trọn gói an toàn bộ nhớ CPU/GPU -> PASSED!")
    print("   ✅ [Test 6] Multi-GPU Smart Save & Load Checkpointing pass hoàn hảo!")

    print("\n" + "=" * 80)
    print(" 🎯 ALL 6 TEST MODULES PASSED WITH 100% SUCCESS!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ [TEST FAILURE]: {e}")
    traceback.print_exc()

finally:
    # Dọn dẹp thư mục tạm
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
        print(f"🧹 Đã dọn dẹp thư mục tạm: {temp_root}")
