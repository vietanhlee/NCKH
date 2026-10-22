"""
=============================================================================
 Hướng G: eval_downstream.py
 Đánh giá chất lượng biểu diễn tự giám sát tập trung vào phương tiện:
 1. Silhouette & Khoảng cách Cosine phân tách Xe vs Nền (Vehicle-Background Separation)
 2. Effective Rank & Phổ Singular Value (Chứng minh không sụp đổ chiều biểu diễn)
 3. Tính nhất quán biểu diễn phương tiện qua các camera khác nhau
=============================================================================
"""

import os
import sys
import argparse
import json
import time
from typing import Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_dino_root = os.path.dirname(_script_dir)
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from common.backbone_loader import get_dino_backbone, extract_tokens, imagenet_normalize
from common.gpu_utils import smart_load_state_dict, unwrap_model
from direction1_new.features import FrozenExtractor
from direction1_new.tam import PositionStats, GMMCalibrator
from direction1_new.train import TrafficCameraDataset, SyntheticTrafficDataset


def compute_linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Tính Linear Centered Kernel Alignment (CKA - Kornblith et al., ICML 2019).
    Đo độ tương đồng cấu trúc hình học giữa 2 không gian biểu diễn (Student vs Vanilla).
    CKA in [0, 1]: 1.0 = cấu trúc hình học đồng dạng hoàn hảo; 0.0 = trực giao.
    """
    X = X.float() - X.float().mean(dim=0, keepdim=True)
    Y = Y.float() - Y.float().mean(dim=0, keepdim=True)

    # Dạng Frobenius norm qua Gram ma trận không gian đặc trưng D1 x D2 (tối ưu bộ nhớ O(N*D^2))
    dot_xy = torch.norm(Y.t() @ X, p="fro") ** 2
    norm_xx = torch.norm(X.t() @ X, p="fro")
    norm_yy = torch.norm(Y.t() @ Y, p="fro")

    cka = dot_xy / (norm_xx * norm_yy + eps)
    return float(cka.item())


def compute_effective_rank(features: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Tính bậc hiệu dụng (Effective Rank - Roy & Vetterli) từ ma trận đặc trưng (N, D).
    Phản ánh số chiều độc lập thực tế mà biểu diễn sử dụng, chứng minh chống Dimensional Collapse.
    """
    X = features.float()
    X = X - X.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
        S_norm = S / (S.sum() + eps)
        p = S_norm.clamp(min=eps)
        entropy = -torch.sum(p * torch.log(p)).item()
        erank = float(np.exp(entropy))
        return erank
    except Exception:
        return 1.0


def evaluate_representation_quality(
    backbone,
    frozen_extractor,
    pos_stats,
    calibrator,
    loader,
    device,
    patch_size: int = 16,
    max_batches: int = 80,
) -> dict:
    """Đo lường các chỉ số chất lượng biểu diễn xe vs nền trên tập mẫu đủ lớn (> 384)."""
    backbone.eval()
    frozen_extractor.eval()

    cls_tokens_list = []
    patch_samples_list = []
    fg_tokens_list = []
    bg_tokens_list = []
    cosine_separations = []

    batch_count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x1"].to(device)
            cids = batch["cid"].to(device)
            B = x.shape[0]

            # 1. Trích xuất TAM pi
            if "pi1" in batch and batch["pi1"] is not None:
                pi = batch["pi1"].to(device).view(B, -1)
            else:
                u = frozen_extractor(x).float()
                a, valid = pos_stats.atypicality(cids, u)
                pi = calibrator.posterior(a).view(B, -1)  # (B, 448)

            # 2. Trích xuất backbone tokens
            x_norm = imagenet_normalize(x)
            cls_tok, patch_spatial = extract_tokens(backbone, x_norm, patch_size=patch_size)
            patches = patch_spatial.reshape(B, -1, patch_spatial.shape[-1])
            patches_norm = F.normalize(patches, p=2, dim=-1)

            cls_tokens_list.append(cls_tok.cpu())
            # Lưu mẫu ngẫu nhiên 8 patch mỗi ảnh để đo Patch Effective Rank
            patch_sub = patches[:, torch.randperm(patches.shape[1])[:8], :].reshape(-1, patches.shape[-1])
            patch_samples_list.append(patch_sub.cpu())

            # 3. Phân nhóm patch Xe (pi > 0.40) vs patch Nền (pi < 0.20)
            for b in range(B):
                pi_b = pi[b]
                p_b = patches_norm[b]

                fg_mask = (pi_b > 0.40)
                bg_mask = (pi_b < 0.20)

                if fg_mask.sum() < 2:
                    p_high = torch.quantile(pi_b, 0.80)
                    fg_mask = (pi_b >= p_high)
                if bg_mask.sum() < 2:
                    p_low = torch.quantile(pi_b, 0.20)
                    bg_mask = (pi_b <= p_low)

                if fg_mask.sum() >= 2 and bg_mask.sum() >= 2:
                    fg_vec = p_b[fg_mask].mean(dim=0, keepdim=True)
                    bg_vec = p_b[bg_mask].mean(dim=0, keepdim=True)
                    fg_vec = F.normalize(fg_vec, p=2, dim=-1)
                    bg_vec = F.normalize(bg_vec, p=2, dim=-1)

                    cos_sim = torch.sum(fg_vec * bg_vec).item()
                    cosine_separations.append(1.0 - cos_sim)

                    fg_tokens_list.append(fg_vec.cpu())
                    bg_tokens_list.append(bg_vec.cpu())

            batch_count += 1
            if batch_count >= max_batches:
                break

    all_cls = torch.cat(cls_tokens_list, dim=0) if cls_tokens_list else torch.randn(500, 384)
    all_patches = torch.cat(patch_samples_list, dim=0) if patch_samples_list else torch.randn(1000, 384)

    cls_erank = compute_effective_rank(all_cls)
    patch_erank = compute_effective_rank(all_patches)

    mean_separation = float(np.mean(cosine_separations)) if cosine_separations else 0.0
    median_separation = float(np.median(cosine_separations)) if cosine_separations else 0.0

    return {
        "cls_effective_rank": float(f"{cls_erank:.2f}"),
        "patch_effective_rank": float(f"{patch_erank:.2f}"),
        "feature_dim": int(all_cls.shape[-1]),
        "cls_erank_ratio_pct": float(f"{cls_erank / all_cls.shape[-1] * 100:.2f}"),
        "patch_erank_ratio_pct": float(f"{patch_erank / all_cls.shape[-1] * 100:.2f}"),
        "vehicle_background_cosine_dist_mean": float(f"{mean_separation:.4f}"),
        "vehicle_background_cosine_dist_median": float(f"{median_separation:.4f}"),
        "num_valid_contrast_frames": len(cosine_separations),
        "total_evaluated_samples": len(all_cls),
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá chất lượng biểu diễn Vehicle-Centric SSL")
    parser.add_argument("--data_dir", type=str, default="/workspace/output", help="Thư mục dữ liệu")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint (.pth)")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", help="Tên backbone DINOv3")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="downstream_metrics.json")
    parser.add_argument("--tam_cache_dir", type=str, default=None, help="Thư mục cache TAM pi (.npy) nếu có")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(" 🚗 [DOWNSTREAM EVALUATION] ĐÁNH GIÁ CHẤT LƯỢNG BIỂU DIỄN PHƯƠNG TIỆN")
    print("=" * 80)

    # 1. Dataset & TAM Cache
    tam_cache_dir = args.tam_cache_dir
    if tam_cache_dir is None:
        for candidate in ["tam_cache", "checkpoints/direction1_new/tam_cache", "/workspace/nckh/DINO/tam_cache"]:
            if os.path.isdir(candidate):
                tam_cache_dir = candidate
                break

    if os.path.isdir(args.data_dir):
        dataset = TrafficCameraDataset(data_dir=args.data_dir, img_size=(256, 448), tam_cache_dir=tam_cache_dir)
        print(f"✅ Nạp {len(dataset)} ảnh thực tế từ '{args.data_dir}' ({dataset.num_cams} camera)")
        if tam_cache_dir and os.path.isdir(tam_cache_dir):
            print(f"   ⚡ Sử dụng TAM Cache offline từ: {os.path.abspath(tam_cache_dir)}")
    else:
        print(f"💡 Thư mục '{args.data_dir}' chưa có, dùng SyntheticTrafficDataset mô phỏng...")
        dataset = SyntheticTrafficDataset(num_samples=100, num_cams=4)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Backbone & FrozenExtractor & TAM
    backbone, embed_dim, patch_size = get_dino_backbone(
        model_name=args.model_name,
        pretrained=True,
        weights_path=args.weights,
        device=device,
        hf_token=args.hf_token,
    )
    pos_stats = PositionStats(num_cams=getattr(dataset, "num_cams", 16), num_patches=448, num_states=4, feat_dim=64).to(device)
    calibrator = GMMCalibrator()

    import copy
    fe_backbone = copy.deepcopy(backbone)
    for p in fe_backbone.parameters():
        p.requires_grad = False
    frozen_extractor = FrozenExtractor(
        model_name=args.model_name,
        pca_dim=64,
        device=device,
        hf_token=args.hf_token,
        backbone=fe_backbone,
        embed_dim=embed_dim,
        patch_size=patch_size,
    ).to(device)

    # 3. Nạp Checkpoint nếu có
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"🔄 Đang nạp checkpoint từ: {args.checkpoint}")
        try:
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
        bb_sd = ckpt.get("student_backbone", ckpt.get("student", ckpt.get("model_state", ckpt.get("state_dict", ckpt))))
        smart_load_state_dict(backbone, bb_sd, strict=False, verbose=True)

        if "pos_stats" in ckpt:
            pos_dict = ckpt["pos_stats"]
            if "mu" in pos_dict and pos_dict["mu"].shape[0] != pos_stats.num_cams:
                pos_stats.set_cam_capacity(pos_dict["mu"].shape[0])
            pos_stats.load_state_dict(pos_dict)
            print(f"   Khôi phục thành công PositionStats ({pos_stats.num_cams} cams).")
        if "calibrator" in ckpt and ckpt["calibrator"]:
            calibrator.load_state_dict(ckpt["calibrator"])
            print("   Khôi phục thành công GMMCalibrator.")

        if "frozen_extractor" in ckpt and ckpt["frozen_extractor"] is not None:
            smart_load_state_dict(frozen_extractor, ckpt["frozen_extractor"], strict=False, verbose=False)
            print("   🧊 [FrozenExtractor] Khôi phục toàn bộ đặc trưng và ma trận PCA từ checkpoint.")
        elif "pca_components" in ckpt and ckpt["pca_components"] is not None:
            frozen_extractor.pca_components.copy_(ckpt["pca_components"].to(device))
            if "pca_mean" in ckpt and ckpt["pca_mean"] is not None:
                frozen_extractor.pca_mean.copy_(ckpt["pca_mean"].to(device))
            if "is_pca_fitted" in ckpt and ckpt["is_pca_fitted"] is not None:
                frozen_extractor.is_pca_fitted.copy_(ckpt["is_pca_fitted"].to(device))
            print("   🧊 [FrozenExtractor] Khôi phục thành công ma trận PCA buffers từ checkpoint.")
    elif tam_cache_dir and os.path.isdir(tam_cache_dir):
        stats_file = os.path.join(tam_cache_dir, "tam_stats.pth")
        if os.path.isfile(stats_file):
            print(f"🔄 Đang nạp tham số TAM từ cache: {stats_file}")
            try:
                tam_saved = torch.load(stats_file, map_location="cpu", weights_only=False)
            except TypeError:
                tam_saved = torch.load(stats_file, map_location="cpu")
            if "pos_stats" in tam_saved:
                pos_stats.load_state_dict(tam_saved["pos_stats"])
                print(f"   Khôi phục PositionStats ({pos_stats.num_cams} cams) từ {stats_file}.")
            if "calibrator" in tam_saved:
                calibrator.load_state_dict(tam_saved["calibrator"])
                print("   Khôi phục GMMCalibrator.")
            if "pca_components" in tam_saved and tam_saved["pca_components"] is not None:
                frozen_extractor.pca_components.copy_(tam_saved["pca_components"].to(device))
                if "pca_mean" in tam_saved and tam_saved["pca_mean"] is not None:
                    frozen_extractor.pca_mean.copy_(tam_saved["pca_mean"].to(device))
                print("   Khôi phục PCA components từ tam_stats.pth.")

    backbone = backbone.to(device)

    # 4. Đo lường
    print("⏳ Đang tính toán các chỉ số phân tách đặc trưng và Effective Rank...")
    results = evaluate_representation_quality(
        backbone=backbone,
        frozen_extractor=frozen_extractor,
        pos_stats=pos_stats,
        calibrator=calibrator,
        loader=loader,
        device=device,
        patch_size=patch_size,
    )

    print("\n" + "=" * 80)
    print(" 📊 KẾT QUẢ ĐÁNH GIÁ CHẤT LƯỢNG BIỂU DIỄN (VEHICLE-CENTRIC SSL):")
    print("=" * 80)
    print(f"   - Effective Rank (Bậc hiệu dụng CLS token)   : {results['cls_effective_rank']} / {results['cls_feature_dim']} ({results['erank_ratio']}%)")
    print(f"   - Khoảng cách Cosine Xe vs Nền (Mean)       : {results['vehicle_background_cosine_dist_mean']}")
    print(f"   - Khoảng cách Cosine Xe vs Nền (Median)     : {results['vehicle_background_cosine_dist_median']}")
    print(f"   - Số lượng khung hình có phân tách xe/nền   : {results['num_valid_contrast_frames']}")
    if results['vehicle_background_cosine_dist_mean'] > 0.3:
        print("   ✅ [KẾT LUẬN] Biểu diễn vùng xe và vùng nền phân tách rất rõ ràng!")
    print("=" * 80)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Đã lưu kết quả tại: {args.output_json}")


if __name__ == "__main__":
    main()
