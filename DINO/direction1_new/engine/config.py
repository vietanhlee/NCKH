"""
=============================================================================
 Hướng 1 Mới (Direction 1 New): Quản lý tham số dòng lệnh CLI (Config)
=============================================================================
"""

import argparse
import os
from typing import List


def get_cli_specified_args(argv_list: List[str]) -> set:
    """
    Xác định tập hợp các tham số được người dùng chỉ định tường minh trên dòng lệnh CLI.
    Giúp phân biệt giữa tham số người dùng muốn ghi đè và tham số mặc định cần kế thừa từ checkpoint.
    """
    specified = set()
    cli_map = {
        "--data_dir": "data_dir", "--origin_dir": "data_dir",
        "--model_name": "model_name", "--backbone": "model_name",
        "--batch_size": "batch_size",
        "--epochs": "epochs",
        "--lr": "lr",
        "--backbone_lr_scale": "backbone_lr_scale",
        "--weight_decay": "weight_decay",
        "--phi": "phi",
        "--q_max": "q_max",
        "--mask_ratio": "mask_ratio",
        "--mask_mode": "mask_mode",
        "--p_srs": "p_srs",
        "--srs_continuous": "srs_continuous", "--no_srs_continuous": "srs_continuous",
        "--cross_cam_srs": "cross_cam_srs",
        "--disable_tam": "disable_tam",
        "--random_pi": "random_pi",
        "--ibot_beta": "ibot_beta",
        "--use_ric": "use_ric", "--no_ric": "use_ric",
        "--lambda_ric": "lambda_ric",
        "--use_cam_adv": "use_cam_adv", "--no_cam_adv": "use_cam_adv",
        "--lambda_cam_adv": "lambda_cam_adv",
        "--max_grl_alpha": "max_grl_alpha",
        "--use_patch_var": "use_patch_var", "--no_patch_var": "use_patch_var",
        "--lambda_patch_var": "lambda_patch_var",
        "--use_anchor": "use_anchor", "--no_anchor": "use_anchor",
        "--lambda_anchor": "lambda_anchor",
        "--use_cam_sampler": "use_cam_sampler", "--no_cam_sampler": "use_cam_sampler",
        "--cams_per_batch": "cams_per_batch",
        "--frames_per_cam": "frames_per_cam",
        "--out_dim": "out_dim",
        "--output_dir": "output_dir", "--save_dir": "output_dir",
        "--weights": "weights",
        "--device": "device",
        "--num_workers": "num_workers",
        "--seed": "seed",
        "--use_amp": "use_amp", "--no_amp": "use_amp",
        "--amp_dtype": "amp_dtype",
        "--resume": "resume",
        "--hf_token": "hf_token",
        "--tam_warmup_frames": "tam_warmup_frames",
        "--tam_cache_dir": "tam_cache_dir",
        "--tam_kmeans_iters": "tam_kmeans_iters",
        "--gmm_buffer_size": "gmm_buffer_size",
        "--gmm_refit_steps": "gmm_refit_steps",
        "--gmm_iters": "gmm_iters",
        "--gmm_ema_alpha": "gmm_ema_alpha",
        "--gmm_subsample_ratio": "gmm_subsample_ratio",
        "--skip_tam_warmup": "skip_tam_warmup",
        "--session_epochs": "session_epochs", "--max_session_epochs": "session_epochs",
        "--warmup_epochs": "warmup_epochs",
    }
    for arg in argv_list:
        clean_arg = arg.split("=")[0].strip()
        if clean_arg in cli_map:
            specified.add(cli_map[clean_arg])
    return specified


def parse_args():
    """Phân tích đối số dòng lệnh CLI cho huấn luyện tự giám sát Hướng 1 Mới."""
    parser = argparse.ArgumentParser(description="Huấn luyện Tự Giám Sát DINOv3 Hướng G (TAM + AGM + SRS + RIC + CamAdv)")
    parser.add_argument("--data_dir", "--origin_dir", dest="data_dir", type=str, default=None, help="Thư mục ảnh giao thông thực tế (ví dụ: output)")
    parser.add_argument("--model_name", "--backbone", dest="model_name", type=str, default="dinov3_vits16", help="Tên backbone DINOv3")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước batch (hoặc số camera mỗi batch nếu dùng CameraSampler)")
    parser.add_argument("--epochs", type=int, default=12, help="Tổng số epochs huấn luyện theo kế hoạch toàn bộ (mặc định: 12)")
    parser.add_argument("--session_epochs", "--max_session_epochs", dest="session_epochs", type=int, default=0,
                        help="Số epochs tối đa chạy trong phiên làm việc hiện tại (0 = chạy hết toàn bộ --epochs, ví dụ: 4)")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Số epochs khởi động ấm Learning Rate tuyến tính (mặc định: 1)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Tốc độ học cơ sở cho Projection Heads")
    parser.add_argument("--backbone_lr_scale", type=float, default=0.2, help="Tỷ lệ giảm LR cho pretrained backbone (tránh làm lệch feature)")
    parser.add_argument("--weight_decay", type=float, default=0.04, help="Hệ số weight decay")
    parser.add_argument("--phi", type=float, default=0.5, help="Tỷ lệ che ngân sách xe AGM")
    parser.add_argument("--q_max", type=float, default=0.6, help="Tỷ lệ tối đa patch xe bị che AGM")
    parser.add_argument("--mask_ratio", type=float, default=0.35, help="Tỷ lệ tổng số patch bị che trong AGM (mặc định 0.35)")
    parser.add_argument("--mask_mode", type=str, default="block", choices=["patch", "block", "random"],
                        help="Chế độ che AGM: 'block' (liền khối xe), 'patch' (phân tầng rời rạc), 'random' (ngẫu nhiên)")
    parser.add_argument("--p_srs", type=float, default=0.5, help="Xác suất áp dụng SRS hoán nền")
    parser.add_argument("--srs_continuous", action="store_true", default=True, help="Hoán đổi toàn bộ nền tĩnh tự nhiên (thay vì chắp vá từng patch)")
    parser.add_argument("--no_srs_continuous", dest="srs_continuous", action="store_false")
    parser.add_argument("--cross_cam_srs", action="store_true", default=False, help="Ablation: Hoán nền từ camera khác để triệt góc máy")
    parser.add_argument("--disable_tam", action="store_true", default=False,
                        help="Baseline DINOv3 Continued SSL thuần túy: tắt hoàn toàn TAM, SRS, RIC, CamAdv và Photometric Jitter vùng xe")
    parser.add_argument("--refine_tam", action="store_true", default=True, help="Tự động áp dụng bộ lọc làm sắc nét tiền cảnh và khử nhiễu mặt đường khi đọc cache TAM.")
    parser.add_argument("--no_refine_tam", dest="refine_tam", action="store_false", help="Tắt bộ lọc làm sắc nét tiền cảnh khi đọc cache.")
    parser.add_argument("--random_pi", action="store_true", default=False, help="Ablation: Sử dụng bản đồ pi ngẫu nhiên thay cho TAM")
    parser.add_argument("--ibot_beta", type=float, default=0.5, help="Trọng số tiền cảnh beta trong iBOT Patch Loss (1 + beta * pi)")
    parser.add_argument("--use_ric", action="store_true", default=True, help="Bật Regional Instance Contrastive (RIC) cùng camera")
    parser.add_argument("--no_ric", dest="use_ric", action="store_false")
    parser.add_argument("--lambda_ric", type=float, default=0.15, help="Hệ số trọng số RIC Loss")
    parser.add_argument("--use_cam_adv", action="store_true", default=True, help="Bật Camera Adversarial Discriminator (GRL) chống shortcut")
    parser.add_argument("--no_cam_adv", dest="use_cam_adv", action="store_false")
    parser.add_argument("--lambda_cam_adv", type=float, default=0.10, help="Hệ số trọng số Camera Adversarial Loss (mặc định nâng lên 0.10 để phản kháng hiệu dụng)")
    parser.add_argument("--max_grl_alpha", type=float, default=0.5, help="Hệ số alpha cực đại cho Camera Adversarial GRL")
    parser.add_argument("--use_patch_var", action="store_true", default=True, help="Bật Patch-level Variance Regularizer chống Dimensional Collapse")
    parser.add_argument("--no_patch_var", dest="use_patch_var", action="store_false")
    parser.add_argument("--lambda_patch_var", type=float, default=0.10, help="Hệ số trọng số Patch Variance Loss")
    parser.add_argument("--use_anchor", action="store_true", default=True, help="Bật Feature Anchor Loss chống trôi dạt biểu diễn khỏi DINOv3 gốc")
    parser.add_argument("--no_anchor", dest="use_anchor", action="store_false")
    parser.add_argument("--lambda_anchor", type=float, default=0.05, help="Hệ số trọng số Feature Anchor Loss")
    parser.add_argument("--use_cam_sampler", action="store_true", default=True, help="Bật CameraGroupedSampler gom nhóm theo camera")
    parser.add_argument("--no_cam_sampler", dest="use_cam_sampler", action="store_false")
    parser.add_argument("--cams_per_batch", type=int, default=4, help="Số camera trong 1 batch của CameraGroupedSampler")
    parser.add_argument("--frames_per_cam", type=int, default=5, help="Số frame mỗi camera trong 1 batch của CameraGroupedSampler (mặc định: 5)")
    parser.add_argument("--batches_per_epoch", type=int, default=0,
                        help="Số batches mỗi epoch khi bật CameraGroupedSampler (0 = 1 lượt camera ~152 batches; -1 = duyệt trọn vẹn số mẫu dataset; >0 = số batches cụ thể, v.d. 1000, 2000)")
    parser.add_argument("--out_dim", type=int, default=4096, help="Số prototype DINO/iBOT head")
    parser.add_argument("--output_dir", "--save_dir", dest="output_dir", type=str, default="checkpoints/direction1_new", help="Thư mục lưu weights")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn custom checkpoint ban đầu (.pth / .safetensors)")
    parser.add_argument("--device", type=str, default="cuda", help="Thiết bị tính toán ('cuda', 'cuda:0', hoặc 'cpu')")
    parser.add_argument("--num_workers", type=int, default=0, help="Số luồng nạp dữ liệu DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Sử dụng Mixed Precision (AMP) để tối ưu VRAM và tăng tốc độ")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Tắt Mixed Precision")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16"],
        help="Kiểu dữ liệu AMP: 'auto' (tự động dùng bfloat16 nếu GPU hỗ trợ Tensor Cores BF16, ngược lại dùng fp16), 'bf16', hoặc 'fp16'",
    )
    parser.add_argument("--resume", type=str, default=None, help="Đường dẫn checkpoint (.pth) để tiếp tục huấn luyện")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace Token")
    parser.add_argument("--tam_warmup_frames", type=int, default=200, help="Số frame mỗi camera dùng để khởi động TAM (fit PCA & K-means, mặc định: 200 frame)")
    parser.add_argument("--tam_cache_dir", type=str, default=None, help="Thư mục chứa cache pi đã tính trước bởi precompute_tam.py (tăng tốc độ nạp 2.5x)")
    parser.add_argument("--tam_kmeans_iters", type=int, default=20, help="Số vòng lặp K-means khi khởi tạo cụm trạng thái nền")
    parser.add_argument("--gmm_buffer_size", type=int, default=300000, help="Dung lượng bộ đệm mẫu atypicality để fit GMM Calibrator (mặc định: 300000 mẫu)")
    parser.add_argument("--gmm_refit_steps", type=int, default=100, help="Chu kỳ số bước huấn luyện cập nhật lại GMM Calibrator (mặc định: 100)")
    parser.add_argument("--gmm_iters", type=int, default=30, help="Số vòng lặp EM khi fit GMM Calibrator")
    parser.add_argument("--gmm_ema_alpha", type=float, default=0.15, help="Hệ số EMA làm mượt tham số GMM khi refit (mặc định: 0.15)")
    parser.add_argument("--gmm_subsample_ratio", type=float, default=0.15, help="Tỷ lệ subsample patch ngẫu nhiên khi train để giảm tương quan không gian (mặc định: 0.15)")
    parser.add_argument("--skip_tam_warmup", action="store_true", default=False, help="Bỏ qua bước khởi động TAM offline")
    parser.add_argument("--log_file", type=str, default=None, help="Tên hoặc đường dẫn file log ghi lại toàn bộ output (mặc định: train.log trong --output_dir)")
    parsed, unknown = parser.parse_known_args()
    if unknown:
        print(f"⚠️ [CLI Warning] Bỏ qua các đối số chưa khai báo: {unknown}")
    if parsed.hf_token:
        os.environ["HF_TOKEN"] = parsed.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = parsed.hf_token
    return parsed
