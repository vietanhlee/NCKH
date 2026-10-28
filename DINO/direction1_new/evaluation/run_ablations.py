"""
=============================================================================
 Hướng G: run_ablations.py
 Tự động hóa ma trận 7 thí nghiệm Ablation Study khoa học:
 1. Full Method (TAM + AGM Block + SRS Continuous + RIC + CamAdv)
 2. w/o RIC (Tắt hard-negative cùng camera, kiểm chứng bất biến vs tập trung)
 3. w/o SRS (Tắt hoán nền, dùng ảnh gốc)
 4. SRS Cross-cam (Hoán nền khác camera)
 5. AGM vs Random Mask (Che ngẫu nhiên thay vì phân tầng theo pi)
 6. TAM pi vs Random pi (pi ngẫu nhiên thay vì mô hình không-thời gian)
 7. Vanilla SSL Baseline (Standard DINOv3 không TAM/SRS/RIC)
=============================================================================
"""

import os
import sys
import argparse
import subprocess
import json
import time

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
_dino_root = os.path.dirname(_pkg_dir)


ABLATION_CONFIGS = [
    {
        "name": "full_method",
        "desc": "Full Proposed Method (TAM + AGM Block + SRS Continuous + RIC + CamAdv)",
        "args": [
            "--mask_mode", "block",
            "--srs_continuous",
            "--ibot_beta", "0.5",
            "--use_ric",
            "--use_cam_adv",
            "--use_cam_sampler",
        ],
    },
    {
        "name": "wo_ric",
        "desc": "w/o RIC (Loại bỏ Hard Negative Contrastive cùng camera)",
        "args": [
            "--mask_mode", "block",
            "--srs_continuous",
            "--ibot_beta", "0.5",
            "--no_ric",
            "--use_cam_adv",
            "--no_cam_sampler",
        ],
    },
    {
        "name": "wo_srs",
        "desc": "w/o SRS (Không hoán đổi vùng tĩnh nền, dùng ảnh gốc p_srs=0.0)",
        "args": [
            "--p_srs", "0.0",
            "--mask_mode", "block",
            "--ibot_beta", "0.5",
            "--use_ric",
            "--use_cam_adv",
        ],
    },
    {
        "name": "srs_cross_cam",
        "desc": "SRS Cross-Camera (Hoán nền từ camera khác để triệt góc máy cố định)",
        "args": [
            "--cross_cam_srs",
            "--mask_mode", "block",
            "--srs_continuous",
            "--ibot_beta", "0.5",
            "--use_ric",
            "--use_cam_adv",
        ],
    },
    {
        "name": "random_mask",
        "desc": "AGM vs Random Masking (Che ngẫu nhiên không có phân tầng pi)",
        "args": [
            "--mask_mode", "random",
            "--srs_continuous",
            "--ibot_beta", "0.5",
            "--use_ric",
            "--use_cam_adv",
        ],
    },
    {
        "name": "random_pi",
        "desc": "TAM pi vs Random pi (Bản đồ pi giả ngẫu nhiên thay cho TAM)",
        "args": [
            "--random_pi",
            "--mask_mode", "block",
            "--srs_continuous",
            "--ibot_beta", "0.5",
            "--use_ric",
            "--use_cam_adv",
        ],
    },
    {
        "name": "vanilla_ssl_baseline",
        "desc": "Vanilla SSL Baseline (Standard DINOv3 không TAM/SRS/RIC)",
        "args": [
            "--p_srs", "0.0",
            "--mask_mode", "random",
            "--ibot_beta", "0.0",
            "--no_ric",
            "--no_cam_adv",
            "--no_cam_sampler",
            "--skip_tam_warmup",
            "--disable_tam",
        ],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Matrix Runner cho Hướng 1 Mới")
    parser.add_argument("--data_dir", type=str, default="/workspace/output", help="Thư mục dữ liệu")
    parser.add_argument("--output_root", type=str, default="checkpoints/ablations", help="Thư mục gốc lưu các ablation runs")
    parser.add_argument("--epochs", type=int, default=2, help="Số epoch cho mỗi ablation run (mặc định 2 cho nhanh)")
    parser.add_argument("--batch_size", type=int, default=4, help="Kích thước batch")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace Token")
    parser.add_argument("--dry_run", action="store_true", help="Chỉ in ra các lệnh CLI mà không thực thi")
    parser.add_argument("--run_all", action="store_true", help="Thực thi tuần tự toàn bộ các bài thí nghiệm")
    parser.add_argument("--configs", nargs="+", default=None, help="Chỉ định danh sách cấu hình muốn chạy (vd: full_method wo_ric)")
    args = parser.parse_args()

    print("=" * 80)
    print(" 🧪 [ABLATION STUDY PIPELINE] MA TRẬN 7 THÍ NGHIỆM ĐỐI CHỨNG HỌC THUẬT")
    print("=" * 80)

    selected_configs = ABLATION_CONFIGS
    if args.configs:
        selected_configs = [c for c in ABLATION_CONFIGS if c["name"] in args.configs]

    commands = []
    python_exe = sys.executable
    train_script = os.path.join(_dino_root, "direction1_new", "train.py")

    for cfg in selected_configs:
        name = cfg["name"]
        save_dir = os.path.join(args.output_root, name)
        cmd = [
            python_exe, train_script,
            "--data_dir", args.data_dir,
            "--output_dir", save_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
        ]
        if args.hf_token:
            cmd.extend(["--hf_token", args.hf_token])
        cmd.extend(cfg["args"])
        commands.append((name, cfg["desc"], cmd, save_dir))

    print(f"Tổng số thí nghiệm: {len(commands)}")
    for idx, (name, desc, cmd, save_dir) in enumerate(commands, 1):
        cmd_str = " ".join(cmd)
        print(f"\n[{idx}/{len(commands)}] {name.upper()}")
        print(f"  📝 Mô tả : {desc}")
        print(f"  📂 Lưu tại: {save_dir}")
        print(f"  💻 Lệnh CLI:\n     {cmd_str}")

    if args.dry_run or not args.run_all:
        print("\n" + "=" * 80)
        print("💡 [GỢI Ý] Đây là chế độ DRY-RUN hoặc liệt kê. Để chạy tự động toàn bộ, hãy thêm tham số `--run_all`.")
        print("=" * 80)
        return

    # Thực thi tuần tự
    results_summary = []
    for idx, (name, desc, cmd, save_dir) in enumerate(commands, 1):
        print("\n" + "=" * 80)
        print(f"🚀 BẮT ĐẦU CHẠY [{idx}/{len(commands)}]: {name.upper()}")
        print("=" * 80)
        start_t = time.time()
        ret = subprocess.run(cmd)
        elapsed = time.time() - start_t
        status = "Thành Công" if ret.returncode == 0 else "Thất Bại"
        print(f"🏁 [{name}] Kết thúc với trạng thái: {status} ({elapsed:.1f}s)")
        results_summary.append({
            "name": name,
            "desc": desc,
            "status": status,
            "return_code": ret.returncode,
            "elapsed_seconds": elapsed,
            "save_dir": save_dir,
        })

    summary_file = os.path.join(args.output_root, "ablation_summary.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)
    print(f"\n📊 Báo cáo tổng kết toàn bộ Ablation Study đã lưu tại: {summary_file}")


if __name__ == "__main__":
    main()
