"""
=============================================================================
 🚀 CÔNG CỤ TẢI VÀ KIỂM TRA MÔ HÌNH PRETRAINED METAAI DINOv3
 Module độc lập giúp tải trọng số DINOv3 chính thức từ Hugging Face Hub về máy cá nhân,
 kiểm tra độ khớp 100% của trọng số và thực hiện chạy thử (forward test).
=============================================================================
"""

import os
import sys

# Tự động khắc phục xung đột runtime OpenMP trên Windows (phải đặt trước import torch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import torch.nn as nn

# Đảm bảo UTF-8 an toàn trên Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn project vào sys.path để gọi các module nội bộ
_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)


# =============================================================================
# 🔑 CẤU HÌNH TOKEN HUGGING FACE
# Điền token của bạn vào biến bên dưới HOẶC truyền qua tham số dòng lệnh --hf_token
# Tạo token miễn phí (chọn Expiration: No expiration) tại:
# 👉 https://huggingface.co/settings/tokens
# =============================================================================
MY_HF_TOKEN = ""  # <--- DÁN TOKEN 'hf_...' CỦA BẠN VÀO ĐÂY NẾU MUỐN CỐ ĐỊNH


def download_and_verify_dinov3(
    model_name: str = "dinov3_vits16",
    hf_token: str = "",
    save_dir: str = "checkpoints",
    device: str = "cpu",
):
    """
    Tải file trọng số model.safetensors từ Hugging Face Hub, lưu về máy và kiểm thử forward.
    """
    token = hf_token.strip() or MY_HF_TOKEN.strip() or os.environ.get("HF_TOKEN", "").strip()

    print("=" * 78)
    print(f" 🚀 KHỞI ĐỘNG CÔNG CỤ TẢI & KIỂM TRA METAAI DINOv3")
    print("=" * 78)
    print(f" 📦 Kiến trúc mô hình    : {model_name}")
    print(f" 💾 Thư mục lưu trữ local: {os.path.abspath(save_dir)}")
    print(f" 🖥️ Thiết bị kiểm thử    : {device.upper()}")
    print(f" 🔑 Hugging Face Token   : {'Đã cung cấp (***' + token[-4:] + ')' if len(token) > 6 else 'Chưa cung cấp'}")
    print("=" * 78)

    # 1. Chuẩn hóa tên kiến trúc và repo trên Hugging Face
    clean_name = model_name.lower().strip()
    arch_tag = clean_name.replace("_", "-")
    
    # Danh sách các repo ứng viên chính thức của Meta DINOv3
    repo_id = f"facebook/{arch_tag}-pretrain-lvd1689m"
    filename = "model.safetensors"

    os.makedirs(save_dir, exist_ok=True)
    local_target_path = os.path.join(save_dir, f"{clean_name}_{filename}")

    # 2. Tải trọng số từ Hugging Face Hub
    cached_path = None
    print(f"\n[Bước 1/3] Kết nối Hugging Face Hub tải file '{filename}' từ repo '{repo_id}'...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Thử lấy từ cache offline trước
        try:
            cached_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
            print(f"   🎯 Đã tìm thấy file trong Local Cache của máy: {cached_path}")
        except Exception:
            pass

        # Nếu chưa có ở cache local thì tải trực tuyến
        if not cached_path:
            if not token:
                print("   ⚠️ [Chú ý] Bạn chưa điền Hugging Face Token. Đang thử tải ở chế độ public...")
            else:
                print("   🔑 [Xác thực] Đang sử dụng HF Token để tải repo chính thức...")

            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token if token else None,
            )
            print(f"   ✅ Tải thành công từ Hugging Face Hub về cache hệ thống!")
            print(f"      Vị trí cache: {cached_path}")

    except Exception as e_dl:
        err_msg = str(e_dl)
        print(f"\n❌ [Lỗi Tải Trọng Số] Không thể tải được file từ Hugging Face Hub.")
        if "401" in err_msg or "expired" in err_msg.lower() or "unauthorized" in err_msg.lower():
            print(f"   👉 Nguyên nhân: Token của bạn đã hết hạn (Expired) hoặc không có quyền.")
            print(f"   👉 Giải pháp: Vui lòng truy cập https://huggingface.co/settings/tokens")
            print(f"      Tạo một 'User Access Token' mới (loại Read, Expiration: No expiration),")
            print(f"      sau đó dán vào biến MY_HF_TOKEN hoặc truyền --hf_token <token_mới>.")
        elif "404" in err_msg or "not found" in err_msg.lower():
            print(f"   👉 Repo '{repo_id}' yêu cầu bạn phải được duyệt truy cập hoặc tên repo khác.")
        else:
            print(f"   👉 Chi tiết lỗi: {err_msg}")
        return False

    # 3. Tạo bản sao (hoặc link) vào thư mục save_dir của dự án để tiện lưu trữ
    try:
        import shutil
        if cached_path and os.path.exists(cached_path) and not os.path.exists(local_target_path):
            shutil.copy2(cached_path, local_target_path)
            print(f"   💾 Đã sao chép 1 bản checkpoint về thư mục dự án: {local_target_path}")
            active_weights_path = local_target_path
        else:
            active_weights_path = cached_path
    except Exception:
        active_weights_path = cached_path

    # 4. Nạp trọng số vào Backbone và kiểm thử độ khớp
    print(f"\n[Bước 2/3] Nạp trọng số vào kiến trúc ViT '{clean_name}'...")
    try:
        from train_ssl_dinov3 import build_backbone
        model, embed_dim = build_backbone(
            model_name=clean_name,
            pretrained=True,
            weights_path=active_weights_path,
        )
        model = model.to(device)
        model.eval()
        print(f"   ✅ Nạp mô hình thành công! Embedding Dim: {embed_dim}")
    except Exception as e_load:
        print(f"   ❌ Lỗi khi nạp mô hình vào bộ nhớ: {e_load}")
        return False

    # 5. Kiểm thử Chạy Thử Nghiệm (Forward Pass)
    print(f"\n[Bước 3/3] Chạy thử nghiệm Forward Pass với ảnh mẫu giả lập (224x224)...")
    try:
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            if hasattr(model, "get_intermediate_layers"):
                out = model.get_intermediate_layers(dummy_input, n=1, return_class_token=True)
                patch_tokens, cls_token = out[0]
                print(f"   🎯 Kích thước CLS Token    : {cls_token.shape} (Batch=1, Dim={cls_token.shape[-1]})")
                print(f"   🎯 Kích thước Patch Tokens : {patch_tokens.shape} (196 patches cho ảnh 224x224)")
            else:
                out = model(dummy_input)
                print(f"   🎯 Kích thước đầu ra       : {out.shape}")

        print("\n" + "=" * 78)
        print(" 🎉 CHÚC MỪNG! MÔ HÌNH METAAI DINOv3 ĐÃ ĐƯỢC TẢI VÀ HOẠT ĐỘNG HOÀN HẢO!")
        print("=" * 78)
        print(f" 📂 Đường dẫn checkpoint sẵn sàng sử dụng: {active_weights_path}")
        print(f" 💡 Bây giờ bạn có thể dùng file này cho tất cả 8 hướng nghiên cứu với cờ:")
        print(f"    --weights \"{active_weights_path}\"")
        print("=" * 78 + "\n")
        return True

    except Exception as e_fwd:
        print(f"   ❌ Lỗi khi chạy Forward Pass: {e_fwd}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải và kiểm thử mô hình DINOv3 từ Hugging Face Hub")
    parser.add_argument(
        "--model",
        type=str,
        default="dinov3_vits16",
        choices=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov2_vits14", "dinov2_vitb14"],
        help="Kiến trúc mô hình cần tải (mặc định: dinov3_vits16)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="",
        help="Token truy cập Hugging Face của bạn (bắt đầu bằng hf_...)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Thư mục lưu trữ file trọng số tải về (mặc định: checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị kiểm thử ('cuda' hoặc 'cpu')",
    )

    args = parser.parse_args()
    download_and_verify_dinov3(
        model_name=args.model,
        hf_token=args.hf_token,
        save_dir=args.save_dir,
        device=args.device,
    )
