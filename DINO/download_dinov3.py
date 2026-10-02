"""
=============================================================================
 🚀 CÔNG CỤ TẢI VÀ XÁC THỰC MÔ HÌNH PRETRAINED METAAI DINOv3
 Module độc lập: Xác thực token Hugging Face trực tiếp với máy chủ,
 BỎ QUA cache trên máy, buộc tải mới 100% file 'model.safetensors' từ repo chính thức,
 nạp vào kiến trúc DINOv3 ViT và chạy thử nghiệm (Forward Test).
=============================================================================
"""

import os
import sys

# Tự động khắc phục xung đột runtime OpenMP trên Windows (phải đặt trước import torch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import shutil
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

# Thêm đường dẫn project vào sys.path để nạp các module nội bộ
_cur_dir = os.path.dirname(os.path.abspath(__file__))
if _cur_dir not in sys.path:
    sys.path.insert(0, _cur_dir)


# =============================================================================
# 🔑 CẤU HÌNH TOKEN HUGGING FACE
# Điền token của bạn vào biến bên dưới HOẶC truyền qua tham số dòng lệnh --hf_token
# Tạo token miễn phí (chọn Expiration: No expiration) tại:
# 👉 https://huggingface.co/settings/tokens
# =============================================================================
MY_HF_TOKEN = ""  # <--- [ĐIỀN TOKEN CỦA BẠN VÀO ĐÂY, VÍ DỤ "hf_xxxx..."]


def verify_hf_token(token: str) -> bool:
    """
    Xác thực token trực tiếp với API Hugging Face Hub (HfApi.whoami).
    Trả về True nếu token hợp lệ và in thông tin tài khoản, False nếu không hợp lệ.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        user_info = api.whoami(token=token)
        user_name = user_info.get("name", "N/A")
        user_fullname = user_info.get("fullname", "")
        auth_type = user_info.get("type", "user")
        print(f"   ✅ [Xác thực thành công] Đã đăng nhập vào Hugging Face Hub!")
        print(f"      👤 Tài khoản : @{user_name}" + (f" ({user_fullname})" if user_fullname else ""))
        print(f"      🏷️ Loại quyền: {auth_type}")
        return True
    except Exception as e:
        err_msg = str(e)
        print(f"   ❌ [Xác thực thất bại] Token không hợp lệ hoặc đã hết hạn!")
        if "expired" in err_msg.lower():
            print(f"      👉 Thông báo từ HF: Token này đã HẾT HẠN (Expired).")
        elif "invalid" in err_msg.lower() or "401" in err_msg:
            print(f"      👉 Thông báo từ HF: Token không đúng hoặc không tồn tại (401 Unauthorized).")
        else:
            print(f"      👉 Chi tiết lỗi: {err_msg}")
        print(f"      👉 Hướng dẫn: Truy cập https://huggingface.co/settings/tokens để tạo token mới (chọn No expiration).")
        return False


def download_and_verify_dinov3(
    model_name: str = "dinov3_vits16",
    hf_token: str = "",
    save_dir: str = "checkpoints",
    force_download: bool = True,
    device: str = "cpu",
):
    """
    Xác thực token, bỏ qua cache máy, tải mới trọng số từ Hugging Face Hub và kiểm thử.
    """
    token = hf_token.strip() or MY_HF_TOKEN.strip() or os.environ.get("HF_TOKEN", "").strip()

    print("=" * 78)
    print(f" 🚀 CÔNG CỤ TẢI MỚI & XÁC THỰC MÔ HÌNH METAAI DINOv3")
    print("=" * 78)
    print(f" 📦 Kiến trúc mô hình    : {model_name}")
    print(f" 💾 Thư mục lưu trữ local: {os.path.abspath(save_dir)}")
    print(f" 🖥️ Thiết bị kiểm thử    : {device.upper()}")
    print(f" 🔄 Buộc tải mới từ mạng : {'CÓ (Bỏ qua toàn bộ cache cũ)' if force_download else 'KHÔNG'}")
    print(f" 🔑 Hugging Face Token   : {'Đã nhập (***' + token[-4:] + ')' if len(token) > 6 else 'Chưa nhập'}")
    print("=" * 78)

    # BƯỚC 1: XÁC THỰC TOKEN VỚI HUGGING FACE HUB
    print(f"\n[Bước 1/4] Xác thực token trực tiếp với máy chủ Hugging Face Hub...")
    if not token:
        print("   ❌ [Thiếu Token] Bạn chưa cung cấp Hugging Face Token!")
        print("      Vui lòng điền vào biến MY_HF_TOKEN ở dòng 37 trong file, hoặc truyền qua cờ:")
        print("      python download_dinov3.py --hf_token \"hf_xxxxxxxxxxxxxx\"")
        return False

    is_valid = verify_hf_token(token)
    if not is_valid:
        return False

    # BƯỚC 2: TẢI TRỌNG SỐ TRỰC TIẾP TỪ HUGGING FACE HUB (BỎ QUA CACHE MÁY)
    clean_name = model_name.lower().strip()
    arch_tag = clean_name.replace("_", "-")
    repo_id = f"facebook/{arch_tag}-pretrain-lvd1689m"
    filename = "model.safetensors"

    os.makedirs(save_dir, exist_ok=True)
    local_target_path = os.path.join(save_dir, f"{clean_name}_{filename}")

    print(f"\n[Bước 2/4] Tải trực tiếp file '{filename}' từ repo '{repo_id}' (Bỏ qua cache máy)...")
    try:
        from huggingface_hub import hf_hub_download

        print(f"   📥 Đang tải trực tuyến từ Hugging Face Hub (force_download={force_download})...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            force_download=force_download,
        )
        print(f"   ✅ Tải hoàn tất từ máy chủ Hugging Face!")
        print(f"      Tệp tải về tạm thời: {downloaded_path}")

        # Sao chép/lưu cố định vào thư mục save_dir của dự án
        shutil.copy2(downloaded_path, local_target_path)
        print(f"   💾 Đã lưu checkpoint cố định về thư mục dự án: {local_target_path}")
        active_weights_path = local_target_path

    except Exception as e_dl:
        err_msg = str(e_dl)
        print(f"\n❌ [Lỗi Tải Trọng Số] Không thể tải file từ repo '{repo_id}'.")
        if "403" in err_msg or "gated" in err_msg.lower():
            print(f"   👉 Repo này yêu cầu chấp thuận điều khoản của Meta AI.")
            print(f"      Vui lòng truy cập https://huggingface.co/{repo_id} trên trình duyệt,")
            print(f"      đăng nhập tài khoản của bạn và bấm 'Agree and access repository'.")
        elif "404" in err_msg:
            print(f"   👉 Không tìm thấy repo hoặc file '{filename}' trên '{repo_id}'.")
        else:
            print(f"   👉 Chi tiết lỗi: {err_msg}")
        return False

    # BƯỚC 3: NẠP TRỌNG SỐ VÀO BACKBONE VÀ KIỂM TRA ĐỘ KHỚP
    print(f"\n[Bước 3/4] Nạp trọng số vừa tải vào kiến trúc ViT '{clean_name}'...")
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

    # BƯỚC 4: CHẠY THỬ NGHIỆM FORWARD PASS (HEALTH CHECK)
    print(f"\n[Bước 4/4] Chạy thử nghiệm Forward Pass với ảnh mẫu giả lập (224x224)...")
    try:
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            if hasattr(model, "get_intermediate_layers"):
                out = model.get_intermediate_layers(dummy_input, n=1, return_class_token=True)
                patch_tokens, cls_token = out[0]
                print(f"   🎯 Kích thước CLS Token    : {cls_token.shape} (Batch=1, Dim={cls_token.shape[-1]})")
                print(f"   🎯 Kích thước Patch Tokens : {patch_tokens.shape} (196 patches không gian cho ảnh 224x224)")
            else:
                out = model(dummy_input)
                print(f"   🎯 Kích thước đầu ra       : {out.shape}")

        print("\n" + "=" * 78)
        print(" 🎉 CHÚC MỪNG! MÔ HÌNH METAAI DINOv3 ĐÃ ĐƯỢC TẢI & XÁC THỰC THÀNH CÔNG 100%!")
        print("=" * 78)
        print(f" 📂 File trọng số đã sẵn sàng tại: {active_weights_path}")
        print(f" 💡 Bạn có thể dùng file này cho tất cả các hướng nghiên cứu với tham số:")
        print(f"    --weights \"{active_weights_path}\"")
        print("=" * 78 + "\n")
        return True

    except Exception as e_fwd:
        print(f"   ❌ Lỗi khi chạy Forward Pass: {e_fwd}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xác thực token và buộc tải mới mô hình DINOv3 từ Hugging Face Hub")
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
        "--no_force",
        action="store_true",
        help="Nếu bật cờ này, sẽ cho phép dùng cache cũ thay vì buộc tải mới",
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
        force_download=not args.no_force,
        device=args.device,
    )
