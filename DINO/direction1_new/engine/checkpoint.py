"""
=============================================================================
 Hướng 1 Mới (Direction 1 New): Tiện ích giải quyết và lưu trữ Checkpoint
=============================================================================
"""

import glob
import os


def resolve_resume_checkpoint(resume_target: str) -> str:
    """
    Chuẩn hóa và tìm kiếm file checkpoint an toàn trên cả Local lẫn Kaggle/Colab/Server.
    Hỗ trợ đường dẫn file trực tiếp, URL Kaggle copy-paste nhầm, hoặc đường dẫn thư mục.
    """
    if not resume_target:
        raise ValueError("Đường dẫn resume không được để trống!")

    # 1. Tự động sửa đường dẫn Kaggle nếu người dùng copy nhầm URL 'models/<user>/<slug>/'
    if not os.path.exists(resume_target):
        norm_path = resume_target.replace("\\", "/")
        if "/kaggle/input/" in norm_path:
            subparts = norm_path.split("/kaggle/input/")[1].split("/")
            if len(subparts) >= 3 and subparts[0] == "models":
                alt_path = os.path.join("/kaggle/input", *subparts[2:])
                if os.path.exists(alt_path):
                    print("💡 [Smart Path] Tự động chuẩn hóa đường dẫn Kaggle:")
                    print(f"   Từ: '{resume_target}'")
                    print(f"   Sang: '{alt_path}'")
                    resume_target = alt_path

    # 2. Nếu là thư mục, tự động quét tìm file checkpoint bên trong
    if os.path.isdir(resume_target):
        found_cand = None
        for candidate in ["last_checkpoint.pth", "dinov3_direction1_new_latest.pth", "dinov3_directionG_latest.pth", "best_checkpoint.pth"]:
            cand_path = os.path.join(resume_target, candidate)
            if os.path.isfile(cand_path):
                found_cand = cand_path
                break
        if found_cand is None:
            pths = glob.glob(os.path.join(resume_target, "**", "*.pth"), recursive=True)
            if pths:
                found_cand = pths[0]
        if found_cand:
            print(f"📂 [Smart Path] Đã tìm thấy checkpoint trong thư mục: {found_cand}")
            resume_target = found_cand
        else:
            raise FileNotFoundError(f"❌ [Resume Error] Thư mục '{resume_target}' không chứa bất kỳ file checkpoint (.pth) nào!")

    # 3. Báo lỗi rõ ràng nếu không tìm thấy file
    if not os.path.isfile(resume_target):
        pths_found = glob.glob("/kaggle/input/**/*.pth", recursive=True)[:5] if os.path.isdir("/kaggle/input") else []
        err_msg = (
            f"\n❌ [LỖI RESUME] Không tìm thấy file checkpoint tại: '{resume_target}'\n"
            f"   Nguyên nhân: Đường dẫn file không tồn tại trên hệ thống.\n"
        )
        if pths_found:
            err_msg += "   💡 Gợi ý các file .pth hiện có trong /kaggle/input:\n"
            for p in pths_found:
                err_msg += f"      - {p}\n"
        err_msg += "   👉 Vui lòng kiểm tra lại đường dẫn file checkpoint.\n"
        raise FileNotFoundError(err_msg)

    return resume_target
