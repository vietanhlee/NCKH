"""
=============================================================================
 DINO Common Logging Module — Multi-Stream Production Logger
 Tự động đồng bộ toàn bộ console output (stdout/stderr) và Python logging
 ra file log vật lý (train.log) trong thư mục lưu trữ checkpoint (save_dir).
 
 Đặc tính chuẩn Production:
   1. Ghi đồng thời ra console và file mà không làm gián đoạn luồng stdout/stderr.
   2. Hỗ trợ chuẩn mã hóa UTF-8 (tiếng Việt, emoji 🚀🔥📊, ký tự đặc biệt).
   3. Tự động làm sạch mã màu ANSI (ANSI escape codes) khi ghi file giúp log sạch,
      dễ đọc trên mọi trình soạn thảo văn bản (Notepad, nano, VS Code).
   4. Tự động flush đảm bảo không mất log ngay cả khi tiến trình bị kill đột ngột.
   5. An toàn tuyệt đối với cơ chế khôi phục stream gốc qua atexit / context manager.
=============================================================================
"""

import atexit
import datetime
import io
import logging
import os
import re
import sys
from typing import Optional, TextIO, Union

# Regex loại bỏ mã màu ANSI trong terminal khi ghi ra file phẳng
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Biến toàn cục theo dõi trạng thái logger hiện tại
_ORIGINAL_STDOUT: Optional[TextIO] = None
_ORIGINAL_STDERR: Optional[TextIO] = None
_CURRENT_LOG_FILE: Optional[io.TextIOWrapper] = None
_CURRENT_LOG_PATH: Optional[str] = None


class TeeStream:
    """
    Stream bọc (wrapper) ghi đồng thời dữ liệu vào stream gốc (console stdout/stderr)
    và file log vật lý trên đĩa.
    """
    def __init__(self, original_stream: TextIO, file_stream: TextIO, is_stderr: bool = False):
        self.original_stream = original_stream
        self.file_stream = file_stream
        self.is_stderr = is_stderr
        self._closed = False

    def write(self, data: str) -> int:
        if not data:
            return 0

        # 1. Ghi vào file log (lọc bỏ mã màu ANSI để file text sạch sẽ, chuẩn báo cáo)
        if not self._closed and self.file_stream and not self.file_stream.closed:
            try:
                clean_text = ANSI_ESCAPE_PATTERN.sub("", data)
                self.file_stream.write(clean_text)
                # Tự động flush khi gặp ký tự xuống dòng
                if "\n" in data or "\r" in data:
                    self.file_stream.flush()
            except Exception:
                pass

        # 2. Ghi ra stream console ban đầu
        try:
            return self.original_stream.write(data)
        except UnicodeEncodeError:
            # Fallback an toàn nếu terminal Windows cũ không hỗ trợ Unicode
            try:
                return self.original_stream.write(data.encode("ascii", "replace").decode("ascii"))
            except Exception:
                return len(data)
        except Exception:
            return len(data)

    def flush(self) -> None:
        if not self._closed and self.file_stream and not self.file_stream.closed:
            try:
                self.file_stream.flush()
            except Exception:
                pass
        try:
            self.original_stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return getattr(self.original_stream, "isatty", lambda: False)()

    def fileno(self) -> int:
        return getattr(self.original_stream, "fileno", lambda: 1)()

    def close(self) -> None:
        self._closed = True

    def __getattr__(self, name: str):
        return getattr(self.original_stream, name)


def setup_training_logger(
    save_dir: str,
    log_file: Optional[str] = None,
    session_title: str = "HUẤN LUYỆN MÔ HÌNH",
    append: bool = True,
    verbose: bool = True,
) -> str:
    """
    Kích hoạt hệ thống tự động lưu toàn bộ log của phiên huấn luyện vào file.

    Args:
        save_dir: Thư mục lưu checkpoint và tài nguyên (sẽ được tự động tạo).
        log_file: Tên hoặc đường dẫn file log cụ thể (mặc định: 'train.log' bên trong save_dir).
        session_title: Tiêu đề phiên huấn luyện để ghi dấu mốc trong file log.
        append: True để ghi nối tiếp (khi resume), False để ghi đè từ đầu.
        verbose: In thông báo xác nhận kích hoạt logger.

    Returns:
        Đường dẫn tuyệt đối đến file log đang hoạt động.
    """
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _CURRENT_LOG_FILE, _CURRENT_LOG_PATH

    os.makedirs(save_dir, exist_ok=True)

    if log_file:
        if os.path.isabs(log_file):
            log_path = log_file
        else:
            log_path = os.path.join(save_dir, log_file)
    else:
        log_path = os.path.join(save_dir, "train.log")

    log_path = os.path.abspath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Nếu đã cấu hình tới đúng file log này thì tiếp tục sử dụng
    if _CURRENT_LOG_PATH == log_path and _CURRENT_LOG_FILE and not _CURRENT_LOG_FILE.closed:
        return log_path

    # Nếu đang có logger khác thì đóng stream cũ
    close_training_logger()

    # Mở file log vật lý ở chế độ UTF-8
    mode = "a" if append else "w"
    file_stream = open(log_path, mode=mode, encoding="utf-8", buffering=1)

    # Lưu lại stdout/stderr gốc
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    _CURRENT_LOG_FILE = file_stream
    _CURRENT_LOG_PATH = log_path

    # Chuyển hướng stdout và stderr qua TeeStream
    sys.stdout = TeeStream(_ORIGINAL_STDOUT, file_stream, is_stderr=False)  # type: ignore
    sys.stderr = TeeStream(_ORIGINAL_STDERR, file_stream, is_stderr=True)   # type: ignore

    # Đồng bộ cấu hình Python standard logging
    root_logger = logging.getLogger()
    # Kiểm tra xem đã có FileHandler tới log_path này chưa
    has_handler = any(
        isinstance(h, logging.FileHandler) and os.path.abspath(h.baseFilename) == log_path
        for h in root_logger.handlers
    )
    if not has_handler:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        root_logger.addHandler(file_handler)

    # Ghi banner đánh dấu phiên huấn luyện
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    banner = (
        f"\n{'=' * 82}\n"
        f" 📝 [TRAINING LOG SESSION] {session_title}\n"
        f"    ⏰ Thời gian bắt đầu: {start_time_str}\n"
        f"    📁 Thư mục lưu trữ  : {save_dir}\n"
        f"    📄 File nhật ký log : {log_path}\n"
        f"{'=' * 82}\n"
    )
    print(banner)

    # Đăng ký hook dọn dẹp khi tiến trình thoát
    atexit.register(close_training_logger)

    if verbose:
        print(f"✅ [Logger] Đã kích hoạt cơ chế tự động ghi log song song vào: {log_path}")

    return log_path


def close_training_logger() -> None:
    """
    Khôi phục lại stdout và stderr nguyên bản, hoàn tất và đóng file log an toàn.
    """
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _CURRENT_LOG_FILE, _CURRENT_LOG_PATH

    if _CURRENT_LOG_FILE and not _CURRENT_LOG_FILE.closed:
        end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            footer = f"\n{'=' * 82}\n 🏁 [TRAINING LOG SESSION] KẾT THÚC PHIÊN: {end_time_str}\n{'=' * 82}\n"
            _CURRENT_LOG_FILE.write(footer)
            _CURRENT_LOG_FILE.flush()
        except Exception:
            pass
        try:
            _CURRENT_LOG_FILE.close()
        except Exception:
            pass

    if _ORIGINAL_STDOUT is not None:
        sys.stdout = _ORIGINAL_STDOUT
        _ORIGINAL_STDOUT = None

    if _ORIGINAL_STDERR is not None:
        sys.stderr = _ORIGINAL_STDERR
        _ORIGINAL_STDERR = None

    _CURRENT_LOG_FILE = None
    _CURRENT_LOG_PATH = None
