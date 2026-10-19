"""
Unit tests cho hệ thống tự động ghi nhật ký huấn luyện (Training Logger).
Kiểm tra tính năng lưu đồng thời console output ra file log, hỗ trợ UTF-8 và emoji.
"""

import os
import shutil
import sys
import tempfile
import pytest

from common.logger import (
    setup_training_logger,
    close_training_logger,
    TeeStream,
    ANSI_ESCAPE_PATTERN,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_dino_logger_")
    yield d
    try:
        close_training_logger()
        shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass


def test_logger_creates_file_and_logs(temp_dir):
    """Kiểm tra logger tạo file log và ghi chính xác nội dung print."""
    log_path = setup_training_logger(
        save_dir=temp_dir,
        log_file="test_train.log",
        session_title="TEST SESSION",
        verbose=False,
    )

    assert os.path.isfile(log_path), "File log phải được tạo trên đĩa."

    # In thử nghiệm tiếng Việt có dấu và emoji
    test_msg_vn = "🔥 [Huấn luyện] Thử nghiệm ghi log tiếng Việt UTF-8 chuẩn production."
    print(test_msg_vn)
    sys.stdout.flush()

    close_training_logger()

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "TEST SESSION" in content
    assert test_msg_vn in content
    assert "KẾT THÚC PHIÊN" in content


def test_logger_cleans_ansi_codes(temp_dir):
    """Kiểm tra logger tự động làm sạch mã màu ANSI trong file log."""
    log_path = setup_training_logger(
        save_dir=temp_dir,
        log_file="clean_ansi.log",
        session_title="ANSI CLEAN TEST",
        verbose=False,
    )

    colored_text = "\x1b[32m[SUCCESS]\x1b[0m Huấn luyện epoch hoàn tất!"
    print(colored_text)
    sys.stdout.flush()

    close_training_logger()

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "\x1b[" not in content, "File log không được chứa mã thoát ANSI."
    assert "[SUCCESS] Huấn luyện epoch hoàn tất!" in content


def test_logger_append_mode(temp_dir):
    """Kiểm tra chế độ append tiếp tục lưu khi resume phiên mới mà không xóa log cũ."""
    # Phiên 1
    setup_training_logger(save_dir=temp_dir, log_file="append.log", session_title="SESSION 1", verbose=False)
    print("Epoch 1 / 10: Loss = 2.500")
    close_training_logger()

    # Phiên 2 (Resume)
    setup_training_logger(save_dir=temp_dir, log_file="append.log", session_title="SESSION 2 (RESUME)", append=True, verbose=False)
    print("Epoch 2 / 10: Loss = 2.100")
    close_training_logger()

    log_path = os.path.join(temp_dir, "append.log")
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "SESSION 1" in content
    assert "Epoch 1 / 10: Loss = 2.500" in content
    assert "SESSION 2 (RESUME)" in content
    assert "Epoch 2 / 10: Loss = 2.100" in content
