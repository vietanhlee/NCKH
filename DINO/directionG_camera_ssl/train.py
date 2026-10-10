"""
=============================================================================
 Alias: Direction G (Camera SSL / Vehicle-Centric SSL)
 Tự động liên kết và chuyển tiếp tới DINO/direction1_new/train.py
 Đảm bảo tương thích 100% với các lệnh gọi cũ directionG_camera_ssl/train.py
=============================================================================
"""

import os
import sys

_dino_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _dino_root not in sys.path:
    sys.path.insert(0, _dino_root)

from direction1_new.train import train_direction_g

if __name__ == "__main__":
    train_direction_g()
