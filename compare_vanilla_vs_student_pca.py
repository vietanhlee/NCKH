import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
_dino_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DINO")
if os.path.isdir(_dino_dir) and _dino_dir not in sys.path:
    sys.path.insert(0, _dino_dir)

from direction1_new.compare_vanilla_vs_student_pca import main

if __name__ == "__main__":
    main()
