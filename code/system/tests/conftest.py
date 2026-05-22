from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SYSTEM_ROOT = REPO_ROOT / "code" / "system"
MTF_BA_ROOT = REPO_ROOT / "code" / "anomaly_detection" / "individual"

for path in (SYSTEM_ROOT, MTF_BA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
