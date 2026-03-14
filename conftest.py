"""pytest 루트 conftest — src 디렉토리를 sys.path에 추가."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
