"""pytest 루트 conftest — src 디렉토리를 sys.path에 추가."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def pytest_configure(config):
    """커스텀 마커 등록."""
    config.addinivalue_line(
        "markers",
        "e2e: Docker 기반 End-to-End 테스트 (실제 서비스 + OpenAI API 필요)",
    )
