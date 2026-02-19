"""IMS 베어링 데이터셋 로딩 및 테스트셋 메타데이터."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 테스트셋 메타데이터
# ---------------------------------------------------------------------------

_TEST_SETS: dict[str, dict] = {
    "1st_test": {
        "channels_per_file": 8,
        "description": "Test 1 — 8채널, Brg3 내륜결함, Brg4 전동체결함",
        "bearings": {
            "BRG-003": {
                "label": "Bearing 3",
                "channels": [4, 5],
                "fault_type": "inner_race",
            },
            "BRG-004": {
                "label": "Bearing 4",
                "channels": [6, 7],
                "fault_type": "rolling_element",
            },
        },
    },
    "2nd_test": {
        "channels_per_file": 4,
        "description": "Test 2 — 4채널, Brg1 외륜결함",
        "bearings": {
            "BRG-001": {
                "label": "Bearing 1",
                "channels": [0],
                "fault_type": "outer_race",
            },
        },
    },
    "3rd_test": {
        "channels_per_file": 4,
        "description": "Test 3 — 4채널, Brg3 외륜결함",
        "bearings": {
            "BRG-003": {
                "label": "Bearing 3",
                "channels": [2],
                "fault_type": "outer_race",
            },
        },
    },
}


def get_test_set_info(test_set_id: str) -> dict:
    """테스트셋 메타데이터 반환.

    Args:
        test_set_id: '1st_test', '2nd_test', '3rd_test' 중 하나.

    Returns:
        channels_per_file, description, bearings 포함 dict.

    Raises:
        KeyError: 존재하지 않는 test_set_id.
    """
    return _TEST_SETS[test_set_id]


def get_bearing_info(test_set_id: str, bearing_id: str) -> dict:
    """특정 테스트셋의 베어링 메타데이터 반환.

    Args:
        test_set_id: '1st_test', '2nd_test', '3rd_test' 중 하나.
        bearing_id: 'BRG-001', 'BRG-003', 'BRG-004' 등.

    Returns:
        label, channels, fault_type 포함 dict.

    Raises:
        KeyError: 존재하지 않는 test_set_id 또는 bearing_id.
    """
    return _TEST_SETS[test_set_id]["bearings"][bearing_id]


# ---------------------------------------------------------------------------
# 파일 로딩
# ---------------------------------------------------------------------------


def parse_timestamp(filename: str) -> datetime:
    """IMS 파일명을 datetime으로 파싱.

    Args:
        filename: '2003.10.22.12.06.24' 형식의 파일명.
    """
    return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")


def list_snapshots(test_dir: str | Path) -> list[tuple[datetime, Path]]:
    """테스트 디렉토리의 스냅샷 파일 목록을 타임스탬프 순으로 반환.

    Args:
        test_dir: IMS 테스트셋 디렉토리 경로.

    Returns:
        (datetime, Path) 튜플 리스트 (시간순 정렬).
    """
    test_dir = Path(test_dir)
    snapshots: list[tuple[datetime, Path]] = []
    for p in test_dir.iterdir():
        if p.is_file() and not p.name.startswith("."):
            try:
                ts = parse_timestamp(p.name)
                snapshots.append((ts, p))
            except ValueError:
                continue
    snapshots.sort(key=lambda x: x[0])
    return snapshots


def load_snapshot(path: str | Path, channels_per_file: int) -> np.ndarray:
    """스냅샷 파일을 ndarray로 로딩.

    IMS 데이터는 탭 구분 텍스트 파일이다.

    Args:
        path: 스냅샷 파일 경로.
        channels_per_file: 채널 수 (1st_test=8, 2nd/3rd_test=4).

    Returns:
        (samples, channels) shape의 ndarray.
    """
    data = np.loadtxt(path, delimiter="\t")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.shape[1] != channels_per_file:
        raise ValueError(
            f"Expected {channels_per_file} channels, got {data.shape[1]}"
        )
    return data


def get_bearing_channels(
    snapshot: np.ndarray, channel_indices: list[int]
) -> np.ndarray:
    """스냅샷에서 특정 베어링 채널 추출.

    Args:
        snapshot: (samples, channels) ndarray.
        channel_indices: 추출할 채널 인덱스 리스트.

    Returns:
        (samples, len(channel_indices)) ndarray.
    """
    return snapshot[:, channel_indices]
