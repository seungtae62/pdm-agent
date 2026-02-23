"""IMS 테스트 리그 기준정보 ID 매핑.

Edge는 설비/베어링/센서의 ID만 보유한다.
상세 메타데이터(사양, 결함 주파수 등)는 Cloud 서버에서 ID로 조회한다.
이 모듈은 IMS 테스트셋 → 기준정보 ID 매핑만 제공한다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 테스트셋 → 기준정보 ID 매핑
# ---------------------------------------------------------------------------

_EQUIPMENT_MAP: dict[str, dict] = {
    "1st_test": {
        "equipment_id": "IMS-TESTRIG-01",
        "edge_node_id": "EDGE-001",
        "bearings": {
            "BRG-003": {
                "bearing_id": "BRG-003",
                "channels": ["ch0", "ch1"],
                "channel_indices": [4, 5],
            },
            "BRG-004": {
                "bearing_id": "BRG-004",
                "channels": ["ch0", "ch1"],
                "channel_indices": [6, 7],
            },
        },
    },
    "2nd_test": {
        "equipment_id": "IMS-TESTRIG-02",
        "edge_node_id": "EDGE-002",
        "bearings": {
            "BRG-001": {
                "bearing_id": "BRG-001",
                "channels": ["ch0"],
                "channel_indices": [0],
            },
        },
    },
    "3rd_test": {
        "equipment_id": "IMS-TESTRIG-03",
        "edge_node_id": "EDGE-003",
        "bearings": {
            "BRG-003": {
                "bearing_id": "BRG-003",
                "channels": ["ch0"],
                "channel_indices": [2],
            },
        },
    },
}


def get_equipment_id(test_set_id: str) -> str:
    """테스트셋의 설비 ID 반환."""
    return _EQUIPMENT_MAP[test_set_id]["equipment_id"]


def get_edge_node_id(test_set_id: str) -> str:
    """테스트셋의 Edge 노드 ID 반환."""
    return _EQUIPMENT_MAP[test_set_id]["edge_node_id"]


def get_bearing_id(test_set_id: str, bearing_id: str) -> str:
    """베어링 ID 반환 (검증용)."""
    return _EQUIPMENT_MAP[test_set_id]["bearings"][bearing_id]["bearing_id"]


def get_channel_indices(test_set_id: str, bearing_id: str) -> list[int]:
    """테스트셋+베어링의 채널 인덱스 반환 (데이터 로딩용)."""
    return _EQUIPMENT_MAP[test_set_id]["bearings"][bearing_id]["channel_indices"]


def get_sensor_channels(test_set_id: str, bearing_id: str) -> list[str]:
    """테스트셋+베어링의 센서 채널 ID 목록 반환."""
    return _EQUIPMENT_MAP[test_set_id]["bearings"][bearing_id]["channels"]
