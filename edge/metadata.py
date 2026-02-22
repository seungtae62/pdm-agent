"""IMS 테스트 리그 설비/베어링/센서 메타데이터.

이벤트 페이로드의 equipment_meta 섹션을 구성하기 위한 정적 데이터.
"""

from __future__ import annotations

from edge.config import DEFECT_FREQUENCIES_HZ, SAMPLING_RATE_HZ, SAMPLES_PER_SNAPSHOT

# ---------------------------------------------------------------------------
# 베어링 공통 사양 (Rexnord ZA-2115)
# ---------------------------------------------------------------------------

BEARING_SPEC: dict = {
    "model": "Rexnord ZA-2115",
    "type": "Double Row Bearing",
    "rolling_elements_count": 16,
    "ball_diameter_inch": 0.331,
    "pitch_diameter_inch": 2.815,
    "contact_angle_deg": 15.17,
    "defect_frequencies_hz": DEFECT_FREQUENCIES_HZ,
}

# ---------------------------------------------------------------------------
# 센서 공통 사양
# ---------------------------------------------------------------------------

SENSOR_SPEC: dict = {
    "sensor_type": "PCB 353B33 High Sensitivity Accelerometer",
    "sampling_rate_hz": SAMPLING_RATE_HZ,
    "samples_per_snapshot": SAMPLES_PER_SNAPSHOT,
    "snapshot_interval_min": 10,
}

# ---------------------------------------------------------------------------
# 테스트 리그별 메타데이터
# ---------------------------------------------------------------------------

_EQUIPMENT_META: dict[str, dict] = {
    "1st_test": {
        "equipment_id": "IMS-TESTRIG-01",
        "equipment_name": "IMS Bearing Test Rig #1",
        "location": "University of Cincinnati, NSF I/UCRC",
        "shaft_rpm": 2000,
        "radial_load_lbs": 6000,
        "operation_start_date": "2003-10-22",
        "edge_node_id": "EDGE-001",
        "sensor_count": 8,
        "bearings": {
            "BRG-003": {
                "bearing_id": "BRG-003",
                "position": "Bearing 3",
                "channels": ["ch0", "ch1"],
                "channel_indices": [4, 5],
                "install_date": "2003-10-22",
            },
            "BRG-004": {
                "bearing_id": "BRG-004",
                "position": "Bearing 4",
                "channels": ["ch0", "ch1"],
                "channel_indices": [6, 7],
                "install_date": "2003-10-22",
            },
        },
    },
    "2nd_test": {
        "equipment_id": "IMS-TESTRIG-02",
        "equipment_name": "IMS Bearing Test Rig #2",
        "location": "University of Cincinnati, NSF I/UCRC",
        "shaft_rpm": 2000,
        "radial_load_lbs": 6000,
        "operation_start_date": "2004-02-12",
        "edge_node_id": "EDGE-002",
        "sensor_count": 4,
        "bearings": {
            "BRG-001": {
                "bearing_id": "BRG-001",
                "position": "Bearing 1",
                "channels": ["ch0"],
                "channel_indices": [0],
                "install_date": "2004-02-12",
            },
        },
    },
    "3rd_test": {
        "equipment_id": "IMS-TESTRIG-03",
        "equipment_name": "IMS Bearing Test Rig #3",
        "location": "University of Cincinnati, NSF I/UCRC",
        "shaft_rpm": 2000,
        "radial_load_lbs": 6000,
        "operation_start_date": "2004-03-04",
        "edge_node_id": "EDGE-003",
        "sensor_count": 4,
        "bearings": {
            "BRG-003": {
                "bearing_id": "BRG-003",
                "position": "Bearing 3",
                "channels": ["ch0"],
                "channel_indices": [2],
                "install_date": "2004-03-04",
            },
        },
    },
}


def get_equipment_meta(test_set_id: str, bearing_id: str) -> dict:
    """테스트셋+베어링에 대한 equipment_meta 페이로드 섹션 생성.

    Args:
        test_set_id: '1st_test', '2nd_test', '3rd_test'.
        bearing_id: 'BRG-001', 'BRG-003', 'BRG-004'.

    Returns:
        equipment_meta dict (페이로드 스키마 형식).

    Raises:
        KeyError: 존재하지 않는 test_set_id 또는 bearing_id.
    """
    equip = _EQUIPMENT_META[test_set_id]
    brg = equip["bearings"][bearing_id]

    return {
        "equipment_id": equip["equipment_id"],
        "equipment_name": equip["equipment_name"],
        "location": equip["location"],
        "shaft_rpm": equip["shaft_rpm"],
        "radial_load_lbs": equip["radial_load_lbs"],
        "operation_start_date": equip["operation_start_date"],
        "bearing": {
            "bearing_id": brg["bearing_id"],
            "position": brg["position"],
            "install_date": brg["install_date"],
            **BEARING_SPEC,
            "last_maintenance_date": None,
        },
        "sensor_config": {
            "sensor_count": equip["sensor_count"],
            "channels": brg["channels"],
            **SENSOR_SPEC,
        },
    }


def get_edge_node_id(test_set_id: str) -> str:
    """테스트셋의 Edge 노드 ID 반환."""
    return _EQUIPMENT_META[test_set_id]["edge_node_id"]


def get_channel_indices(test_set_id: str, bearing_id: str) -> list[int]:
    """테스트셋+베어링의 채널 인덱스 반환."""
    return _EQUIPMENT_META[test_set_id]["bearings"][bearing_id]["channel_indices"]
