"""설비(Equipment) 기준정보.

Cloud 서버에서 equipment_id로 조회하는 설비 메타데이터.
"""

from __future__ import annotations

EQUIPMENT: dict[str, dict] = {
    "IMS-TESTRIG-01": {
        "equipment_id": "IMS-TESTRIG-01",
        "equipment_name": "IMS Bearing Test Rig #1",
        "location": "University of Cincinnati, NSF I/UCRC",
        "shaft_rpm": 2000,
        "radial_load_lbs": 6000,
        "operation_start_date": "2003-10-22",
        "description": "NASA IMS Test Set 1 — 8채널, 4베어링, 35일 운전",
    },
    "IMS-TESTRIG-02": {
        "equipment_id": "IMS-TESTRIG-02",
        "equipment_name": "IMS Bearing Test Rig #2",
        "location": "University of Cincinnati, NSF I/UCRC",
        "shaft_rpm": 2000,
        "radial_load_lbs": 6000,
        "operation_start_date": "2004-02-12",
        "description": "NASA IMS Test Set 2 — 4채널, 4베어링, 7일 운전 (급속 열화)",
    },
    "IMS-TESTRIG-03": {
        "equipment_id": "IMS-TESTRIG-03",
        "equipment_name": "IMS Bearing Test Rig #3",
        "location": "University of Cincinnati, NSF I/UCRC",
        "shaft_rpm": 2000,
        "radial_load_lbs": 6000,
        "operation_start_date": "2004-03-04",
        "description": "NASA IMS Test Set 3 — 4채널, 4베어링, 30일 운전",
    },
}


def get_equipment(equipment_id: str) -> dict:
    """설비 기준정보 조회.

    Args:
        equipment_id: 설비 ID (예: 'IMS-TESTRIG-01').

    Returns:
        설비 메타데이터 dict.

    Raises:
        KeyError: 존재하지 않는 equipment_id.
    """
    return EQUIPMENT[equipment_id]
