"""베어링(Bearing) 기준정보.

Cloud 서버에서 bearing_id로 조회하는 베어링 메타데이터.
결함 특성 주파수, 사양, 설치 정보 등을 포함한다.
"""

from __future__ import annotations

BEARINGS: dict[str, dict] = {
    # --- IMS-TESTRIG-01 ---
    "IMS-TESTRIG-01/BRG-001": {
        "bearing_id": "BRG-001",
        "equipment_id": "IMS-TESTRIG-01",
        "position": "Bearing 1",
        "model": "Rexnord ZA-2115",
        "type": "Double Row Bearing",
        "rolling_elements_count": 16,
        "ball_diameter_inch": 0.331,
        "pitch_diameter_inch": 2.815,
        "contact_angle_deg": 15.17,
        "install_date": "2003-10-22",
        "known_fault_type": None,
        "defect_frequencies_hz": {
            "BPFO": 236.4,
            "BPFI": 296.9,
            "BSF": 141.1,
            "FTF": 14.8,
        },
    },
    "IMS-TESTRIG-01/BRG-002": {
        "bearing_id": "BRG-002",
        "equipment_id": "IMS-TESTRIG-01",
        "position": "Bearing 2",
        "model": "Rexnord ZA-2115",
        "type": "Double Row Bearing",
        "rolling_elements_count": 16,
        "ball_diameter_inch": 0.331,
        "pitch_diameter_inch": 2.815,
        "contact_angle_deg": 15.17,
        "install_date": "2003-10-22",
        "known_fault_type": None,
        "defect_frequencies_hz": {
            "BPFO": 236.4,
            "BPFI": 296.9,
            "BSF": 141.1,
            "FTF": 14.8,
        },
    },
    "IMS-TESTRIG-01/BRG-003": {
        "bearing_id": "BRG-003",
        "equipment_id": "IMS-TESTRIG-01",
        "position": "Bearing 3",
        "model": "Rexnord ZA-2115",
        "type": "Double Row Bearing",
        "rolling_elements_count": 16,
        "ball_diameter_inch": 0.331,
        "pitch_diameter_inch": 2.815,
        "contact_angle_deg": 15.17,
        "install_date": "2003-10-22",
        "known_fault_type": "inner_race",
        "defect_frequencies_hz": {
            "BPFO": 236.4,
            "BPFI": 296.9,
            "BSF": 141.1,
            "FTF": 14.8,
        },
    },
    "IMS-TESTRIG-01/BRG-004": {
        "bearing_id": "BRG-004",
        "equipment_id": "IMS-TESTRIG-01",
        "position": "Bearing 4",
        "model": "Rexnord ZA-2115",
        "type": "Double Row Bearing",
        "rolling_elements_count": 16,
        "ball_diameter_inch": 0.331,
        "pitch_diameter_inch": 2.815,
        "contact_angle_deg": 15.17,
        "install_date": "2003-10-22",
        "known_fault_type": "rolling_element",
        "defect_frequencies_hz": {
            "BPFO": 236.4,
            "BPFI": 296.9,
            "BSF": 141.1,
            "FTF": 14.8,
        },
    },
    # --- IMS-TESTRIG-02 ---
    "IMS-TESTRIG-02/BRG-001": {
        "bearing_id": "BRG-001",
        "equipment_id": "IMS-TESTRIG-02",
        "position": "Bearing 1",
        "model": "Rexnord ZA-2115",
        "type": "Double Row Bearing",
        "rolling_elements_count": 16,
        "ball_diameter_inch": 0.331,
        "pitch_diameter_inch": 2.815,
        "contact_angle_deg": 15.17,
        "install_date": "2004-02-12",
        "known_fault_type": "outer_race",
        "defect_frequencies_hz": {
            "BPFO": 236.4,
            "BPFI": 296.9,
            "BSF": 141.1,
            "FTF": 14.8,
        },
    },
    # --- IMS-TESTRIG-03 ---
    "IMS-TESTRIG-03/BRG-003": {
        "bearing_id": "BRG-003",
        "equipment_id": "IMS-TESTRIG-03",
        "position": "Bearing 3",
        "model": "Rexnord ZA-2115",
        "type": "Double Row Bearing",
        "rolling_elements_count": 16,
        "ball_diameter_inch": 0.331,
        "pitch_diameter_inch": 2.815,
        "contact_angle_deg": 15.17,
        "install_date": "2004-03-04",
        "known_fault_type": "outer_race",
        "defect_frequencies_hz": {
            "BPFO": 236.4,
            "BPFI": 296.9,
            "BSF": 141.1,
            "FTF": 14.8,
        },
    },
}


def get_bearing(equipment_id: str, bearing_id: str) -> dict:
    """베어링 기준정보 조회.

    Args:
        equipment_id: 설비 ID (예: 'IMS-TESTRIG-01').
        bearing_id: 베어링 ID (예: 'BRG-003').

    Returns:
        베어링 메타데이터 dict.

    Raises:
        KeyError: 존재하지 않는 조합.
    """
    return BEARINGS[f"{equipment_id}/{bearing_id}"]


def list_bearings(equipment_id: str) -> list[dict]:
    """특정 설비의 전체 베어링 목록 반환."""
    return [
        v for k, v in BEARINGS.items()
        if k.startswith(f"{equipment_id}/")
    ]
