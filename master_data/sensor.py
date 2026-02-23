"""센서(Sensor) 기준정보.

Cloud 서버에서 sensor_id 또는 (equipment_id, bearing_id)로 조회하는
센서 배치 및 사양 정보.
"""

from __future__ import annotations

SENSORS: dict[str, dict] = {
    # --- IMS-TESTRIG-01 / BRG-003 ---
    "IMS-TESTRIG-01/BRG-003/ch0": {
        "sensor_id": "IMS-TESTRIG-01/BRG-003/ch0",
        "equipment_id": "IMS-TESTRIG-01",
        "bearing_id": "BRG-003",
        "channel": "ch0",
        "channel_index": 4,
        "sensor_type": "PCB 353B33 High Sensitivity Accelerometer",
        "sampling_rate_hz": 20000,
        "samples_per_snapshot": 20480,
        "snapshot_interval_min": 10,
        "mounting_direction": "vertical",
    },
    "IMS-TESTRIG-01/BRG-003/ch1": {
        "sensor_id": "IMS-TESTRIG-01/BRG-003/ch1",
        "equipment_id": "IMS-TESTRIG-01",
        "bearing_id": "BRG-003",
        "channel": "ch1",
        "channel_index": 5,
        "sensor_type": "PCB 353B33 High Sensitivity Accelerometer",
        "sampling_rate_hz": 20000,
        "samples_per_snapshot": 20480,
        "snapshot_interval_min": 10,
        "mounting_direction": "horizontal",
    },
    # --- IMS-TESTRIG-01 / BRG-004 ---
    "IMS-TESTRIG-01/BRG-004/ch0": {
        "sensor_id": "IMS-TESTRIG-01/BRG-004/ch0",
        "equipment_id": "IMS-TESTRIG-01",
        "bearing_id": "BRG-004",
        "channel": "ch0",
        "channel_index": 6,
        "sensor_type": "PCB 353B33 High Sensitivity Accelerometer",
        "sampling_rate_hz": 20000,
        "samples_per_snapshot": 20480,
        "snapshot_interval_min": 10,
        "mounting_direction": "vertical",
    },
    "IMS-TESTRIG-01/BRG-004/ch1": {
        "sensor_id": "IMS-TESTRIG-01/BRG-004/ch1",
        "equipment_id": "IMS-TESTRIG-01",
        "bearing_id": "BRG-004",
        "channel": "ch1",
        "channel_index": 7,
        "sensor_type": "PCB 353B33 High Sensitivity Accelerometer",
        "sampling_rate_hz": 20000,
        "samples_per_snapshot": 20480,
        "snapshot_interval_min": 10,
        "mounting_direction": "horizontal",
    },
    # --- IMS-TESTRIG-02 / BRG-001 ---
    "IMS-TESTRIG-02/BRG-001/ch0": {
        "sensor_id": "IMS-TESTRIG-02/BRG-001/ch0",
        "equipment_id": "IMS-TESTRIG-02",
        "bearing_id": "BRG-001",
        "channel": "ch0",
        "channel_index": 0,
        "sensor_type": "PCB 353B33 High Sensitivity Accelerometer",
        "sampling_rate_hz": 20000,
        "samples_per_snapshot": 20480,
        "snapshot_interval_min": 10,
        "mounting_direction": "vertical",
    },
    # --- IMS-TESTRIG-03 / BRG-003 ---
    "IMS-TESTRIG-03/BRG-003/ch0": {
        "sensor_id": "IMS-TESTRIG-03/BRG-003/ch0",
        "equipment_id": "IMS-TESTRIG-03",
        "bearing_id": "BRG-003",
        "channel": "ch0",
        "channel_index": 2,
        "sensor_type": "PCB 353B33 High Sensitivity Accelerometer",
        "sampling_rate_hz": 20000,
        "samples_per_snapshot": 20480,
        "snapshot_interval_min": 10,
        "mounting_direction": "vertical",
    },
}


def get_sensor(equipment_id: str, bearing_id: str, channel: str) -> dict:
    """센서 기준정보 조회.

    Args:
        equipment_id: 설비 ID.
        bearing_id: 베어링 ID.
        channel: 채널 ID (예: 'ch0').

    Returns:
        센서 메타데이터 dict.

    Raises:
        KeyError: 존재하지 않는 조합.
    """
    return SENSORS[f"{equipment_id}/{bearing_id}/{channel}"]


def list_sensors(equipment_id: str, bearing_id: str) -> list[dict]:
    """특정 베어링의 전체 센서 목록 반환."""
    prefix = f"{equipment_id}/{bearing_id}/"
    return [v for k, v in SENSORS.items() if k.startswith(prefix)]
