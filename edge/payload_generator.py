"""이벤트 페이로드 생성기.

Edge 파이프라인 결과물(특징량, 이상감지)과 master_data 기준정보를
Cloud 에이전트용 이벤트 페이로드 JSON으로 조립한다.

기준정보 ID로 master_data에서 설비/베어링/센서 메타데이터를 조회하여
equipment_meta 중첩 구조를 구성하고, edge computing 결과와 함께 반환한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from edge.anomaly_detection import AnomalyResult
from master_data.bearing import get_bearing
from master_data.equipment import get_equipment
from master_data.sensor import list_sensors


def build_event_payload(
    *,
    equipment_id: str,
    bearing_id: str,
    edge_node_id: str,
    timestamp: datetime,
    features: dict[str, dict[str, dict]],
    anomaly_result: AnomalyResult,
    operation_days_elapsed: int,
    total_running_hours: float,
    event_seq: int = 1,
) -> dict:
    """Edge 파이프라인 결과를 이벤트 페이로드로 조립.

    기준정보 ID로 master_data에서 설비/베어링/센서 메타데이터를 조회하여
    equipment_meta 중첩 구조를 구성한다.

    Args:
        equipment_id: 설비 ID (예: 'IMS-TESTRIG-01').
        bearing_id: 베어링 ID (예: 'BRG-003').
        edge_node_id: Edge 노드 ID (예: 'EDGE-001').
        timestamp: 이벤트 타임스탬프 (스크립트 실행 시점).
        features: extract_snapshot_features() 결과 (채널별 특징량).
        anomaly_result: detect_anomaly() 결과.
        operation_days_elapsed: 운전 경과 일수.
        total_running_hours: 총 운전 시간.
        event_seq: 이벤트 시퀀스 번호.

    Returns:
        페이로드 dict.
    """
    date_str = timestamp.strftime("%Y%m%d")
    event_id = f"EVT-{date_str}-{event_seq:04d}"
    event_type = "anomaly_alert" if anomaly_result.anomaly_detected else "periodic_monitoring"

    # equipment_meta 조립
    equipment = get_equipment(equipment_id)
    bearing = get_bearing(equipment_id, bearing_id)
    sensors = list_sensors(equipment_id, bearing_id)

    equipment_meta = {
        "equipment_id": equipment["equipment_id"],
        "equipment_name": equipment["equipment_name"],
        "location": equipment["location"],
        "shaft_rpm": equipment["shaft_rpm"],
        "radial_load_lbs": equipment["radial_load_lbs"],
        "operation_start_date": equipment["operation_start_date"],
        "bearing": {
            "bearing_id": bearing["bearing_id"],
            "position": bearing["position"],
            "install_date": bearing["install_date"],
            "model": bearing["model"],
            "type": bearing["type"],
            "rolling_elements_count": bearing["rolling_elements_count"],
            "ball_diameter_inch": bearing["ball_diameter_inch"],
            "pitch_diameter_inch": bearing["pitch_diameter_inch"],
            "contact_angle_deg": bearing["contact_angle_deg"],
            "defect_frequencies_hz": bearing["defect_frequencies_hz"],
        },
        "sensor_config": {
            "sensor_count": len(sensors),
            "channels": [s["channel"] for s in sensors],
            "sensor_type": sensors[0]["sensor_type"],
            "sampling_rate_hz": sensors[0]["sampling_rate_hz"],
            "samples_per_snapshot": sensors[0]["samples_per_snapshot"],
            "snapshot_interval_min": sensors[0]["snapshot_interval_min"],
        },
        "operation_days_elapsed": operation_days_elapsed,
        "total_running_hours": total_running_hours,
    }

    # anomaly_detection_result
    anomaly_section = {
        "model_id": anomaly_result.model_id,
        "anomaly_detected": anomaly_result.anomaly_detected,
        "anomaly_score": round(anomaly_result.anomaly_score, 4),
        "anomaly_threshold": anomaly_result.anomaly_threshold,
        "health_state": anomaly_result.health_state,
        "confidence": round(anomaly_result.confidence, 4),
    }

    # current_features (채널별)
    current_features = {
        "snapshot_timestamp": timestamp.isoformat(),
        "time_domain": _round_dict(features["time_domain"]),
        "frequency_domain": _round_dict(features["frequency_domain"]),
    }

    # ml_rul_prediction (PoC: 미구현)
    ml_rul = {
        "model_id": "rul_v1",
        "predicted_rul_hours": None,
        "confidence_interval_hours": {"lower": None, "upper": None},
        "prediction_status": "not_applicable",
        "reason": "RUL model not deployed in current PoC",
    }

    return {
        "event_id": event_id,
        "timestamp": timestamp.isoformat(),
        "event_type": event_type,
        "edge_node_id": edge_node_id,
        "equipment_meta": equipment_meta,
        "anomaly_detection_result": anomaly_section,
        "current_features": current_features,
        "ml_rul_prediction": ml_rul,
    }


def save_payload(payload: dict, output_path: str | Path) -> Path:
    """페이로드를 JSON 파일로 저장.

    Args:
        payload: build_event_payload() 결과.
        output_path: 출력 파일 경로.

    Returns:
        저장된 파일 Path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return output_path


# ---------------------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------------------


def _round_dict(d: dict, ndigits: int = 6) -> dict:
    """중첩 dict의 float 값을 반올림."""
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _round_dict(value, ndigits)
        elif isinstance(value, float):
            result[key] = round(value, ndigits)
        else:
            result[key] = value
    return result
