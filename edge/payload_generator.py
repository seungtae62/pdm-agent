"""이벤트 페이로드 생성기.

Edge 파이프라인 결과물(특징량, 이상감지)을 기준정보 ID와 함께
Cloud 에이전트용 이벤트 페이로드 JSON으로 조립한다.

Edge는 설비/베어링/센서의 ID만 보유하며, 상세 메타데이터는
Cloud 서버에서 ID로 조회한다. 페이로드에는 기준정보 ID +
edge computing 결과(룰 검사, 모델 검사, 특징량)만 포함한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from edge.anomaly_detection import AnomalyResult


def build_event_payload(
    *,
    equipment_id: str,
    bearing_id: str,
    edge_node_id: str,
    timestamp: datetime,
    features: dict[str, dict[str, dict]],
    anomaly_result: AnomalyResult,
    event_seq: int = 1,
    sensor_channels: list[str] | None = None,
) -> dict:
    """Edge 파이프라인 결과를 이벤트 페이로드로 조립.

    페이로드에는 기준정보 ID와 edge computing 결과만 포함한다.
    설비/베어링/센서의 상세 메타데이터는 Cloud 서버에서 ID로 조회.

    Args:
        equipment_id: 설비 ID (예: 'IMS-TESTRIG-01').
        bearing_id: 베어링 ID (예: 'BRG-003').
        edge_node_id: Edge 노드 ID (예: 'EDGE-001').
        timestamp: 이벤트 타임스탬프 (스크립트 실행 시점).
        features: extract_snapshot_features() 결과 (채널별 특징량).
        anomaly_result: detect_anomaly() 결과.
        event_seq: 이벤트 시퀀스 번호.
        sensor_channels: 센서 채널 ID 목록 (예: ['ch0', 'ch1']).

    Returns:
        페이로드 dict.
    """
    date_str = timestamp.strftime("%Y%m%d")
    event_id = f"EVT-{date_str}-{event_seq:04d}"
    event_type = "anomaly_alert" if anomaly_result.anomaly_detected else "periodic_monitoring"

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
        "equipment_id": equipment_id,
        "bearing_id": bearing_id,
        "sensor_channels": sensor_channels or [],
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
