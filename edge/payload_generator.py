"""이벤트 페이로드 생성기.

Edge 파이프라인 결과물(특징량, 이상감지, HI)을 메타데이터와 함께
Cloud 에이전트용 이벤트 페이로드 JSON으로 조립한다.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from edge.anomaly_detection import AnomalyResult
from edge.feature_pipeline import extract_snapshot_features
from edge.metadata import get_channel_indices, get_edge_node_id, get_equipment_meta


def build_event_payload(
    *,
    test_set_id: str,
    bearing_id: str,
    snapshot_timestamp: datetime,
    operation_start_date: datetime,
    features: dict[str, dict[str, dict]],
    anomaly_result: AnomalyResult,
    event_seq: int = 1,
) -> dict:
    """Edge 파이프라인 결과를 이벤트 페이로드로 조립.

    Args:
        test_set_id: IMS 테스트셋 ID.
        bearing_id: 베어링 ID.
        snapshot_timestamp: 스냅샷 타임스탬프.
        operation_start_date: 운전 시작일.
        features: extract_snapshot_features() 결과 (채널별 특징량).
        anomaly_result: detect_anomaly() 결과.
        event_seq: 이벤트 시퀀스 번호.

    Returns:
        페이로드 스키마에 맞는 dict.
    """
    # 운전 경과
    elapsed = snapshot_timestamp - operation_start_date
    days_elapsed = elapsed.days
    total_hours = elapsed.total_seconds() / 3600

    # 이벤트 메타
    date_str = snapshot_timestamp.strftime("%Y%m%d")
    event_id = f"EVT-{date_str}-{event_seq:04d}"
    event_type = "anomaly_alert" if anomaly_result.anomaly_detected else "periodic_monitoring"

    # equipment_meta
    equipment_meta = get_equipment_meta(test_set_id, bearing_id)
    equipment_meta["operation_days_elapsed"] = days_elapsed
    equipment_meta["total_running_hours"] = round(total_hours, 1)

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
        "snapshot_timestamp": snapshot_timestamp.isoformat(),
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
        "timestamp": snapshot_timestamp.isoformat(),
        "event_type": event_type,
        "edge_node_id": get_edge_node_id(test_set_id),
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
