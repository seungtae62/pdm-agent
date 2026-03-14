"""save_memory 노드 — 판단 결과 저장.

diagnosis_result를 PostgreSQL에 저장하고,
Qdrant analysis_history에 벡터 적재한다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from agent.memory.store import MemoryStore
from agent.state import PdMAgentState

logger = logging.getLogger(__name__)


def save_memory(
    state: PdMAgentState,
    *,
    store: MemoryStore | None = None,
) -> dict:
    """Memory 저장 노드.

    Args:
        state: 현재 State.
        store: MemoryStore 인스턴스. None이면 저장 건너뜀.

    Returns:
        State 업데이트 dict (변경 없음).
    """
    payload = state.get("event_payload", {})
    diagnosis = state.get("diagnosis_result", {})

    equipment_meta = payload.get("equipment_meta", {})
    equipment_id = equipment_meta.get("equipment_id", "")
    bearing_info = equipment_meta.get("bearing", {})
    bearing_id = bearing_info.get("bearing_id", "")
    event_id = payload.get("event_id", "")
    timestamp_str = payload.get("timestamp", "")

    # 타임스탬프 파싱
    try:
        event_timestamp = datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        event_timestamp = datetime.now()

    # Tool 사용 기록 수집
    tools_used = []
    for msg in state.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc.get("name", "unknown"))

    if store is None:
        logger.info(
            f"[save_memory] MemoryStore 없음, 저장 건너뜀 "
            f"(event_id={event_id}, risk_level={diagnosis.get('risk_level')})"
        )
        return {}

    try:
        memory_id = store.save(
            equipment_id=equipment_id,
            bearing_id=bearing_id,
            event_id=event_id,
            event_timestamp=event_timestamp,
            diagnosis_result=diagnosis,
            tools_used=tools_used if tools_used else None,
            deep_research=state.get("deep_research_activated", False),
        )
        logger.info(f"[save_memory] 저장 완료: memory_id={memory_id}")
    except Exception as e:
        logger.error(f"[save_memory] 저장 실패: {e}")

    return {}
