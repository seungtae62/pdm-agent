"""load_memory 노드 — Memory 로드.

event_payload에서 설비/베어링 ID를 추출하여
PostgreSQL에서 이전 판단 이력을 조회하고 memory_context에 저장한다.
"""

from __future__ import annotations

import logging

from agent.memory.store import MemoryStore
from agent.state import PdMAgentState

logger = logging.getLogger(__name__)


def load_memory(state: PdMAgentState, *, store: MemoryStore | None = None) -> dict:
    """Memory 로드 노드.

    Args:
        state: 현재 State.
        store: MemoryStore 인스턴스. None이면 memory_context를 빈 dict로 설정.

    Returns:
        State 업데이트 dict.
    """
    payload = state["event_payload"]
    equipment_id = payload.get("equipment_id", "")
    bearing_id = payload.get("bearing_id", "")

    if store is None:
        logger.info("[load_memory] MemoryStore 없음, 빈 context 반환")
        return {
            "memory_context": {
                "equipment_id": equipment_id,
                "bearing_id": bearing_id,
                "history_summary": "",
                "history_records": [],
            }
        }

    records = store.load_recent(equipment_id, bearing_id)
    summary = MemoryStore.summarize_history(records)

    logger.info(
        f"[load_memory] {equipment_id}/{bearing_id}: {len(records)}건 이력 로드"
    )

    return {
        "memory_context": {
            "equipment_id": equipment_id,
            "bearing_id": bearing_id,
            "history_summary": summary,
            "history_records": records,
        }
    }
