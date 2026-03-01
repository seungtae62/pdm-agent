"""PostgreSQL 기반 Long-term Memory CRUD.

pdm_agent_memory 테이블에 에이전트 판단 결과를 저장하고 조회한다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class MemoryStore:
    """PostgreSQL Memory Store.

    Args:
        dsn: PostgreSQL 연결 문자열.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def initialize(self) -> None:
        """테이블 생성 (존재하지 않으면)."""
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        with psycopg.connect(self.dsn) as conn:
            conn.execute(schema_sql)
            conn.commit()
        logger.info("[Memory] 테이블 초기화 완료")

    def load_recent(
        self,
        equipment_id: str,
        bearing_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """최근 판단 이력 조회.

        Args:
            equipment_id: 설비 ID.
            bearing_id: 베어링 ID.
            limit: 최대 조회 건수.

        Returns:
            최근 판단 이력 리스트 (시간 역순).
        """
        query = """
            SELECT memory_id, event_id, event_timestamp,
                   fault_type, fault_stage, degradation_speed,
                   risk_level, ml_rul_hours, agent_rul_assessment,
                   recommendation, uncertainty_notes, reasoning_summary,
                   tools_used, deep_research
            FROM pdm_agent_memory
            WHERE equipment_id = %s AND bearing_id = %s
            ORDER BY event_timestamp DESC
            LIMIT %s
        """
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn:
            rows = conn.execute(query, (equipment_id, bearing_id, limit)).fetchall()

        logger.info(
            f"[Memory] 이력 조회: {equipment_id}/{bearing_id} → {len(rows)}건"
        )
        return rows

    def save(
        self,
        equipment_id: str,
        bearing_id: str,
        event_id: str,
        event_timestamp: datetime,
        diagnosis_result: dict,
        tools_used: list[str] | None = None,
        deep_research: bool = False,
    ) -> str:
        """판단 결과 저장.

        Args:
            equipment_id: 설비 ID.
            bearing_id: 베어링 ID.
            event_id: 이벤트 ID.
            event_timestamp: 이벤트 타임스탬프.
            diagnosis_result: 에이전트 판단 결과 dict.
            tools_used: 사용된 Tool 목록.
            deep_research: Deep Research 발동 여부.

        Returns:
            저장된 memory_id (UUID 문자열).
        """
        rul = diagnosis_result.get("rul_assessment", {})

        query = """
            INSERT INTO pdm_agent_memory (
                equipment_id, bearing_id, event_id, event_timestamp,
                fault_type, fault_stage, degradation_speed, risk_level,
                ml_rul_hours, agent_rul_assessment,
                recommendation, uncertainty_notes, reasoning_summary,
                tools_used, deep_research
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s
            )
            RETURNING memory_id
        """
        params = (
            equipment_id,
            bearing_id,
            event_id,
            event_timestamp,
            diagnosis_result.get("fault_type"),
            diagnosis_result.get("fault_stage"),
            diagnosis_result.get("degradation_speed"),
            diagnosis_result.get("risk_level"),
            rul.get("ml_rul_hours"),
            rul.get("agent_assessment"),
            diagnosis_result.get("recommendation"),
            diagnosis_result.get("uncertainty_notes"),
            diagnosis_result.get("reasoning_summary"),
            json.dumps(tools_used) if tools_used else None,
            deep_research,
        )

        with psycopg.connect(self.dsn) as conn:
            result = conn.execute(query, params).fetchone()
            conn.commit()

        memory_id = str(result[0])
        logger.info(
            f"[Memory] 저장 완료: {equipment_id}/{bearing_id}/{event_id} "
            f"→ memory_id={memory_id}"
        )
        return memory_id

    @staticmethod
    def summarize_history(records: list[dict]) -> str:
        """이력을 자연어 요약으로 변환.

        Args:
            records: load_recent()의 반환값.

        Returns:
            자연어 요약 문자열. 이력이 없으면 빈 문자열.
        """
        if not records:
            return ""

        lines = [f"이전 분석 이력 ({len(records)}건, 최신순):"]
        for r in records:
            ts = r["event_timestamp"]
            if isinstance(ts, datetime):
                ts = ts.strftime("%Y-%m-%d %H:%M")
            fault = r.get("fault_type") or "none"
            stage = r.get("fault_stage", 0)
            risk = r.get("risk_level") or "unknown"
            summary = r.get("reasoning_summary") or ""
            if len(summary) > 100:
                summary = summary[:100] + "..."
            lines.append(
                f"  - [{ts}] 결함: {fault}, 단계: {stage}, "
                f"위험도: {risk}. {summary}"
            )

        return "\n".join(lines)
