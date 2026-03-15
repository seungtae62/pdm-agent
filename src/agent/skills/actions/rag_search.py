"""RAG 검색 Action Skills.

Qdrant 기반 RAG 검색을 LangGraph Tool로 등록한다.
MCP 프로토콜을 거치지 않고 RAGServer를 직접 호출하여
서브프로세스 오버헤드를 제거한다.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# RAGServer 지연 초기화 (싱글턴)
_rag_server = None


def _get_rag_server():
    """RAGServer 인스턴스를 지연 초기화로 반환."""
    global _rag_server
    if _rag_server is None:
        from mcp_servers.rag_server import RAGServer, RAGServerConfig

        _rag_server = RAGServer(RAGServerConfig.from_env())
        logger.info("[action-skill] RAGServer 초기화 완료")
    return _rag_server


@tool
def search_maintenance_history(
    query: str,
    equipment_id: str | None = None,
    bearing_id: str | None = None,
) -> str:
    """과거 고장/정비 이력을 의미적으로 검색합니다. 유사 결함 사례의 진행 경과, 근본 원인, 고장까지 소요 시간 등을 참조합니다.

    Args:
        query: 검색 쿼리 (예: '내륜 결함 급속 열화 사례').
        equipment_id: 설비 ID 필터 (선택).
        bearing_id: 베어링 ID 필터 (선택).
    """
    server = _get_rag_server()
    results = server.search_maintenance_history(
        query=query, equipment_id=equipment_id, bearing_id=bearing_id
    )
    logger.info(
        "[action-skill] search_maintenance_history: query='%s', results=%d",
        query[:50],
        len(results),
    )
    return json.dumps(results, ensure_ascii=False, default=str)


@tool
def search_equipment_manual(
    query: str,
    doc_type: str | None = None,
) -> str:
    """설비 매뉴얼, FMEA 문서, 정비 절차서를 검색합니다. 설비 사양, 결함 메커니즘, 급속 열화 조건, 교체 절차 등을 참조합니다.

    Args:
        query: 검색 쿼리 (예: '외륜 결함 급속 열화 조건').
        doc_type: 문서 유형 필터 (선택). 예: spec, fault_guide, procedure, fmea.
    """
    server = _get_rag_server()
    results = server.search_equipment_manual(query=query, doc_type=doc_type)
    logger.info(
        "[action-skill] search_equipment_manual: query='%s', results=%d",
        query[:50],
        len(results),
    )
    return json.dumps(results, ensure_ascii=False, default=str)


@tool
def search_analysis_history(
    query: str,
    equipment_id: str | None = None,
    bearing_id: str | None = None,
) -> str:
    """에이전트의 과거 분석 판단 이력을 의미적으로 검색합니다. 유사한 패턴의 과거 판단과 결과를 참조합니다.

    Args:
        query: 검색 쿼리 (예: 'BPFI 상승 내륜 결함 2단계').
        equipment_id: 설비 ID 필터 (선택).
        bearing_id: 베어링 ID 필터 (선택).
    """
    server = _get_rag_server()
    results = server.search_analysis_history(
        query=query, equipment_id=equipment_id, bearing_id=bearing_id
    )
    logger.info(
        "[action-skill] search_analysis_history: query='%s', results=%d",
        query[:50],
        len(results),
    )
    return json.dumps(results, ensure_ascii=False, default=str)


def get_action_tools() -> list:
    """RAG Action Skill Tool 리스트를 반환."""
    return [
        search_maintenance_history,
        search_equipment_manual,
        search_analysis_history,
    ]
