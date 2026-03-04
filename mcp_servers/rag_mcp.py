"""RAG 검색 FastMCP Server.

RAGServer의 3개 검색 메서드를 MCP Tool로 래핑한다.
stdio transport로 서브프로세스에서 실행된다.
"""

from __future__ import annotations

import json
import os
import sys

# 서브프로세스에서 프로젝트 루트 import 보장
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("rag-server")

_server_instance: "RAGServer | None" = None


def _get_server():
    """지연 초기화로 RAGServer 인스턴스 반환."""
    global _server_instance
    if _server_instance is None:
        from mcp_servers.rag_server import RAGServer, RAGServerConfig

        _server_instance = RAGServer(RAGServerConfig.from_env())
    return _server_instance


@mcp.tool()
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
    server = _get_server()
    results = server.search_maintenance_history(
        query=query, equipment_id=equipment_id, bearing_id=bearing_id
    )
    return json.dumps(results, ensure_ascii=False, default=str)


@mcp.tool()
def search_equipment_manual(
    query: str,
    doc_type: str | None = None,
) -> str:
    """설비 매뉴얼, FMEA 문서, 정비 절차서를 검색합니다. 설비 사양, 결함 메커니즘, 급속 열화 조건, 교체 절차 등을 참조합니다.

    Args:
        query: 검색 쿼리 (예: '외륜 결함 급속 열화 조건').
        doc_type: 문서 유형 필터 (선택). 예: spec, fault_guide, procedure, fmea.
    """
    server = _get_server()
    results = server.search_equipment_manual(query=query, doc_type=doc_type)
    return json.dumps(results, ensure_ascii=False, default=str)


@mcp.tool()
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
    server = _get_server()
    results = server.search_analysis_history(
        query=query, equipment_id=equipment_id, bearing_id=bearing_id
    )
    return json.dumps(results, ensure_ascii=False, default=str)


if __name__ == "__main__":
    mcp.run()
