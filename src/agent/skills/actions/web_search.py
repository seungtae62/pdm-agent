"""웹 검색 Action Skill.

Tavily API를 사용하여 외부 기술 문헌을 검색한다.
내부 RAG로 충분한 정보를 확보하지 못했을 때만 사용하며,
검색 결과는 외부 참고 자료로서 검증이 필요하다.
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_tavily_client = None


def _get_client():
    """TavilyClient 인스턴스를 지연 초기화로 반환.

    Returns:
        TavilyClient 인스턴스. API key 미설정 시 None.
    """
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.warning("[action-skill] TAVILY_API_KEY 미설정, 웹 검색 비활성화")
            return None
        from tavily import TavilyClient

        _tavily_client = TavilyClient(api_key=api_key)
        logger.info("[action-skill] TavilyClient 초기화 완료")
    return _tavily_client


@tool
def search_web(query: str) -> str:
    """외부 웹에서 베어링 결함, 진동 분석, 정비 기술 문헌 등을 검색합니다.

    내부 RAG 검색으로 충분한 정보를 확보하지 못했을 때만 호출합니다.
    검색 결과는 외부 참고 자료이므로 교차 검증이 필요합니다.

    Args:
        query: 검색 쿼리 (영문 권장).
    """
    client = _get_client()
    if client is None:
        return (
            "웹 검색을 사용할 수 없습니다 (TAVILY_API_KEY 미설정). "
            "내부 RAG 검색 결과와 도메인 지식을 기반으로 분석을 계속하세요."
        )

    try:
        response = client.search(query, max_results=5)
    except Exception as e:
        logger.error("[action-skill] search_web 실패: %s", e)
        return f"웹 검색 중 오류 발생: {e}. 내부 RAG 검색 결과를 기반으로 분석을 계속하세요."

    logger.info("[action-skill] search_web: query=%s", query)

    results = response.get("results", [])
    if not results:
        return "검색 결과가 없습니다."

    formatted = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        formatted.append(f"[{i}] {title}\n    URL: {url}\n    {content}")

    return "\n\n".join(formatted)


def get_web_search_tools() -> list:
    """웹 검색 Action Skill Tool 리스트를 반환."""
    return [search_web]
