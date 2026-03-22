"""웹 검색 Action Skill.

SerpAPI를 사용하여 외부 기술 문헌을 검색한다.
내부 RAG로 충분한 정보를 확보하지 못했을 때만 사용하며,
검색 결과는 외부 참고 자료로서 검증이 필요하다.
"""

from __future__ import annotations

import logging
import os

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_SERPAPI_BASE_URL = "https://serpapi.com/search.json"


@tool
def search_web(query: str) -> str:
    """외부 웹에서 베어링 결함, 진동 분석, 정비 기술 문헌 등을 검색합니다.

    내부 RAG 검색으로 충분한 정보를 확보하지 못했을 때만 호출합니다.
    검색 결과는 외부 참고 자료이므로 교차 검증이 필요합니다.

    Args:
        query: 검색 쿼리 (영문 권장).
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return (
            "웹 검색을 사용할 수 없습니다 (SERPAPI_API_KEY 미설정). "
            "내부 RAG 검색 결과와 도메인 지식을 기반으로 분석을 계속하세요."
        )

    try:
        resp = requests.get(
            _SERPAPI_BASE_URL,
            params={
                "q": query,
                "api_key": api_key,
                "engine": "google",
                "num": 5,
                "hl": "en",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("[action-skill] search_web 실패: %s", e)
        return f"웹 검색 중 오류 발생: {e}. 내부 RAG 검색 결과를 기반으로 분석을 계속하세요."

    logger.info("[action-skill] search_web: query=%s", query)

    results = data.get("organic_results", [])
    if not results:
        return "검색 결과가 없습니다."

    formatted = []
    for i, r in enumerate(results[:5], 1):
        title = r.get("title", "")
        url = r.get("link", "")
        snippet = r.get("snippet", "")
        formatted.append(f"[{i}] {title}\n    URL: {url}\n    {snippet}")

    return "\n\n".join(formatted)


def get_web_search_tools() -> list:
    """웹 검색 Action Skill Tool 리스트를 반환."""
    return [search_web]
