"""Research Agent 노드 — 관점별 집중 검색.

각 perspective에 대해 해당 도구를 사용하여 검색을 수행하고,
confidence score를 산출한다.
기존 Action Skills(rag_search, web_search)를 재사용한다.
asyncio.gather()로 3개 관점을 병렬 실행한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.deep_search.prompts import RESEARCH_PROMPT
from agent.deep_search.state import DeepSearchState

logger = logging.getLogger(__name__)

# agent_role → Tool 함수 이름 매핑
_ROLE_TO_TOOLS: dict[str, list[str]] = {
    "maintenance_history": ["search_maintenance_history"],
    "analysis_history": ["search_analysis_history"],
    "equipment_manual": ["search_equipment_manual"],
    "external_search": ["search_web"],
}


def _find_tool(tools: list, name: str):
    """Tool 리스트에서 이름으로 Tool을 검색."""
    for t in tools:
        if t.name == name:
            return t
    return None


def _calculate_confidence(raw_results: str) -> float:
    """검색 결과의 confidence score를 산출.

    검색 결과의 양과 품질에 기반한 간단한 휴리스틱.
    0.0 (결과 없음) ~ 1.0 (풍부한 결과).

    Args:
        raw_results: Tool 호출 결과 문자열.

    Returns:
        confidence score (0.0 ~ 1.0).
    """
    if not raw_results or raw_results == "검색 결과가 없습니다.":
        return 0.0

    # JSON 파싱 시도 (RAG 결과)
    try:
        parsed = json.loads(raw_results)
        if isinstance(parsed, list):
            count = len(parsed)
            if count == 0:
                return 0.0
            elif count == 1:
                return 0.4
            elif count <= 3:
                return 0.7
            else:
                return 0.9
    except (json.JSONDecodeError, TypeError):
        pass

    # 텍스트 길이 기반 (웹 검색 결과)
    length = len(raw_results)
    if length < 50:
        return 0.1
    elif length < 200:
        return 0.4
    elif length < 500:
        return 0.6
    else:
        return 0.8


def _extract_sources(raw_results: str, source_type: str) -> list[dict]:
    """검색 결과에서 출처 정보를 추출.

    Args:
        raw_results: Tool 호출 결과 문자열.
        source_type: internal_rag / external_web.

    Returns:
        Citation dict 리스트.
    """
    sources = []

    # RAG 결과 (JSON 리스트)
    try:
        parsed = json.loads(raw_results)
        if isinstance(parsed, list):
            for item in parsed:
                source = {
                    "source_type": source_type,
                    "doc_id": item.get("id", item.get("doc_id", "")),
                    "title": item.get(
                        "title",
                        item.get("metadata", {}).get("title", ""),
                    ),
                    "url": "",
                    "relevance_score": item.get("score", 0.0),
                }
                sources.append(source)
            return sources
    except (json.JSONDecodeError, TypeError):
        pass

    # 웹 검색 결과 (텍스트 포맷)
    if source_type == "external_web":
        url_pattern = r"URL:\s*(https?://\S+)"
        title_pattern = r"\[\d+\]\s*(.+)"
        urls = re.findall(url_pattern, raw_results)
        titles = re.findall(title_pattern, raw_results)

        for i, url in enumerate(urls):
            title = titles[i] if i < len(titles) else ""
            sources.append(
                {
                    "source_type": "external_web",
                    "doc_id": "",
                    "title": title.strip(),
                    "url": url.strip(),
                    "relevance_score": 0.5,
                }
            )

    return sources


async def research(state: DeepSearchState, *, llm: BaseChatModel, tools: list) -> dict:
    """관점별 집중 검색 수행.

    각 perspective에 대해:
    1. 해당 agent_role에 맞는 Tool로 검색
    2. LLM으로 결과 정리
    3. confidence score 산출
    4. 출처 추출

    Args:
        state: Deep Search 상태.
        llm: LangChain ChatModel.
        tools: 기존 Action Skills Tool 리스트.

    Returns:
        State 업데이트 dict (search_results).
    """
    perspectives = state.get("perspectives", [])
    original_query = state.get("original_query", "")
    reasoning_context = state.get("reasoning_context", "정보 없음")
    critic_feedback = state.get("critic_feedback", [])

    if not perspectives:
        logger.warning("[deep_search:researcher] perspectives가 비어있음, 스킵")
        return {"search_results": []}

    if not tools:
        logger.warning("[deep_search:researcher] tools가 비어있음, 스킵")
        return {"search_results": []}

    # Critic 피드백이 있으면 revision이 필요한 관점만 재검색
    if critic_feedback:
        failed_perspectives = {
            cf["perspective"] for cf in critic_feedback if not cf.get("passed", True)
        }
        if failed_perspectives:
            perspectives = [
                p for p in perspectives if p.get("perspective") in failed_perspectives
            ]
            logger.info(
                "[deep_search:researcher] Critic 피드백 기반 재검색: %s",
                failed_perspectives,
            )

    search_results = list(state.get("search_results", []))
    # 기존 결과 중 재검색 대상이 아닌 것 유지
    if critic_feedback:
        failed_set = {
            cf["perspective"] for cf in critic_feedback if not cf.get("passed", True)
        }
        search_results = [
            sr for sr in search_results if sr.get("perspective") not in failed_set
        ]

    async def _research_one_perspective(p: dict) -> dict:
        """단일 perspective에 대한 검색을 수행한다."""
        perspective = p.get("perspective", "알 수 없음")
        sub_query = p.get("sub_query", original_query)
        agent_role = p.get("agent_role", "maintenance_history")
        search_tool_names = p.get("search_tools", _ROLE_TO_TOOLS.get(agent_role, []))

        logger.info(
            "[deep_search:researcher] 검색 시작 — 관점: %s, 역할: %s",
            perspective,
            agent_role,
        )

        # Tool 호출 (sync tool.invoke를 asyncio.to_thread로 감싸서 비동기 실행)
        raw_results_parts = []
        all_sources = []
        for tool_name in search_tool_names:
            t = _find_tool(tools, tool_name)
            if t is None:
                logger.warning("[deep_search:researcher] Tool 미발견: %s", tool_name)
                continue

            try:
                result = await asyncio.to_thread(t.invoke, {"query": sub_query})
                raw_results_parts.append(f"[{tool_name}]\n{result}")

                source_type = "external_web" if "web" in tool_name else "internal_rag"
                sources = _extract_sources(str(result), source_type)
                all_sources.extend(sources)
            except Exception as e:
                logger.error(
                    "[deep_search:researcher] Tool 호출 실패 %s: %s",
                    tool_name,
                    e,
                )
                raw_results_parts.append(f"[{tool_name}] 검색 실패: {e}")

        raw_results = "\n\n".join(raw_results_parts)

        # LLM으로 검색 결과 정리
        research_prompt = RESEARCH_PROMPT.format(
            perspective=perspective,
            sub_query=sub_query,
            agent_role=agent_role,
            original_query=original_query,
            reasoning_context=reasoning_context,
        )

        try:
            response = await llm.ainvoke(
                [
                    SystemMessage(
                        content="당신은 PdM Research Agent입니다. "
                        "검색 결과를 정리하고 출처를 명시하세요."
                    ),
                    HumanMessage(
                        content=f"{research_prompt}\n\n## 검색 결과 (raw)\n{raw_results}"
                    ),
                ]
            )
            organized_results = response.content or raw_results
        except Exception as e:
            logger.error("[deep_search:researcher] 결과 정리 LLM 실패: %s", e)
            organized_results = raw_results

        confidence = _calculate_confidence(raw_results)

        logger.info(
            "[deep_search:researcher] 검색 완료 — 관점: %s, "
            "confidence: %.1f, 출처: %d건",
            perspective,
            confidence,
            len(all_sources),
        )

        return {
            "perspective": perspective,
            "sub_query": sub_query,
            "agent_role": agent_role,
            "results": organized_results,
            "confidence": confidence,
            "sources": all_sources,
            "needs_revision": False,
        }

    # 3개 관점 병렬 실행
    parallel_results = await asyncio.gather(
        *[_research_one_perspective(p) for p in perspectives],
        return_exceptions=True,
    )

    for r in parallel_results:
        if isinstance(r, Exception):
            logger.error("[deep_search:researcher] 병렬 검색 예외: %s", r)
            continue
        search_results.append(r)

    return {"search_results": search_results}
