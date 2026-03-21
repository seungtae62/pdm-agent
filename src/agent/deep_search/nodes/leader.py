"""Leader Agent 노드 — 질문 분해 및 교차 참조 합성.

STORM 스타일 다관점 분해(decompose)와
교차 참조 합성(synthesize)을 수행한다.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.deep_search.prompts import DECOMPOSITION_PROMPT, SYNTHESIS_PROMPT
from agent.deep_search.state import DeepSearchState

logger = logging.getLogger(__name__)


def _fallback_perspectives(original_query: str) -> list[dict]:
    """고정 3관점 폴백을 반환.

    Args:
        original_query: 사용자 원본 질문.

    Returns:
        3개 관점 dict 리스트.
    """
    return [
        {
            "perspective": "정비 엔지니어",
            "sub_query": original_query,
            "agent_role": "maintenance_history",
            "search_tools": ["search_maintenance_history"],
        },
        {
            "perspective": "신뢰성 엔지니어",
            "sub_query": original_query,
            "agent_role": "analysis_history",
            "search_tools": ["search_analysis_history"],
        },
        {
            "perspective": "설비 전문가",
            "sub_query": original_query,
            "agent_role": "equipment_manual",
            "search_tools": ["search_equipment_manual", "search_web"],
        },
    ]


def _extract_json(text: str) -> list | dict | None:
    """LLM 응답에서 JSON을 추출.

    코드블록 내부 또는 raw JSON을 파싱한다.

    Args:
        text: LLM 응답 텍스트.

    Returns:
        파싱된 JSON 객체. 실패 시 None.
    """
    # 코드블록 내부 추출
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # raw JSON 시도
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    return None


async def decompose(state: DeepSearchState, *, llm: BaseChatModel) -> dict:
    """STORM 스타일 다관점 질문 분해.

    사용자 질문을 여러 전문가 관점의 하위 질문으로 분해한다.

    Args:
        state: Deep Search 상태.
        llm: LangChain ChatModel.

    Returns:
        State 업데이트 dict (perspectives).
    """
    original_query = state.get("original_query", "")
    reasoning_context = state.get("reasoning_context", "정보 없음")

    prompt = DECOMPOSITION_PROMPT.format(
        original_query=original_query,
        reasoning_context=reasoning_context,
    )

    logger.info("[deep_search:leader] 질문 분해 시작: %s", original_query[:80])

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content="당신은 베어링 예지보전 심층 분석의 Leader Agent입니다. "
                    "JSON만 출력하세요."
                ),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as e:
        logger.error("[deep_search:leader] 질문 분해 LLM 호출 실패: %s", e)
        # 폴백: 고정 3관점 분해
        return {
            "perspectives": _fallback_perspectives(original_query),
        }

    parsed = _extract_json(response.content or "")
    if not parsed or not isinstance(parsed, list) or len(parsed) != 3:
        logger.warning(
            "[deep_search:leader] JSON 파싱 실패 또는 관점 수 불일치 "
            "(expected 3, got %s), 폴백 분해 사용. 응답: %s",
            len(parsed) if isinstance(parsed, list) else "N/A",
            (response.content or "")[:200],
        )
        return {
            "perspectives": _fallback_perspectives(original_query),
        }

    logger.info(
        "[deep_search:leader] %d개 관점 분해 완료: %s",
        len(parsed),
        [p.get("perspective", "?") for p in parsed],
    )
    return {"perspectives": parsed}


async def synthesize(state: DeepSearchState, *, llm: BaseChatModel) -> dict:
    """교차 참조 합성.

    모든 관점의 검색 결과와 Critic 피드백을 종합하여
    최종 답변을 생성한다.

    Args:
        state: Deep Search 상태.
        llm: LangChain ChatModel.

    Returns:
        State 업데이트 dict (synthesis, citations).
    """
    original_query = state.get("original_query", "")
    search_results = state.get("search_results", [])
    critic_feedback = state.get("critic_feedback", [])

    # 검색 결과를 텍스트로 정리
    results_parts = []
    for sr in search_results:
        perspective = sr.get("perspective", "알 수 없음")
        results_text = sr.get("results", "결과 없음")
        confidence = sr.get("confidence", 0.0)
        results_parts.append(
            f"### 관점: {perspective} (신뢰도: {confidence:.1f})\n{results_text}"
        )
    all_results_text = (
        "\n\n---\n\n".join(results_parts) if results_parts else "검색 결과 없음"
    )

    # Critic 피드백을 텍스트로 정리
    feedback_parts = []
    for cf in critic_feedback:
        perspective = cf.get("perspective", "알 수 없음")
        passed = "통과" if cf.get("passed", False) else "미통과"
        feedback = cf.get("feedback", "")
        feedback_parts.append(f"- {perspective}: {passed} — {feedback}")
    critic_feedback_text = (
        "\n".join(feedback_parts) if feedback_parts else "검증 미수행"
    )

    prompt = SYNTHESIS_PROMPT.format(
        original_query=original_query,
        all_results_text=all_results_text,
        critic_feedback_text=critic_feedback_text,
    )

    logger.info("[deep_search:leader] 교차 참조 합성 시작")

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content="당신은 베어링 예지보전 심층 분석의 Leader Agent입니다. "
                    "여러 관점의 결과를 교차 참조하여 종합 분석하세요."
                ),
                HumanMessage(content=prompt),
            ]
        )
        synthesis_text = response.content or ""
    except Exception as e:
        logger.error("[deep_search:leader] 합성 LLM 호출 실패: %s", e)
        synthesis_text = (
            f"Deep Group Search 합성 실패: {e}\n\n"
            "개별 검색 결과:\n" + all_results_text
        )

    # 모든 검색 결과에서 출처 수집
    all_citations = []
    for sr in search_results:
        for source in sr.get("sources", []):
            if source not in all_citations:
                all_citations.append(source)

    logger.info("[deep_search:leader] 합성 완료, 출처 %d건", len(all_citations))

    return {
        "synthesis": synthesis_text,
        "citations": all_citations,
    }
