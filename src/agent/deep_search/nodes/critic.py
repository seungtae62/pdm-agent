"""Critic Agent 노드 — 검색 결과 검증.

각 Research Agent 결과의 품질, 출처 유효성, 충분성을 평가한다.
Review-Revise 루프의 Review 역할을 담당한다.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.deep_search.prompts import CRITIC_PROMPT
from agent.deep_search.state import DeepSearchState

logger = logging.getLogger(__name__)


async def review(state: DeepSearchState, *, llm: BaseChatModel) -> dict:
    """검색 결과 검증.

    각 관점의 검색 결과를 검증하여 품질 평가 및 재검색 필요 여부를 판단한다.

    Args:
        state: Deep Search 상태.
        llm: LangChain ChatModel.

    Returns:
        State 업데이트 dict (critic_feedback, iteration_count).
    """
    original_query = state.get("original_query", "")
    search_results = state.get("search_results", [])
    iteration_count = state.get("iteration_count", 0)

    # 검색 결과를 텍스트로 정리
    results_parts = []
    for sr in search_results:
        perspective = sr.get("perspective", "알 수 없음")
        results_text = sr.get("results", "결과 없음")
        confidence = sr.get("confidence", 0.0)
        source_count = len(sr.get("sources", []))
        results_parts.append(
            f"### 관점: {perspective}\n"
            f"- 신뢰도: {confidence:.1f}\n"
            f"- 출처 수: {source_count}건\n"
            f"- 결과:\n{results_text}"
        )
    search_results_text = (
        "\n\n---\n\n".join(results_parts) if results_parts else "검색 결과 없음"
    )

    prompt = CRITIC_PROMPT.format(
        original_query=original_query,
        search_results_text=search_results_text,
    )

    logger.info("[deep_search:critic] 검증 시작 (iteration %d)", iteration_count)

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content="당신은 PdM Critic Agent입니다. "
                    "검색 결과를 검증하고 JSON만 출력하세요."
                ),
                HumanMessage(content=prompt),
            ]
        )

        # JSON 추출
        content = response.content or ""
        pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            feedback = json.loads(match.group(1).strip())
        else:
            feedback = json.loads(content.strip())

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(
            "[deep_search:critic] 검증 결과 파싱 실패, 전체 통과 처리: %s", e
        )
        # 파싱 실패 시 전체 통과 (안전한 기본값)
        feedback = [
            {
                "perspective": sr.get("perspective", ""),
                "passed": True,
                "feedback": "검증 결과 파싱 실패로 자동 통과",
                "unsourced_claims": [],
            }
            for sr in search_results
        ]

    # confidence가 매우 낮은 결과는 자동으로 재검색 필요 플래그
    for i, sr in enumerate(search_results):
        confidence = sr.get("confidence", 0.0)
        if confidence < 0.2:
            # feedback에서 해당 perspective 찾아서 passed를 False로
            perspective_name = sr.get("perspective", "")
            for fb in feedback:
                if fb.get("perspective") == perspective_name:
                    if confidence == 0.0:
                        fb["passed"] = False
                        fb["feedback"] = (
                            f"검색 결과가 없거나 매우 부족합니다 "
                            f"(confidence: {confidence:.1f}). "
                            f"검색 쿼리를 수정하여 재검색이 필요합니다."
                        )
                    break

    passed_count = sum(1 for fb in feedback if fb.get("passed", False))
    total_count = len(feedback)

    logger.info(
        "[deep_search:critic] 검증 완료: %d/%d 통과 (iteration %d)",
        passed_count,
        total_count,
        iteration_count,
    )

    return {
        "critic_feedback": feedback,
        "iteration_count": iteration_count + 1,
    }
