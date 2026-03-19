"""deep_group_search 노드 — Deep Group Search Sub-Graph 실행.

Main Agent의 대화형 상호작용에서 deep_research_activated=True일 때
Deep Group Search Sub-Graph를 호출하고, 결과를 Main State에 반영한다.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from agent.config import AgentConfig
from agent.deep_search.graph import run_deep_search
from agent.state import PdMAgentState

logger = logging.getLogger(__name__)


async def deep_group_search(
    state: PdMAgentState,
    *,
    config: AgentConfig | None = None,
    tools: list | None = None,
) -> dict:
    """Deep Group Search Sub-Graph를 실행하고 결과를 Main State에 반영.

    마지막 사용자 메시지에서 질문을 추출하고,
    memory_context에서 추론 맥락을 전달한다.

    Args:
        state: Main Agent State.
        config: 에이전트 설정.
        tools: 기존 Action Skills Tool 리스트.

    Returns:
        Main State 업데이트 dict.
    """
    messages = state.get("messages", [])

    # 마지막 사용자 메시지에서 질문 추출
    original_query = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            original_query = msg.content
            break

    if not original_query:
        logger.warning("[deep_group_search] 사용자 메시지를 찾을 수 없음")
        return {
            "messages": [
                AIMessage(
                    content="심층 분석을 위한 질문을 찾을 수 없습니다. "
                    "질문을 다시 입력해 주세요."
                )
            ],
            "deep_research_activated": False,
        }

    # Memory context에서 추론 맥락 추출
    memory_context = state.get("memory_context", {})
    reasoning_context = memory_context.get("history_summary", "이전 이력 없음")

    logger.info(
        "[deep_group_search] Deep Group Search 시작: %s",
        original_query[:80],
    )

    try:
        result = await run_deep_search(
            original_query=original_query,
            reasoning_context=reasoning_context,
            config=config,
            tools=tools,
        )

        synthesis = result.get("synthesis", "")
        citations = result.get("citations", [])

        # 출처 정보를 답변에 포함
        if citations:
            citation_text = "\n\n---\n### 참조 출처\n"
            for i, c in enumerate(citations, 1):
                source_type = c.get("source_type", "unknown")
                title = c.get("title", "")
                doc_id = c.get("doc_id", "")
                url = c.get("url", "")

                if source_type == "external_web":
                    label = f"(외부 참고, 검증 필요)"
                else:
                    label = f"(내부 문서)"

                ref_parts = [f"[{i}]"]
                if title:
                    ref_parts.append(title)
                if doc_id:
                    ref_parts.append(f"ID: {doc_id}")
                if url:
                    ref_parts.append(url)
                ref_parts.append(label)
                citation_text += " ".join(ref_parts) + "\n"

            full_response = synthesis + citation_text
        else:
            full_response = synthesis

        logger.info(
            "[deep_group_search] Deep Group Search 완료, "
            "출처 %d건",
            len(citations),
        )

        return {
            "messages": [AIMessage(content=full_response)],
            "deep_research_activated": False,
        }

    except Exception as e:
        logger.error("[deep_group_search] Sub-Graph 실행 실패: %s", e)
        return {
            "messages": [
                AIMessage(
                    content=f"Deep Group Search 실행 중 오류가 발생했습니다: {e}\n"
                    "일반 검색으로 대체합니다."
                )
            ],
            "deep_research_activated": False,
        }
