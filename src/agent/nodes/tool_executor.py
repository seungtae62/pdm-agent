"""tool_executor 노드 — MCP Tool 실행.

ToolNode를 감싸 MCP 도구를 실행하고,
tool_calls_count / deep_research_activated 부기 로직을 유지한다.
"""

from __future__ import annotations

import logging

from langgraph.prebuilt import ToolNode

from agent.state import PdMAgentState

logger = logging.getLogger(__name__)


def create_tool_executor(tools: list):
    """ToolNode를 감싸는 tool_executor 노드 팩토리.

    Args:
        tools: MCP에서 검색된 Tool 리스트.

    Returns:
        tool_executor 노드 함수.
    """
    tool_node = ToolNode(tools)

    async def tool_executor(state: PdMAgentState) -> dict:
        """Tool 실행 노드."""
        messages = state.get("messages", [])
        tool_calls_count = state.get("tool_calls_count", 0)

        # 마지막 메시지에 tool_calls가 없으면 조기 반환
        last_message = messages[-1] if messages else None
        tool_calls = getattr(last_message, "tool_calls", None) or []

        if not tool_calls:
            logger.warning("[tool_executor] tool_calls 없음, 건너뜀")
            return {"next_action": "continue_reasoning"}

        logger.info(
            f"[tool_executor] {len(tool_calls)}건 실행 시작: "
            f"{[c['name'] for c in tool_calls]}"
        )

        # ToolNode로 MCP 도구 실행 → ToolMessage 리스트 반환
        result = await tool_node.ainvoke(state)
        tool_messages = result["messages"]

        new_count = tool_calls_count + len(tool_calls)
        deep_research = new_count >= 3

        logger.info(
            f"[tool_executor] {len(tool_calls)}건 실행 완료, "
            f"누적 tool_calls_count={new_count}"
        )

        return {
            "messages": tool_messages,
            "tool_calls_count": new_count,
            "deep_research_activated": deep_research,
            "next_action": "continue_reasoning",
        }

    return tool_executor
