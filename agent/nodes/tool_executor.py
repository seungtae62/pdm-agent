"""tool_executor 노드 — MCP Tool 실행.

reasoning 노드에서 결정된 Tool 호출을 실행하고,
결과를 Observation 메시지로 messages에 추가한다.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import ToolMessage

from agent.state import PdMAgentState
from mcp_servers.rag_server import RAGServer
from mcp_servers.notification_server import NotificationServer

logger = logging.getLogger(__name__)


def tool_executor(
    state: PdMAgentState,
    *,
    rag_server: RAGServer | None = None,
    notification_server: NotificationServer | None = None,
) -> dict:
    """Tool 실행 노드.

    Args:
        state: 현재 State.
        rag_server: RAG 검색 서버. None이면 더미 응답.
        notification_server: 알림 서버. None이면 더미 응답.

    Returns:
        State 업데이트 dict.
    """
    messages = state.get("messages", [])
    tool_calls_count = state.get("tool_calls_count", 0)

    # 마지막 메시지에서 tool_calls 추출
    last_message = messages[-1] if messages else None
    tool_calls = getattr(last_message, "tool_calls", None) or []

    if not tool_calls:
        logger.warning("[tool_executor] tool_calls 없음, 건너뜀")
        return {"next_action": "continue_reasoning"}

    tool_messages = []
    tools_used = []

    for call in tool_calls:
        name = call["name"]
        args = call.get("args", {})
        call_id = call.get("id", name)
        tools_used.append(name)

        logger.info(f"[tool_executor] 실행: {name}({json.dumps(args, ensure_ascii=False)[:100]})")

        try:
            result = _execute_tool(
                name, args,
                rag_server=rag_server,
                notification_server=notification_server,
            )
            result_text = json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"[tool_executor] {name} 실행 실패: {e}")
            result_text = json.dumps({"error": str(e)})

        tool_messages.append(
            ToolMessage(content=result_text, tool_call_id=call_id)
        )

    new_count = tool_calls_count + len(tool_calls)
    deep_research = new_count >= 3  # 3회 이상 Tool 호출 시 Deep Research로 간주

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


def _execute_tool(
    name: str,
    args: dict,
    *,
    rag_server: RAGServer | None,
    notification_server: NotificationServer | None,
) -> dict | list:
    """Tool 이름으로 분기하여 실행."""
    if name == "search_maintenance_history":
        if rag_server is None:
            return [{"score": 0.0, "text": "(RAG 서버 미연결)", "metadata": {}}]
        return rag_server.search_maintenance_history(**args)

    elif name == "search_equipment_manual":
        if rag_server is None:
            return [{"score": 0.0, "text": "(RAG 서버 미연결)", "metadata": {}}]
        return rag_server.search_equipment_manual(**args)

    elif name == "search_analysis_history":
        if rag_server is None:
            return [{"score": 0.0, "text": "(RAG 서버 미연결)", "metadata": {}}]
        return rag_server.search_analysis_history(**args)

    elif name == "notify_maintenance_staff":
        if notification_server is None:
            return {"success": True, "message": "(알림 서버 미연결, PoC 모드)"}
        result = notification_server.notify_maintenance_staff(**args)
        return {"success": result.success, "message": result.message}

    else:
        raise ValueError(f"알 수 없는 Tool: {name}")
