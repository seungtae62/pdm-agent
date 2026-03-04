"""reasoning 노드 — GPT-4o ReAct 추론.

시스템 프롬프트 + event_payload + memory_context로 LLM 추론을 수행하고,
next_action을 결정한다. Tool 호출이 필요하면 tool_calls를 메시지에 포함한다.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent.prompts.system_prompt import load_system_prompt
from agent.state import PdMAgentState

logger = logging.getLogger(__name__)


def _build_initial_message(state: PdMAgentState) -> str:
    """첫 추론을 위한 사용자 메시지 생성."""
    payload = state["event_payload"]
    memory = state.get("memory_context", {})
    history_summary = memory.get("history_summary", "")

    parts = ["## 이벤트 페이로드\n```json"]
    parts.append(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    parts.append("```")

    if history_summary:
        parts.append(f"\n## 이전 분석 이력\n{history_summary}")
    else:
        parts.append("\n## 이전 분석 이력\n이전 분석 이력 없음.")

    parts.append(
        "\n위 이벤트 페이로드를 분석하세요. "
        "추론 절차(Thought 1~5)에 따라 단계적으로 분석하고, "
        "최종 판단 결과를 아래 JSON 형식으로 제시하세요:\n"
        "```json\n"
        '{\n  "fault_type": "...",\n  "fault_stage": 0,\n'
        '  "degradation_speed": "...",\n'
        '  "rul_assessment": {"ml_rul_hours": null, "agent_assessment": "...", "confidence_level": "..."},\n'
        '  "risk_level": "...",\n  "recommendation": "...",\n'
        '  "uncertainty_notes": "...",\n  "reasoning_summary": "..."\n}\n```'
    )

    return "\n".join(parts)


def reasoning(state: PdMAgentState, *, llm: BaseChatModel, tools: list) -> dict:
    """추론 노드.

    Args:
        state: 현재 State.
        llm: LangChain ChatModel.
        tools: MCP에서 검색된 Tool 리스트 (bind_tools용).

    Returns:
        State 업데이트 dict.
    """
    messages = list(state.get("messages", []))

    # 첫 호출: 시스템 프롬프트 + 이벤트 페이로드
    if not messages:
        system_prompt = load_system_prompt()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=_build_initial_message(state)),
        ]

    # LLM 호출 (tools bound)
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)

    logger.info(
        f"[reasoning] LLM 응답: tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0}"
    )

    # next_action 결정
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        next_action = "call_tool"
    else:
        next_action = "generate_report"

    return {
        "messages": [response],
        "next_action": next_action,
    }
