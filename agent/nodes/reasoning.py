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

# Tool 정의 (LangChain bind_tools용)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_maintenance_history",
            "description": "과거 고장/정비 이력을 의미적으로 검색합니다. 유사 결함 사례의 진행 경과, 근본 원인, 고장까지 소요 시간 등을 참조합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리 (예: '내륜 결함 급속 열화 사례')",
                    },
                    "equipment_id": {
                        "type": "string",
                        "description": "설비 ID 필터 (선택)",
                    },
                    "bearing_id": {
                        "type": "string",
                        "description": "베어링 ID 필터 (선택)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_equipment_manual",
            "description": "설비 매뉴얼, FMEA 문서, 정비 절차서를 검색합니다. 설비 사양, 결함 메커니즘, 급속 열화 조건, 교체 절차 등을 참조합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리 (예: '외륜 결함 급속 열화 조건')",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "문서 유형 필터 (선택). 예: spec, fault_guide, procedure, fmea",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_analysis_history",
            "description": "에이전트의 과거 분석 판단 이력을 의미적으로 검색합니다. 유사한 패턴의 과거 판단과 결과를 참조합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리 (예: 'BPFI 상승 내륜 결함 2단계')",
                    },
                    "equipment_id": {
                        "type": "string",
                        "description": "설비 ID 필터 (선택)",
                    },
                    "bearing_id": {
                        "type": "string",
                        "description": "베어링 ID 필터 (선택)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify_maintenance_staff",
            "description": "정비 담당자에게 분석 결과 및 정비 권고 알림을 전송합니다. 위험도 Watch 이상이거나 인간의 확인이 필요할 때 호출합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "알림 메시지 (분석 결과 요약 + 정비 권고)",
                    },
                    "risk_level": {
                        "type": "string",
                        "description": "위험도 (watch / warning / critical)",
                    },
                    "equipment_id": {
                        "type": "string",
                        "description": "설비 ID",
                    },
                },
                "required": ["message", "risk_level", "equipment_id"],
            },
        },
    },
]


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


def reasoning(state: PdMAgentState, *, llm: BaseChatModel) -> dict:
    """추론 노드.

    Args:
        state: 현재 State.
        llm: LangChain ChatModel (bind_tools 적용됨).

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
    llm_with_tools = llm.bind_tools(TOOL_DEFINITIONS)
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
