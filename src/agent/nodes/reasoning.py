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
from agent.skills.registry import load_matching_skills
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

    # Skills 주입
    skills_content = load_matching_skills(state)
    if skills_content:
        parts.append(f"\n## Active Domain Skills\n{skills_content}")

    parts.append(
        "\n위 이벤트 페이로드를 분석하세요. "
        "추론 절차(Thought 1~5)에 따라 단계적으로 분석하고, "
        "최종 판단 결과를 반드시 아래 JSON 형식으로 ```json 코드블록 안에 제시하세요.\n"
        "enum 필드(fault_type, degradation_speed, confidence_level, risk_level)는 "
        "반드시 지정된 영어 값만 사용하세요:\n"
        "```json\n"
        "{\n"
        '  "fault_type": "inner_race|outer_race|rolling_element|cage|none|unknown",\n'
        '  "fault_stage": 0,\n'
        '  "degradation_speed": "stable|normal|accelerating|abnormal",\n'
        '  "rul_assessment": {\n'
        '    "ml_rul_hours": null,\n'
        '    "agent_assessment": "한국어 서술",\n'
        '    "confidence_level": "high|medium|low"\n'
        "  },\n"
        '  "risk_level": "normal|watch|warning|critical",\n'
        '  "recommendation": "한국어 정비 권고",\n'
        '  "uncertainty_notes": "한국어 불확실성 고지",\n'
        '  "reasoning_summary": "한국어 추론 요약"\n'
        "}\n```"
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
    try:
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        logger.error(f"[reasoning] LLM 호출 실패: {e}")
        from langchain_core.messages import AIMessage

        error_msg = f"LLM 호출 실패로 자동 분석을 수행할 수 없습니다. 오류: {e}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "next_action": "generate_report",
        }

    logger.info(
        f"[reasoning] LLM 응답: tool_calls={len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0}"
    )

    # next_action 결정
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        next_action = "call_tool"
    else:
        # 이상 이벤트에서 첫 reasoning인데 Tool 호출 없이 끝난 경우
        # → search_maintenance_history 호출을 유도
        tool_count = state.get("tool_calls_count", 0)
        is_anomaly = (
            state.get("event_payload", {})
            .get("anomaly_detection_result", {})
            .get("anomaly_detected", False)
        )
        if is_anomaly and tool_count == 0:
            logger.info(
                "[reasoning] 이상 이벤트인데 Tool 미호출, "
                "search_maintenance_history 호출 유도"
            )
            eq_meta = state.get("event_payload", {}).get("equipment_meta", {})
            eq_id = eq_meta.get("equipment_id", "")
            brg_id = eq_meta.get("bearing", {}).get("bearing_id", "")
            nudge = HumanMessage(
                content=(
                    "분석이 진행 중입니다. Thought 3에 따라, "
                    "이 설비의 과거 정비 이력을 확인해야 합니다. "
                    f"search_maintenance_history를 호출하세요. "
                    f"(equipment_id: {eq_id}, bearing_id: {brg_id})"
                )
            )
            try:
                nudge_messages = messages + [response, nudge]
                llm_with_tools = llm.bind_tools(tools)
                nudge_response = llm_with_tools.invoke(nudge_messages)
                nudge_tool_calls = (
                    getattr(nudge_response, "tool_calls", None) or []
                )
                if nudge_tool_calls:
                    return {
                        "messages": [response, nudge, nudge_response],
                        "next_action": "call_tool",
                    }
                # 그래도 Tool 미호출이면 계속 진행
                return {
                    "messages": [response, nudge, nudge_response],
                    "next_action": "generate_report",
                }
            except Exception as e:
                logger.error(f"[reasoning] Tool 유도 실패: {e}")

        # Tool 호출 없이 끝남 → 진단 JSON이 포함되어 있는지 확인
        content = response.content or ""
        if '"fault_type"' in content and '"risk_level"' in content:
            next_action = "generate_report"
        else:
            # JSON 미포함 → 한 번 더 요청
            logger.info("[reasoning] 진단 JSON 미포함, JSON 출력 재요청")
            followup = HumanMessage(
                content=(
                    "분석이 완료되었습니다. "
                    "이제 위 분석 내용을 기반으로 최종 진단 결과를 "
                    "반드시 ```json 코드블록 안에 JSON만 출력하세요. "
                    "다른 텍스트 없이 JSON만 출력합니다.\n"
                    "fault_type, degradation_speed, confidence_level, "
                    "risk_level은 반드시 지정된 영어 값만 사용하세요:\n"
                    "```json\n"
                    "{\n"
                    '  "fault_type": "inner_race|outer_race|rolling_element'
                    '|cage|none|unknown",\n'
                    '  "fault_stage": 0,\n'
                    '  "degradation_speed": "stable|normal|accelerating'
                    '|abnormal",\n'
                    '  "rul_assessment": {\n'
                    '    "ml_rul_hours": null,\n'
                    '    "agent_assessment": "한국어 서술",\n'
                    '    "confidence_level": "high|medium|low"\n'
                    "  },\n"
                    '  "risk_level": "normal|watch|warning|critical",\n'
                    '  "recommendation": "한국어 정비 권고",\n'
                    '  "uncertainty_notes": "한국어 불확실성 고지",\n'
                    '  "reasoning_summary": "한국어 추론 요약"\n'
                    "}\n```"
                )
            )
            try:
                followup_messages = messages + [response, followup]
                json_response = llm.invoke(followup_messages)
                logger.info("[reasoning] JSON 재요청 응답 수신")
                return {
                    "messages": [response, followup, json_response],
                    "next_action": "generate_report",
                }
            except Exception as e:
                logger.error(f"[reasoning] JSON 재요청 실패: {e}")
                next_action = "generate_report"

    return {
        "messages": [response],
        "next_action": next_action,
    }
