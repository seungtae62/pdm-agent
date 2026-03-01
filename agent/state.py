"""PdM Agent State 스키마.

LangGraph StateGraph의 상태를 정의한다.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PdMAgentState(TypedDict):
    """PdM Agent의 StateGraph 상태.

    Attributes:
        event_payload: Edge에서 수신한 이벤트 페이로드 dict.
        memory_context: PostgreSQL에서 조회한 이전 판단 이력 요약.
        messages: ReAct 추론 과정의 메시지 누적 (LangGraph add_messages reducer).
        diagnosis_result: 에이전트 판단 결과 (구조화된 dict).
        tool_calls_count: Tool 호출 횟수 (안전장치용).
        deep_research_activated: Deep Research 발동 여부.
        report: 분석 리포트 텍스트.
        work_order: 작업지시서 텍스트.
        next_action: 워크플로우 분기 제어.
    """

    # 입력 데이터
    event_payload: dict

    # Memory에서 불러온 이력
    memory_context: dict

    # ReAct 추론 과정 (add_messages reducer로 누적)
    messages: Annotated[list[BaseMessage], add_messages]

    # 에이전트 판단 결과 (ReAct 루프 완료 후 파싱)
    # {
    #   "fault_type": str,           # inner_race / outer_race / rolling_element / cage / none / unknown
    #   "fault_stage": int,          # 0: 정상, 1~4: P-F 곡선 단계
    #   "degradation_speed": str,    # stable / normal / accelerating / abnormal
    #   "rul_assessment": {
    #     "ml_rul_hours": float | None,
    #     "agent_assessment": str,
    #     "confidence_level": str
    #   },
    #   "risk_level": str,           # normal / watch / warning / critical
    #   "recommendation": str,
    #   "uncertainty_notes": str,
    #   "reasoning_summary": str
    # }
    diagnosis_result: dict

    # Tool 호출 관련
    tool_calls_count: int
    deep_research_activated: bool

    # 리포트 및 작업지시서
    report: str
    work_order: str

    # 워크플로우 제어
    next_action: str  # continue_reasoning / call_tool / generate_report / end
