"""generate_report 노드 — 분석 리포트 생성.

Warning/Critical 위험도에서 diagnosis_result를 기반으로 분석 리포트를 생성한다.
Normal/Watch에서는 간결한 상태 확인 메시지만 생성한다.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent.prompts.templates import REPORT_PROMPT
from agent.state import PdMAgentState

logger = logging.getLogger(__name__)


def generate_report(state: PdMAgentState, *, llm: BaseChatModel) -> dict:
    """분석 리포트 생성 노드.

    Args:
        state: 현재 State.
        llm: LangChain ChatModel.

    Returns:
        State 업데이트 dict.
    """
    diagnosis = state.get("diagnosis_result", {})
    risk_level = diagnosis.get("risk_level", "normal")

    # Normal/Watch: 리포트 생성 생략
    if risk_level in ("normal", "watch"):
        summary = diagnosis.get("reasoning_summary", "")
        recommendation = diagnosis.get("recommendation", "")
        report = f"[{risk_level.upper()}] {summary}\n\n권고: {recommendation}"
        logger.info(f"[generate_report] {risk_level} — 간결 리포트")
        return {"report": report}

    # Warning/Critical: LLM으로 상세 리포트 생성
    memory = state.get("memory_context", {})
    prompt = REPORT_PROMPT.format(
        diagnosis_result=json.dumps(diagnosis, indent=2, ensure_ascii=False),
        memory_context=memory.get("history_summary", "이력 없음"),
    )

    response = llm.invoke([
        SystemMessage(content="당신은 예지보전 분석 리포트를 작성하는 전문가입니다."),
        HumanMessage(content=prompt),
    ])

    report = response.content
    logger.info(f"[generate_report] {risk_level} — 상세 리포트 생성 완료 ({len(report)}자)")
    return {"report": report}
