"""generate_work_order 노드 — 작업지시서 생성.

Warning/Critical 위험도에서만 실행된다.
Normal/Watch에서는 건너뛴다.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent.prompts.templates import WORK_ORDER_PROMPT
from agent.state import PdMAgentState

logger = logging.getLogger(__name__)


def generate_work_order(state: PdMAgentState, *, llm: BaseChatModel) -> dict:
    """작업지시서 생성 노드.

    Args:
        state: 현재 State.
        llm: LangChain ChatModel.

    Returns:
        State 업데이트 dict.
    """
    diagnosis = state.get("diagnosis_result", {})
    risk_level = diagnosis.get("risk_level", "normal")

    # Normal/Watch: 작업지시서 건너뜀
    if risk_level in ("normal", "watch"):
        logger.info(f"[generate_work_order] {risk_level} — 작업지시서 건너뜀")
        return {"work_order": ""}

    # Warning/Critical: LLM으로 작업지시서 생성
    report = state.get("report", "")
    prompt = WORK_ORDER_PROMPT.format(
        diagnosis_result=json.dumps(diagnosis, indent=2, ensure_ascii=False),
        report=report,
    )

    response = llm.invoke([
        SystemMessage(content="당신은 예지보전 정비 작업지시서를 작성하는 전문가입니다."),
        HumanMessage(content=prompt),
    ])

    work_order = response.content
    logger.info(f"[generate_work_order] {risk_level} — 작업지시서 생성 완료 ({len(work_order)}자)")
    return {"work_order": work_order}
