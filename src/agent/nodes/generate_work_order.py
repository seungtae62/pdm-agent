"""generate_work_order 노드 — 구조화 작업지시서 생성.

Warning/Critical 위험도에서만 실행된다.
LLM이 RAG 작업지시서와 동일한 JSON 스키마로 출력하고,
결정적 필드(wo_number, equipment, location, request_date)는 코드에서 오버라이드한다.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from agent.prompts.templates import WORK_ORDER_PROMPT
from agent.state import PdMAgentState

logger = logging.getLogger(__name__)


def _parse_json_response(text: str) -> dict:
    """LLM 응답에서 JSON 파싱. 코드펜스 제거 fallback 포함."""
    # 코드펜스 제거
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip().rstrip("`")
    return json.loads(cleaned)


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
        return {"work_order": {}}

    # 설비 메타데이터 추출
    event_payload = state.get("event_payload", {})
    equipment_meta = event_payload.get("equipment_meta", {})
    equipment_id = equipment_meta.get("equipment_id", "")
    equipment_name = equipment_meta.get("equipment_name", "")
    location = equipment_meta.get("location", "")
    equipment_info = f"{equipment_name} / {equipment_id}" if equipment_name else equipment_id

    # wo_number 자동 생성: WO-{날짜}-{event_id 뒤 3자리}
    event_id = event_payload.get("event_id", "000")
    event_suffix = event_id[-3:] if len(event_id) >= 3 else event_id
    now = datetime.now(timezone.utc)
    wo_number = f"WO-{now.strftime('%Y%m%d')}-{event_suffix}"
    request_date = now.strftime("%Y-%m-%d")

    # work_type 매핑
    work_type = "긴급수리" if risk_level == "critical" else "예방정비"

    # Warning/Critical: LLM으로 구조화 작업지시서 생성
    report = state.get("report", "")
    prompt = WORK_ORDER_PROMPT.format(
        diagnosis_result=json.dumps(diagnosis, indent=2, ensure_ascii=False),
        report=report,
        equipment_info=equipment_info,
        location=location,
        wo_number=wo_number,
        request_date=request_date,
        work_type=work_type,
    )

    try:
        response = llm.invoke([
            SystemMessage(
                content=(
                    "당신은 예지보전 정비 작업지시서를 작성하는 전문가입니다. "
                    "반드시 유효한 JSON만 출력하세요. "
                    "모든 텍스트 값은 한국어로 작성하세요. "
                    "도메인 전문 용어와 고유명사는 원어 그대로 사용합니다."
                )
            ),
            HumanMessage(content=prompt),
        ])
        work_order = _parse_json_response(response.content)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"[generate_work_order] JSON 파싱 실패: {e}")
        # Fallback: 최소 구조 반환
        work_order = {
            "summary": response.content if "response" in dir() else "",
            "checklist": [],
            "materials": [],
            "tools": [],
            "post_checks": [],
            "attachments": [],
        }

    # 결정적 필드 오버라이드
    work_order["wo_number"] = wo_number
    work_order["equipment"] = equipment_info
    work_order["location"] = location
    work_order["request_date"] = request_date
    work_order["work_type"] = work_type

    logger.info(
        f"[generate_work_order] {risk_level} — 구조화 작업지시서 생성 완료 "
        f"(wo_number={wo_number})"
    )
    return {"work_order": work_order}
