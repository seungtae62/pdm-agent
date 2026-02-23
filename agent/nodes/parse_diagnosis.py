"""parse_diagnosis 노드 — 진단 결과 파싱.

ReAct 루프 완료 후 LLM 응답에서 diagnosis_result를 구조화된 JSON으로 추출한다.
"""

from __future__ import annotations

import json
import logging
import re

from agent.state import PdMAgentState

logger = logging.getLogger(__name__)

# diagnosis_result 기본값
DEFAULT_DIAGNOSIS = {
    "fault_type": "unknown",
    "fault_stage": 0,
    "degradation_speed": "unknown",
    "rul_assessment": {
        "ml_rul_hours": None,
        "agent_assessment": "판단 불가",
        "confidence_level": "low",
    },
    "risk_level": "normal",
    "recommendation": "",
    "uncertainty_notes": "",
    "reasoning_summary": "",
}


def parse_diagnosis(state: PdMAgentState) -> dict:
    """진단 결과 파싱 노드.

    마지막 AI 메시지에서 JSON 블록을 추출하여 diagnosis_result로 파싱한다.

    Args:
        state: 현재 State.

    Returns:
        State 업데이트 dict.
    """
    messages = state.get("messages", [])

    # AI 메시지에서 마지막 텍스트 응답 추출
    last_content = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_call_id"):
            last_content = msg.content
            break

    if not last_content:
        logger.warning("[parse_diagnosis] AI 응답 없음, 기본값 사용")
        return {"diagnosis_result": DEFAULT_DIAGNOSIS.copy()}

    # JSON 블록 추출
    diagnosis = _extract_json(last_content)

    if diagnosis is None:
        logger.warning("[parse_diagnosis] JSON 추출 실패, 기본값 + reasoning_summary 사용")
        diagnosis = DEFAULT_DIAGNOSIS.copy()
        # 전체 응답을 reasoning_summary로 저장
        diagnosis["reasoning_summary"] = last_content[:2000]
    else:
        # 누락 필드 보완
        for key, default_val in DEFAULT_DIAGNOSIS.items():
            if key not in diagnosis:
                diagnosis[key] = default_val

    logger.info(
        f"[parse_diagnosis] 파싱 완료: "
        f"fault_type={diagnosis.get('fault_type')}, "
        f"risk_level={diagnosis.get('risk_level')}"
    )

    return {"diagnosis_result": diagnosis}


def _extract_json(text: str) -> dict | None:
    """텍스트에서 JSON 블록을 추출.

    ```json ... ``` 코드블록 또는 { ... } 패턴을 시도한다.
    """
    # 1. ```json ... ``` 코드블록
    pattern = r"```(?:json)?\s*\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and "fault_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # 2. { ... } 패턴 (가장 큰 것)
    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(brace_pattern, text, re.DOTALL)

    for match in sorted(matches, key=len, reverse=True):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and "fault_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None
