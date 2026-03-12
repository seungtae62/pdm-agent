"""Agent Skills 레지스트리 및 로더.

조건부로 도메인 지식 Skill을 로드하여 LLM 컨텍스트에 주입한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent


@dataclass
class SkillEntry:
    """Skill 레지스트리 항목."""

    name: str
    filename: str
    description: str
    condition: Callable[[dict], bool]
    priority: int = 0  # 낮을수록 먼저 로드


def _is_anomaly(state: dict) -> bool:
    """anomaly_detected가 true인지 확인."""
    payload = state.get("event_payload", {})
    adr = payload.get("anomaly_detection_result", {})
    return adr.get("anomaly_detected", False)


def _is_deep_research(state: dict) -> bool:
    """deep_research_activated가 true인지 확인."""
    return state.get("deep_research_activated", False)


def _is_alert(state: dict) -> bool:
    """health_state가 warning 또는 critical인지 확인."""
    payload = state.get("event_payload", {})
    adr = payload.get("anomaly_detection_result", {})
    health = adr.get("health_state", "normal").lower()
    return health in ("warning", "critical")


SKILL_REGISTRY: list[SkillEntry] = [
    SkillEntry(
        name="fault-diagnosis",
        filename="fault_diagnosis.md",
        description="베어링 결함 주파수 해석 및 P-F 곡선 단계 판정",
        condition=_is_anomaly,
        priority=0,
    ),
    SkillEntry(
        name="feature-interpret",
        filename="feature_interpret.md",
        description="특징량 복합 해석 패턴",
        condition=_is_anomaly,
        priority=1,
    ),
    SkillEntry(
        name="deep-research",
        filename="deep_research.md",
        description="분석적 심층 조사 절차",
        condition=_is_deep_research,
        priority=2,
    ),
    SkillEntry(
        name="response-normal",
        filename="response_normal.md",
        description="Normal/Watch 위험도 응답 양식",
        condition=lambda state: not _is_alert(state),
        priority=10,
    ),
    SkillEntry(
        name="response-alert",
        filename="response_alert.md",
        description="Warning/Critical 위험도 응답 양식",
        condition=_is_alert,
        priority=10,
    ),
]


def load_matching_skills(state: dict) -> str:
    """State 조건에 맞는 Skills를 로드하여 결합된 텍스트를 반환.

    Args:
        state: PdMAgentState dict.

    Returns:
        로드된 Skills 텍스트. 없으면 빈 문자열.
    """
    matched: list[tuple[int, str, str]] = []

    for entry in SKILL_REGISTRY:
        try:
            if entry.condition(state):
                skill_path = SKILLS_DIR / entry.filename
                if skill_path.exists():
                    content = skill_path.read_text(encoding="utf-8")
                    matched.append((entry.priority, entry.name, content))
                    logger.info("[skills] 로드: %s", entry.name)
                else:
                    logger.warning("[skills] 파일 없음: %s", entry.filename)
        except Exception as e:
            logger.error("[skills] %s 조건 평가 실패: %s", entry.name, e)

    if not matched:
        logger.info("[skills] 매칭된 Skill 없음")
        return ""

    # priority 순으로 정렬
    matched.sort(key=lambda x: x[0])

    parts = []
    for _, name, content in matched:
        parts.append(f"### Skill: {name}\n\n{content}")

    loaded_names = [m[1] for m in matched]
    logger.info("[skills] %d개 로드 완료: %s", len(matched), loaded_names)

    return "\n\n---\n\n".join(parts)
