"""Agent Skills 레지스트리 및 로더.

조건부로 도메인 지식 Skill을 로드하여 LLM 컨텍스트에 주입한다.
Core Skills는 코드 내장, User Skills는 파일 시스템 기반 CRUD.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

logger = logging.getLogger(__name__)

CORE_SKILLS_DIR = Path(__file__).parent / "core"
USER_SKILLS_DIR = Path(__file__).parent / "users"


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
    """State 조건에 맞는 Core Skills를 로드하여 결합된 텍스트를 반환.

    Args:
        state: PdMAgentState dict.

    Returns:
        로드된 Skills 텍스트. 없으면 빈 문자열.
    """
    matched: list[tuple[int, str, str]] = []

    for entry in SKILL_REGISTRY:
        try:
            if entry.condition(state):
                skill_path = CORE_SKILLS_DIR / entry.filename
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


# ---------------------------------------------------------------------------
# User Skills CRUD
# ---------------------------------------------------------------------------

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# condition 문자열 → 매칭 context 매핑
_CONDITION_CONTEXTS: dict[str, set[str]] = {
    "always": {"chat", "agent", "anomaly_detected", "alert"},
    "chat": {"chat"},
    "agent": {"agent", "anomaly_detected", "alert"},
    "anomaly_detected": {"anomaly_detected"},
    "alert": {"alert"},
}


def _user_dir(user_id: str) -> Path:
    """사용자별 skill 디렉토리 경로."""
    return USER_SKILLS_DIR / user_id


def _parse_user_skill(filepath: Path) -> dict | None:
    """User skill .md 파일을 파싱하여 메타데이터 + content를 반환."""
    try:
        raw = filepath.read_text(encoding="utf-8")
    except Exception:
        return None

    meta: dict = {}
    body = raw

    match = _FRONT_MATTER_RE.match(raw)
    if match:
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        body = raw[match.end():]

    return {
        "name": meta.get("name", filepath.stem),
        "description": meta.get("description", ""),
        "condition": meta.get("condition", "always"),
        "priority": meta.get("priority", 50),
        "content": body.strip(),
        "filepath": str(filepath),
    }


def load_user_skills(user_id: str, context: str) -> str:
    """사용자의 Skill 중 context에 매칭되는 것을 로드.

    Args:
        user_id: 사용자 ID.
        context: 현재 컨텍스트 ("chat", "agent", "anomaly_detected", "alert").

    Returns:
        매칭된 user skills 텍스트. 없으면 빈 문자열.
    """
    user_dir = _user_dir(user_id)
    if not user_dir.exists():
        return ""

    skills: list[tuple[int, str, str]] = []

    for md_file in sorted(user_dir.glob("*.md")):
        parsed = _parse_user_skill(md_file)
        if not parsed:
            continue

        # condition 매칭
        allowed = _CONDITION_CONTEXTS.get(parsed["condition"], set())
        if context not in allowed:
            continue

        skills.append((parsed["priority"], parsed["name"], parsed["content"]))

    if not skills:
        return ""

    skills.sort(key=lambda x: x[0])
    parts = [f"### User Skill: {name}\n\n{content}" for _, name, content in skills]

    logger.info(
        "[skills] user=%s %d개 user skill 로드: %s",
        user_id,
        len(skills),
        [s[1] for s in skills],
    )

    return "\n\n---\n\n".join(parts)


def list_user_skills(user_id: str) -> list[dict]:
    """사용자의 모든 Skill 목록을 반환."""
    user_dir = _user_dir(user_id)
    if not user_dir.exists():
        return []

    result = []
    for md_file in sorted(user_dir.glob("*.md")):
        parsed = _parse_user_skill(md_file)
        if parsed:
            parsed.pop("content", None)
            result.append(parsed)

    return result


def save_user_skill(
    user_id: str,
    name: str,
    description: str,
    condition: str,
    content: str,
    priority: int = 50,
) -> str:
    """User Skill을 파일로 저장.

    Args:
        user_id: 사용자 ID.
        name: Skill 이름 (파일명에도 사용).
        description: Skill 설명.
        condition: 조건 ("always", "chat", "agent", "anomaly_detected", "alert").
        content: Skill 본문.
        priority: 우선순위 (기본 50).

    Returns:
        저장된 파일 경로.
    """
    user_dir = _user_dir(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    # 파일명: name에서 공백/특수문자를 하이픈으로
    safe_name = re.sub(r"[^a-zA-Z0-9가-힣_-]", "-", name).strip("-")
    if not safe_name:
        safe_name = "unnamed-skill"
    filepath = user_dir / f"{safe_name}.md"

    front_matter = yaml.dump(
        {
            "name": name,
            "description": description,
            "condition": condition,
            "priority": priority,
        },
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()

    file_content = f"---\n{front_matter}\n---\n\n{content}\n"
    filepath.write_text(file_content, encoding="utf-8")

    logger.info("[skills] user=%s skill 저장: %s → %s", user_id, name, filepath)
    return str(filepath)


def delete_user_skill(user_id: str, name: str) -> bool:
    """User Skill을 삭제.

    Args:
        user_id: 사용자 ID.
        name: 삭제할 Skill 이름.

    Returns:
        삭제 성공 여부.
    """
    user_dir = _user_dir(user_id)
    if not user_dir.exists():
        return False

    # 이름으로 파일 검색
    for md_file in user_dir.glob("*.md"):
        parsed = _parse_user_skill(md_file)
        if parsed and parsed["name"] == name:
            md_file.unlink()
            logger.info("[skills] user=%s skill 삭제: %s", user_id, name)
            return True

    return False
