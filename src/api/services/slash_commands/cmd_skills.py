"""/skills 슬래시 커맨드 핸들러."""

from __future__ import annotations

from agent.skills.actions.notification import get_notification_tools
from agent.skills.actions.rag_search import get_action_tools
from agent.skills.actions.web_search import get_web_search_tools
from agent.skills.registry import SKILL_REGISTRY, list_user_skills
from api.services.slash_commands.base import SlashCommandResult


def handle_skills(user_id: str) -> SlashCommandResult:
    """Knowledge / Action / User Skills를 분류하여 마크다운으로 반환."""
    sections: list[str] = []

    # --- Knowledge Skills ---
    lines = ["#### Knowledge Skills\n"]
    for entry in SKILL_REGISTRY:
        lines.append(f"- **{entry.name}**: {entry.description}")
    sections.append("\n".join(lines))

    # --- Action Skills ---
    lines = ["#### Action Skills\n"]
    action_tools = get_action_tools() + get_notification_tools() + get_web_search_tools()
    for t in action_tools:
        lines.append(f"- **{t.name}**: {t.description}")
    sections.append("\n".join(lines))

    # --- User Skills ---
    lines = ["#### User Skills\n"]
    user_skills = list_user_skills(user_id)
    if user_skills:
        for skill in user_skills:
            lines.append(f"- **{skill['name']}**: {skill.get('description', '')}")
    else:
        lines.append("등록된 스킬이 없습니다.")
    sections.append("\n".join(lines))

    content = "\n\n".join(sections)
    return SlashCommandResult(content=content, handled=True)
