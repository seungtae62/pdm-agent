"""슬래시 커맨드 라우터."""

from __future__ import annotations

from typing import Callable

from api.services.slash_commands.base import SlashCommandResult
from api.services.slash_commands.cmd_skills import handle_skills

# 커맨드명 → 핸들러(user_id) 매핑
COMMAND_REGISTRY: dict[str, Callable[[str], SlashCommandResult]] = {
    "skills": handle_skills,
}


def handle_slash_command(message: str, user_id: str) -> SlashCommandResult | None:
    """슬래시 커맨드 라우터.

    "/"로 시작하는 메시지를 파싱하여 등록된 핸들러로 디스패치.
    미등록 커맨드면 None 반환 → LLM fallback.
    """
    stripped = message.strip()
    if not stripped.startswith("/"):
        return None

    command = stripped.split()[0].lstrip("/").lower()
    handler = COMMAND_REGISTRY.get(command)
    if handler is None:
        return None

    return handler(user_id)
