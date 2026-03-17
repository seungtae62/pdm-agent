"""슬래시 커맨드 기본 데이터 클래스."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SlashCommandResult:
    """슬래시 커맨드 실행 결과."""

    content: str
    handled: bool
