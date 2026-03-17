"""슬래시 커맨드 테스트."""

from __future__ import annotations

from api.services.slash_commands import handle_slash_command


class TestHandleSlashCommand:
    """handle_slash_command 라우터 테스트."""

    def test_skills_returns_three_categories(self) -> None:
        result = handle_slash_command("/skills", user_id="test-user")
        assert result is not None
        assert result.handled is True
        assert "Knowledge Skills" in result.content
        assert "Action Skills" in result.content
        assert "User Skills" in result.content

    def test_unknown_command_returns_none(self) -> None:
        result = handle_slash_command("/unknown_cmd", user_id="test-user")
        assert result is None

    def test_non_slash_message_returns_none(self) -> None:
        result = handle_slash_command("hello world", user_id="test-user")
        assert result is None

    def test_skills_lists_knowledge_entries(self) -> None:
        result = handle_slash_command("/skills", user_id="test-user")
        assert result is not None
        # SKILL_REGISTRY에 등록된 스킬이 포함되어야 함
        assert "fault-diagnosis" in result.content

    def test_skills_lists_action_tools(self) -> None:
        result = handle_slash_command("/skills", user_id="test-user")
        assert result is not None
        # RAG search tool 이름 포함 확인
        assert "search_maintenance_history" in result.content

    def test_skills_shows_no_user_skills_message(self) -> None:
        result = handle_slash_command("/skills", user_id="nonexistent-user-12345")
        assert result is not None
        assert "등록된 스킬이 없습니다" in result.content
