"""채팅용 도구 — Skill 관리 + RAG 검색."""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, tool

from agent.skills.actions.rag_search import get_action_tools
from agent.skills.registry import (
    delete_user_skill,
    list_user_skills,
    save_user_skill,
)


def get_chat_tools(user_id: str) -> list[BaseTool]:
    """user_id를 바인딩한 Skill 관리 도구 목록을 반환.

    Args:
        user_id: 현재 사용자 ID.

    Returns:
        LangChain BaseTool 리스트.
    """

    @tool
    def create_user_skill(
        name: str,
        description: str,
        condition: str,
        content: str,
    ) -> str:
        """사용자 Skill을 생성합니다.

        Args:
            name: Skill 이름 (예: vibration-trend-analysis).
            description: Skill에 대한 한 줄 설명.
            condition: 적용 조건 - always, chat, agent, anomaly_detected, alert 중 하나.
            content: Skill 본문 내용 (마크다운).
        """
        valid_conditions = {"always", "chat", "agent", "anomaly_detected", "alert"}
        if condition not in valid_conditions:
            return f"오류: condition은 {valid_conditions} 중 하나여야 합니다."

        filepath = save_user_skill(
            user_id=user_id,
            name=name,
            description=description,
            condition=condition,
            content=content,
        )
        return f"Skill '{name}'이 저장되었습니다. (경로: {filepath})"

    @tool
    def list_user_skills_tool() -> str:
        """현재 사용자의 등록된 Skill 목록을 조회합니다."""
        skills = list_user_skills(user_id)
        if not skills:
            return "등록된 Skill이 없습니다."
        return json.dumps(skills, ensure_ascii=False, indent=2)

    @tool
    def delete_user_skill_tool(name: str) -> str:
        """사용자 Skill을 삭제합니다.

        Args:
            name: 삭제할 Skill 이름.
        """
        success = delete_user_skill(user_id, name)
        if success:
            return f"Skill '{name}'이 삭제되었습니다."
        return f"Skill '{name}'을 찾을 수 없습니다."

    # Skill 관리 도구 + RAG 검색 도구
    return [
        create_user_skill,
        list_user_skills_tool,
        delete_user_skill_tool,
        *get_action_tools(),
    ]
