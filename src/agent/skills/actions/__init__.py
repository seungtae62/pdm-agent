"""Action Skills — 외부 데이터 조회를 수행하는 실행형 Skills.

Knowledge Skills가 프롬프트에 텍스트를 주입하는 반면,
Action Skills는 LangGraph Tool로 등록되어 LLM이 호출할 수 있다.

MCP-Skills 역할 분리 원칙:
- Action Skills: 과거 이력 검색 등 에이전트 판단에 필요한 데이터 조회
- MCP: 알림 발송 등 외부 시스템과의 실제 상호작용
"""

from agent.skills.actions.web_search import get_web_search_tools

__all__ = ["get_web_search_tools"]
