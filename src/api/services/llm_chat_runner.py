"""LLM 기반 채팅 러너."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.config import AgentConfig, create_chat_model
from agent.skills.registry import load_user_skills
from api.models.stream import ChatCompletedEvent, ChatErrorEvent, ChatTokenEvent
from api.services.chat_prompt import build_chat_system_prompt
from api.services.chat_tools import get_chat_tools
from api.services.run_manager import RunManager
from api.services.slash_commands import handle_slash_command

logger = logging.getLogger(__name__)


class LLMChatRunner:
    """실제 LLM을 사용하는 채팅 러너.

    OpenAI/Anthropic 등 LangChain ChatModel을 통해 대화하고,
    토큰 단위 스트리밍 + Skill 관리 도구 호출을 지원한다.
    """

    def __init__(self, config: AgentConfig) -> None:
        self._llm = create_chat_model(config)

    async def run(
        self,
        run_id: str,
        session_id: str,
        message: str,
        run_manager: RunManager,
    ) -> None:
        """채팅 메시지를 처리하고 응답을 스트리밍."""
        now = lambda: datetime.now(timezone.utc).isoformat()

        try:
            session = run_manager.get_chat_session(session_id)
            if not session:
                raise ValueError(f"Chat session not found: {session_id}")

            user_id = session.user_id

            # 슬래시 커맨드 처리 (LLM 호출 스킵)
            if message.startswith("/"):
                result = handle_slash_command(message, user_id)
                if result and result.handled:
                    await run_manager.emit_chat_event(
                        session_id,
                        ChatCompletedEvent(
                            run_id=run_id,
                            session_id=session_id,
                            content=result.content,
                            timestamp=now(),
                        ),
                    )
                    session.message_history.append(
                        {"role": "assistant", "content": result.content}
                    )
                    return

            # Run context 조회 (연결된 run이 있으면 진단 결과 포함)
            run_context = None
            if run_id:
                run_info = run_manager.get_run(run_id)
                if run_info and run_info.diagnosis_result:
                    run_context = {
                        "diagnosis": run_info.diagnosis_result,
                    }
                    if run_info.report:
                        run_context["report"] = run_info.report
                    if run_info.work_order:
                        run_context["work_order"] = run_info.work_order

            # User skills 로드
            user_skills_text = load_user_skills(user_id, "chat")

            # 시스템 프롬프트 구성
            system_prompt = build_chat_system_prompt(run_context, user_skills_text)

            # 메시지 히스토리 변환
            messages = [SystemMessage(content=system_prompt)]
            for msg in session.message_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

            # Skill 관리 도구 바인딩
            tools = get_chat_tools(user_id)
            llm_with_tools = self._llm.bind_tools(tools)

            # 첫 호출 — tool_calls 여부 확인
            first_response = await llm_with_tools.ainvoke(messages)
            tool_calls = getattr(first_response, "tool_calls", None) or []

            if tool_calls:
                # Tool 실행
                from langchain_core.messages import ToolMessage

                messages.append(first_response)

                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    # 매칭되는 tool 찾아서 실행
                    result = ""
                    for t in tools:
                        if t.name == tool_name:
                            result = await t.ainvoke(tool_args)
                            break
                    else:
                        result = f"Unknown tool: {tool_name}"

                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tc["id"])
                    )

                # Tool 결과 포함하여 최종 응답 스트리밍
                full_content = ""
                async for chunk in llm_with_tools.astream(messages):
                    token = chunk.content
                    if token:
                        full_content += token
                        await run_manager.emit_chat_event(
                            session_id,
                            ChatTokenEvent(
                                run_id=run_id,
                                session_id=session_id,
                                token=token,
                            ),
                        )
            else:
                # Tool 호출 없음 — 첫 응답을 스트리밍으로 재실행
                full_content = ""
                async for chunk in llm_with_tools.astream(messages):
                    token = chunk.content
                    if token:
                        full_content += token
                        await run_manager.emit_chat_event(
                            session_id,
                            ChatTokenEvent(
                                run_id=run_id,
                                session_id=session_id,
                                token=token,
                            ),
                        )

            # 완료 이벤트
            await run_manager.emit_chat_event(
                session_id,
                ChatCompletedEvent(
                    run_id=run_id,
                    session_id=session_id,
                    content=full_content,
                    timestamp=now(),
                ),
            )

            # 히스토리에 assistant 응답 저장
            session.message_history.append(
                {"role": "assistant", "content": full_content}
            )

        except Exception as e:
            logger.error("[llm_chat] 오류: %s", e, exc_info=True)
            await run_manager.emit_chat_event(
                session_id,
                ChatErrorEvent(
                    run_id=run_id,
                    session_id=session_id,
                    message=str(e),
                    timestamp=now(),
                ),
            )

        finally:
            await run_manager.end_chat_stream(session_id)
