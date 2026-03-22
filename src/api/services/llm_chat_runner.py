"""LLM 기반 채팅 러너."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.config import AgentConfig, create_chat_model
from agent.skills.registry import load_user_skills
from api.models.stream import (
    ChatCompletedEvent,
    ChatErrorEvent,
    ChatTokenEvent,
    DeepSearchStepEvent,
)
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
        *,
        deep_search: bool = False,
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

            # Deep Search 모드
            if deep_search:
                await self._run_deep_search(run_id, session_id, message, run_manager)
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

    async def _run_deep_search(
        self,
        run_id: str,
        session_id: str,
        message: str,
        run_manager: RunManager,
    ) -> None:
        """Deep Search 모드로 채팅 메시지를 처리."""
        now = lambda: datetime.now(timezone.utc).isoformat()

        try:
            from agent.deep_search.graph import build_deep_search_graph
            from agent.deep_search.state import DeepSearchState

            # 1. started 이벤트
            await run_manager.emit_chat_event(
                session_id,
                DeepSearchStepEvent(
                    run_id=run_id,
                    session_id=session_id,
                    step_type="started",
                    role="Deep Research",
                    content="심층 분석을 시작합니다...",
                    status="thinking",
                ),
            )

            # reasoning_context 구성 (run의 diagnosis_result 활용)
            reasoning_context = ""
            if run_id:
                run_info = run_manager.get_run(run_id)
                if run_info and run_info.diagnosis_result:
                    import json

                    ctx_parts = []
                    ctx_parts.append(
                        f"진단 결과: {json.dumps(run_info.diagnosis_result, ensure_ascii=False)}"
                    )
                    if run_info.report:
                        ctx_parts.append(f"분석 보고서: {run_info.report}")
                    if run_info.work_order:
                        ctx_parts.append(
                            f"작업 지시서: {json.dumps(run_info.work_order, ensure_ascii=False)}"
                        )
                    reasoning_context = "\n\n".join(ctx_parts)

            # 2. 그래프 빌드
            config = AgentConfig.from_env()
            graph = await build_deep_search_graph(config)

            initial_state: DeepSearchState = {
                "original_query": message,
                "reasoning_context": reasoning_context,
                "perspectives": [],
                "search_results": [],
                "critic_feedback": [],
                "synthesis": "",
                "citations": [],
                "iteration_count": 0,
                "max_iterations": 3,
            }

            # 3. astream_events로 그래프 실행 및 이벤트 매핑
            current_node: str = ""
            perspectives: list[dict] = []
            search_results: list[dict] = []
            critic_feedback: list[dict] = []
            current_researcher_index: int = 0

            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event.get("event", "")
                name = event.get("name", "")

                # ── on_chain_start: 노드 진입 ──
                if kind == "on_chain_start":
                    if name == "decompose":
                        current_node = "decompose"
                        await run_manager.emit_chat_event(
                            session_id,
                            DeepSearchStepEvent(
                                run_id=run_id,
                                session_id=session_id,
                                step_type="decompose",
                                role="Leader",
                                content="",
                                status="thinking",
                            ),
                        )
                    elif name == "research":
                        current_node = "research"
                        # 병렬 실행: 모든 perspective를 동시에 "thinking" 상태로 전송
                        total = len(perspectives) if perspectives else 0
                        for idx, p in enumerate(perspectives):
                            p_name = p.get("perspective", p.get("name", ""))
                            role_label = (
                                f"Researcher {idx + 1}/{total}: {p_name}"
                                if p_name
                                else f"Researcher {idx + 1}/{total}"
                            )
                            await run_manager.emit_chat_event(
                                session_id,
                                DeepSearchStepEvent(
                                    run_id=run_id,
                                    session_id=session_id,
                                    step_type="researcher",
                                    role=role_label,
                                    content="",
                                    status="thinking",
                                    perspective_index=idx,
                                    total_perspectives=total,
                                ),
                            )
                    elif name == "review":
                        current_node = "review"
                        await run_manager.emit_chat_event(
                            session_id,
                            DeepSearchStepEvent(
                                run_id=run_id,
                                session_id=session_id,
                                step_type="critic",
                                role="Critic",
                                content="",
                                status="thinking",
                            ),
                        )
                    elif name == "synthesize":
                        current_node = "synthesize"
                        await run_manager.emit_chat_event(
                            session_id,
                            DeepSearchStepEvent(
                                run_id=run_id,
                                session_id=session_id,
                                step_type="synthesis",
                                role="Leader",
                                content="",
                                status="thinking",
                            ),
                        )

                # ── on_chat_model_stream: 토큰 스트리밍 ──
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {})
                    if hasattr(chunk, "content"):
                        token = chunk.content
                    elif isinstance(chunk, dict):
                        token = chunk.get("content", "")
                    else:
                        token = ""

                    if token and current_node:
                        step_type_map = {
                            "decompose": "decompose",
                            "research": "researcher",
                            "review": "critic",
                            "synthesize": "synthesis",
                        }
                        role_map = {
                            "decompose": "Leader",
                            "review": "Critic",
                            "synthesize": "Leader",
                        }
                        step_type = step_type_map.get(current_node, current_node)
                        role = role_map.get(current_node, "")

                        if current_node == "research":
                            total = len(perspectives) if perspectives else 0
                            role = (
                                f"Researcher {current_researcher_index + 1}/{total}"
                                if total > 0
                                else "Researcher"
                            )
                            if perspectives and current_researcher_index < len(
                                perspectives
                            ):
                                p = perspectives[current_researcher_index]
                                p_name = p.get("perspective", p.get("name", ""))
                                if p_name:
                                    role += f": {p_name}"

                        await run_manager.emit_chat_event(
                            session_id,
                            DeepSearchStepEvent(
                                run_id=run_id,
                                session_id=session_id,
                                step_type=step_type,
                                role=role,
                                content=token,
                                status="thinking",
                                perspective_index=(
                                    current_researcher_index
                                    if current_node == "research"
                                    else None
                                ),
                                total_perspectives=(
                                    len(perspectives)
                                    if current_node == "research" and perspectives
                                    else None
                                ),
                            ),
                        )

                # ── on_chain_end: 노드 완료 ──
                elif kind == "on_chain_end":
                    if name == "decompose":
                        # decompose 완료 후 perspectives 추출
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            perspectives = output.get("perspectives", [])
                        total = len(perspectives)
                        perspective_names = [
                            p.get("perspective", p.get("name", ""))
                            for p in perspectives
                        ]
                        await run_manager.emit_chat_event(
                            session_id,
                            DeepSearchStepEvent(
                                run_id=run_id,
                                session_id=session_id,
                                step_type="decompose",
                                role="Leader",
                                content=", ".join(perspective_names),
                                status="done",
                                total_perspectives=total if total > 0 else None,
                            ),
                        )
                    elif name == "research":
                        # research 완료 후 search_results 추출 (병렬 실행 결과)
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            new_results = output.get("search_results", [])
                            if new_results:
                                search_results = new_results

                        # 병렬 실행: 모든 perspective 결과를 개별 이벤트로 발송
                        total = len(perspectives) if perspectives else len(search_results)
                        for idx, sr in enumerate(search_results):
                            p_name = sr.get("perspective", "")
                            confidence_val = sr.get("confidence")
                            role_label = f"Researcher {idx + 1}/{total}: {p_name}"

                            await run_manager.emit_chat_event(
                                session_id,
                                DeepSearchStepEvent(
                                    run_id=run_id,
                                    session_id=session_id,
                                    step_type="researcher",
                                    role=role_label,
                                    content="",
                                    status="done",
                                    perspective_index=idx,
                                    total_perspectives=total,
                                    confidence=confidence_val,
                                ),
                            )
                        current_researcher_index = len(search_results)
                    elif name == "review":
                        # review 완료 후 critic_feedback 추출
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            new_feedback = output.get("critic_feedback", [])
                            if new_feedback:
                                critic_feedback = new_feedback

                        # Critic 결과 요약 생성
                        passed_count = sum(
                            1 for cf in critic_feedback if cf.get("passed", False)
                        )
                        total = len(critic_feedback)
                        feedback_lines = []
                        for cf in critic_feedback:
                            p = cf.get("perspective", "")
                            status_label = "Pass" if cf.get("passed") else "Revise"
                            feedback_lines.append(f"{p}: [{status_label}]")

                        content = (
                            f"{passed_count}/{total} 통과\n" + "\n".join(feedback_lines)
                            if total > 0
                            else ""
                        )

                        await run_manager.emit_chat_event(
                            session_id,
                            DeepSearchStepEvent(
                                run_id=run_id,
                                session_id=session_id,
                                step_type="critic",
                                role="Critic",
                                content=content,
                                status="done",
                            ),
                        )
                    elif name == "LangGraph":
                        # 그래프 전체 완료 — 최종 synthesis 추출
                        output = event.get("data", {}).get("output", {})
                        synthesis = ""
                        if isinstance(output, dict):
                            synthesis = output.get("synthesis", "")

                        # ── Weighted Voting 계산 및 이벤트 ──
                        voting_weights: list[dict] = []
                        total_score = 0.0
                        for sr in search_results:
                            perspective = sr.get("perspective", "")
                            confidence = sr.get("confidence", 0.0)
                            # 해당 perspective의 critic feedback 찾기
                            cf_passed = True
                            for cf in critic_feedback:
                                if cf.get("perspective") == perspective:
                                    cf_passed = cf.get("passed", True)
                                    break
                            # Weight = confidence * critic_multiplier
                            critic_mult = 1.0 if cf_passed else 0.5
                            score = confidence * critic_mult
                            total_score += score
                            voting_weights.append(
                                {
                                    "perspective": perspective,
                                    "confidence": confidence,
                                    "passed": cf_passed,
                                    "score": score,
                                }
                            )

                        # Normalize to percentages
                        for vw in voting_weights:
                            vw["weight"] = round(
                                (
                                    (vw["score"] / total_score * 100)
                                    if total_score > 0
                                    else 0
                                ),
                                1,
                            )

                        # Sort by weight descending
                        voting_weights.sort(key=lambda x: x["weight"], reverse=True)

                        if voting_weights:
                            # Emit voting event
                            voting_lines = []
                            for vw in voting_weights:
                                bar_len = int(vw["weight"] / 10)
                                bar = "\u25a0" * bar_len + "\u2591" * (10 - bar_len)
                                voting_lines.append(
                                    f"{bar}  {vw['perspective']}  {vw['weight']}%"
                                )

                            await run_manager.emit_chat_event(
                                session_id,
                                DeepSearchStepEvent(
                                    run_id=run_id,
                                    session_id=session_id,
                                    step_type="voting",
                                    role="Weighted Voting",
                                    content="\n".join(voting_lines),
                                    status="done",
                                    voting_weights=voting_weights,
                                ),
                            )

                        # synthesis 완료 이벤트
                        await run_manager.emit_chat_event(
                            session_id,
                            DeepSearchStepEvent(
                                run_id=run_id,
                                session_id=session_id,
                                step_type="synthesis",
                                role="Leader",
                                content="",
                                status="done",
                            ),
                        )

                        # ChatCompletedEvent
                        await run_manager.emit_chat_event(
                            session_id,
                            ChatCompletedEvent(
                                run_id=run_id,
                                session_id=session_id,
                                content=synthesis,
                                timestamp=now(),
                            ),
                        )

                        # 히스토리 저장
                        chat_session = run_manager.get_chat_session(session_id)
                        if chat_session:
                            chat_session.message_history.append(
                                {"role": "assistant", "content": synthesis}
                            )

        except Exception as e:
            logger.error("[llm_chat:deep_search] 오류: %s", e, exc_info=True)
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
