"""Agent runner interface and implementations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from api.models.events import EventPayload
from api.models.stream import (
    DiagnosisEvent,
    ErrorEvent,
    NodeEnteredEvent,
    ReasoningTokenEvent,
    ReportGeneratedEvent,
    RunCompletedEvent,
    RunStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
    WorkOrderGeneratedEvent,
)
from api.services.run_manager import RunManager, RunStatus

logger = logging.getLogger(__name__)

GRAPH_NODES = {
    "load_memory",
    "reasoning",
    "tool_executor",
    "parse_diagnosis",
    "generate_report",
    "generate_work_order",
    "save_memory",
}


@runtime_checkable
class AgentRunner(Protocol):
    """Protocol for agent runners."""

    async def run(
        self,
        event_payload: EventPayload,
        run_manager: RunManager,
        run_id: str,
    ) -> None:
        """Execute agent with the given event payload."""
        ...


class MockAgentRunner:
    """Mock agent runner for demo/testing.

    Simulates agent graph execution by emitting fake events
    with realistic delays.
    """

    async def run(
        self,
        event_payload: EventPayload,
        run_manager: RunManager,
        run_id: str,
    ) -> None:
        """Simulate agent execution."""
        now = lambda: datetime.now(timezone.utc).isoformat()

        try:
            run_manager.set_status(run_id, RunStatus.RUNNING)

            # run_started
            await run_manager.emit_event(
                run_id,
                RunStartedEvent(
                    run_id=run_id,
                    timestamp=now(),
                    event_id=event_payload.event_id,
                ),
            )
            await asyncio.sleep(0.5)

            # load_memory
            await run_manager.emit_event(
                run_id,
                NodeEnteredEvent(
                    run_id=run_id, node_name="load_memory", timestamp=now()
                ),
            )
            await asyncio.sleep(1.0)

            # reasoning
            await run_manager.emit_event(
                run_id,
                NodeEnteredEvent(
                    run_id=run_id, node_name="reasoning", timestamp=now()
                ),
            )
            await asyncio.sleep(0.5)

            # Stream some reasoning tokens
            is_anomaly = event_payload.anomaly_detection_result.anomaly_detected
            health = event_payload.anomaly_detection_result.health_state

            reasoning_text = (
                f"장비 {event_payload.equipment_meta.equipment_id}의 "
                f"베어링 {event_payload.equipment_meta.bearing.bearing_id} "
                f"데이터를 분석합니다. "
            )
            if is_anomaly:
                reasoning_text += (
                    f"이상 감지됨 (상태: {health}, "
                    f"점수: {event_payload.anomaly_detection_result.anomaly_score:.2f}). "
                    "주파수 영역 분석을 수행하겠습니다."
                )
            else:
                reasoning_text += "정상 범위 내 데이터입니다."

            for token in reasoning_text.split():
                await run_manager.emit_event(
                    run_id,
                    ReasoningTokenEvent(run_id=run_id, token=token + " "),
                )
                await asyncio.sleep(0.1)

            # tool_executor
            await run_manager.emit_event(
                run_id,
                NodeEnteredEvent(
                    run_id=run_id, node_name="tool_executor", timestamp=now()
                ),
            )

            if is_anomaly:
                # Tool call: frequency analysis
                await run_manager.emit_event(
                    run_id,
                    ToolCallEvent(
                        run_id=run_id,
                        tool_name="frequency_analysis",
                        arguments={
                            "equipment_id": event_payload.equipment_meta.equipment_id,
                            "channels": event_payload.equipment_meta.sensor_config.channels,
                        },
                        timestamp=now(),
                    ),
                )
                await asyncio.sleep(1.5)

                await run_manager.emit_event(
                    run_id,
                    ToolResultEvent(
                        run_id=run_id,
                        tool_name="frequency_analysis",
                        result={
                            "dominant_defect": "BPFI",
                            "severity": health,
                            "recommendation": "즉시 점검 필요"
                            if health == "critical"
                            else "모니터링 강화",
                        },
                        timestamp=now(),
                    ),
                )

            await asyncio.sleep(0.5)

            # parse_diagnosis
            await run_manager.emit_event(
                run_id,
                NodeEnteredEvent(
                    run_id=run_id,
                    node_name="parse_diagnosis",
                    timestamp=now(),
                ),
            )
            await asyncio.sleep(1.0)

            severity = "critical" if health == "critical" else (
                "warning" if is_anomaly else "normal"
            )
            diagnosis = {
                "equipment_id": event_payload.equipment_meta.equipment_id,
                "bearing_id": event_payload.equipment_meta.bearing.bearing_id,
                "severity": severity,
                "fault_type": "inner_race_defect" if is_anomaly else "none",
                "confidence": event_payload.anomaly_detection_result.confidence,
                "recommended_action": (
                    "즉시 교체"
                    if severity == "critical"
                    else "모니터링 강화"
                    if severity == "warning"
                    else "정상 운전 유지"
                ),
            }

            await run_manager.emit_event(
                run_id,
                DiagnosisEvent(
                    run_id=run_id, diagnosis=diagnosis, timestamp=now()
                ),
            )
            await asyncio.sleep(0.5)

            # generate_report
            await run_manager.emit_event(
                run_id,
                NodeEnteredEvent(
                    run_id=run_id,
                    node_name="generate_report",
                    timestamp=now(),
                ),
            )
            await asyncio.sleep(1.0)

            report = (
                f"# 진단 리포트\n\n"
                f"- 장비: {event_payload.equipment_meta.equipment_name}\n"
                f"- 베어링: {event_payload.equipment_meta.bearing.model}\n"
                f"- 상태: {severity}\n"
                f"- 가동시간: {event_payload.equipment_meta.total_running_hours}h\n"
            )
            await run_manager.emit_event(
                run_id,
                ReportGeneratedEvent(
                    run_id=run_id, report=report, timestamp=now()
                ),
            )

            # generate_work_order (only for warning/critical)
            if is_anomaly:
                await run_manager.emit_event(
                    run_id,
                    NodeEnteredEvent(
                        run_id=run_id,
                        node_name="generate_work_order",
                        timestamp=now(),
                    ),
                )
                await asyncio.sleep(1.0)

                work_order = {
                    "work_order_id": f"WO-{run_id[:8].upper()}",
                    "priority": "urgent" if severity == "critical" else "normal",
                    "equipment_id": event_payload.equipment_meta.equipment_id,
                    "task": diagnosis["recommended_action"],
                    "assigned_to": None,
                }
                await run_manager.emit_event(
                    run_id,
                    WorkOrderGeneratedEvent(
                        run_id=run_id,
                        work_order=work_order,
                        timestamp=now(),
                    ),
                )

            # save_memory
            await run_manager.emit_event(
                run_id,
                NodeEnteredEvent(
                    run_id=run_id, node_name="save_memory", timestamp=now()
                ),
            )
            await asyncio.sleep(0.5)

            # run_completed
            await run_manager.emit_event(
                run_id,
                RunCompletedEvent(
                    run_id=run_id,
                    summary=f"진단 완료: {severity} - {diagnosis['recommended_action']}",
                    timestamp=now(),
                ),
            )
            run_manager.set_status(run_id, RunStatus.COMPLETED)

        except Exception as e:
            await run_manager.emit_event(
                run_id,
                ErrorEvent(
                    run_id=run_id, message=str(e), timestamp=now()
                ),
            )
            run_manager.set_status(run_id, RunStatus.FAILED)

        finally:
            await run_manager.end_stream(run_id)


class LangGraphAgentRunner:
    """실제 LangGraph Agent를 실행하는 AgentRunner.

    LangGraph의 astream_events를 SSE 이벤트로 변환하여
    run_manager를 통해 스트리밍합니다.
    """

    def __init__(self, config: "AgentConfig | None" = None):
        from agent.config import AgentConfig

        self.config = config or AgentConfig.from_env()

    async def run(
        self,
        event_payload: EventPayload,
        run_manager: RunManager,
        run_id: str,
    ) -> None:
        """LangGraph Agent 실행 및 SSE 이벤트 스트리밍."""
        from agent.graph import build_graph

        now = lambda: datetime.now(timezone.utc).isoformat()

        try:
            run_manager.set_status(run_id, RunStatus.RUNNING)

            # run_started
            await run_manager.emit_event(
                run_id,
                RunStartedEvent(
                    run_id=run_id,
                    timestamp=now(),
                    event_id=event_payload.event_id,
                ),
            )
            logger.info(
                f"[SSE] run_started | run={run_id[:8]}"
                f" | event_id={event_payload.event_id}"
            )

            # Build graph
            graph = await build_graph(config=self.config)

            # Initial state
            payload_dict = event_payload.model_dump()
            initial_state = {
                "event_payload": payload_dict,
                "memory_context": {},
                "messages": [],
                "diagnosis_result": {},
                "tool_calls_count": 0,
                "deep_research_activated": False,
                "report": "",
                "work_order": "",
                "next_action": "",
            }

            # Track token count for logging
            token_char_count = 0
            final_state = None

            async with asyncio.timeout(300):
                async for event in graph.astream_events(
                    initial_state, version="v2"
                ):
                    kind = event["event"]
                    name = event.get("name", "")
                    data = event.get("data", {})

                    # Node entered
                    if kind == "on_chain_start" and name in GRAPH_NODES:
                        await run_manager.emit_event(
                            run_id,
                            NodeEnteredEvent(
                                run_id=run_id,
                                node_name=name,
                                timestamp=now(),
                            ),
                        )
                        logger.info(
                            f"[SSE] node_entered | run={run_id[:8]}"
                            f" | node={name}"
                        )

                    # LLM token streaming
                    elif kind == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            token_char_count += len(chunk.content)
                            await run_manager.emit_event(
                                run_id,
                                ReasoningTokenEvent(
                                    run_id=run_id,
                                    token=chunk.content,
                                ),
                            )

                    # Tool call start
                    elif kind == "on_tool_start":
                        arguments = data.get("input", {})
                        await run_manager.emit_event(
                            run_id,
                            ToolCallEvent(
                                run_id=run_id,
                                tool_name=name,
                                arguments=arguments
                                if isinstance(arguments, dict)
                                else {},
                                timestamp=now(),
                            ),
                        )
                        logger.info(
                            f"[SSE] tool_call | run={run_id[:8]}"
                            f" | tool={name}"
                        )

                    # Tool result
                    elif kind == "on_tool_end":
                        output = data.get("output", "")
                        await run_manager.emit_event(
                            run_id,
                            ToolResultEvent(
                                run_id=run_id,
                                tool_name=name,
                                result=output,
                                timestamp=now(),
                            ),
                        )
                        logger.info(
                            f"[SSE] tool_result | run={run_id[:8]}"
                            f" | tool={name}"
                        )

                    # Graph completed
                    elif kind == "on_chain_end" and name == "LangGraph":
                        final_state = data.get("output", {})

            if token_char_count > 0:
                logger.info(
                    f"[SSE] reasoning_token | run={run_id[:8]}"
                    f" | tokens={token_char_count} chars"
                )

            # Extract results from final state
            if final_state:
                # Diagnosis
                diagnosis = final_state.get("diagnosis_result", {})
                if diagnosis:
                    await run_manager.emit_event(
                        run_id,
                        DiagnosisEvent(
                            run_id=run_id,
                            diagnosis=diagnosis,
                            timestamp=now(),
                        ),
                    )
                    logger.info(
                        f"[SSE] diagnosis | run={run_id[:8]}"
                        f" | fault={diagnosis.get('fault_type', 'unknown')}"
                        f" | risk={diagnosis.get('severity', 'unknown')}"
                    )

                # Report
                report = final_state.get("report", "")
                if report:
                    await run_manager.emit_event(
                        run_id,
                        ReportGeneratedEvent(
                            run_id=run_id,
                            report=report,
                            timestamp=now(),
                        ),
                    )
                    logger.info(
                        f"[SSE] report_generated | run={run_id[:8]}"
                        f" | length={len(report)}"
                    )

                # Work order
                work_order = final_state.get("work_order", "")
                if work_order:
                    wo = (
                        work_order
                        if isinstance(work_order, dict)
                        else {"content": work_order}
                    )
                    await run_manager.emit_event(
                        run_id,
                        WorkOrderGeneratedEvent(
                            run_id=run_id,
                            work_order=wo,
                            timestamp=now(),
                        ),
                    )
                    logger.info(
                        f"[SSE] work_order | run={run_id[:8]}"
                        f" | priority={wo.get('priority', 'unknown')}"
                    )

            # Run completed
            severity = (
                final_state.get("diagnosis_result", {}).get(
                    "severity", "unknown"
                )
                if final_state
                else "unknown"
            )
            summary = f"진단 완료: {severity}"
            await run_manager.emit_event(
                run_id,
                RunCompletedEvent(
                    run_id=run_id,
                    summary=summary,
                    timestamp=now(),
                ),
            )
            run_manager.set_status(run_id, RunStatus.COMPLETED)
            logger.info(
                f"[SSE] run_completed | run={run_id[:8]}"
                f" | summary={summary}"
            )

        except Exception as e:
            logger.error(
                f"[SSE] error | run={run_id[:8]} | {e}",
                exc_info=True,
            )
            await run_manager.emit_event(
                run_id,
                ErrorEvent(
                    run_id=run_id,
                    message=str(e),
                    timestamp=now(),
                ),
            )
            run_manager.set_status(run_id, RunStatus.FAILED)

        finally:
            await run_manager.end_stream(run_id)
