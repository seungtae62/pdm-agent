"""Pydantic models for SSE stream events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class RunStartedEvent(BaseModel):
    """Agent run started."""

    event: Literal["run_started"] = "run_started"
    run_id: str
    timestamp: str
    event_id: str


class NodeEnteredEvent(BaseModel):
    """Graph node entered."""

    event: Literal["node_entered"] = "node_entered"
    run_id: str
    node_name: str
    timestamp: str


class ReasoningTokenEvent(BaseModel):
    """LLM token streaming."""

    event: Literal["reasoning_token"] = "reasoning_token"
    run_id: str
    token: str


class ToolCallEvent(BaseModel):
    """MCP Tool call started."""

    event: Literal["tool_call"] = "tool_call"
    run_id: str
    tool_name: str
    arguments: dict[str, Any] = {}
    timestamp: str


class ToolResultEvent(BaseModel):
    """Tool result returned."""

    event: Literal["tool_result"] = "tool_result"
    run_id: str
    tool_name: str
    result: Any
    timestamp: str


class DiagnosisEvent(BaseModel):
    """Diagnosis result."""

    event: Literal["diagnosis"] = "diagnosis"
    run_id: str
    diagnosis: dict[str, Any]
    timestamp: str


class ReportGeneratedEvent(BaseModel):
    """Analysis report generated."""

    event: Literal["report_generated"] = "report_generated"
    run_id: str
    report: str
    timestamp: str


class WorkOrderGeneratedEvent(BaseModel):
    """Work order generated."""

    event: Literal["work_order_generated"] = "work_order_generated"
    run_id: str
    work_order: dict[str, Any]
    timestamp: str


class RunCompletedEvent(BaseModel):
    """Run completed."""

    event: Literal["run_completed"] = "run_completed"
    run_id: str
    summary: str
    timestamp: str


class ErrorEvent(BaseModel):
    """Error occurred."""

    event: Literal["error"] = "error"
    run_id: str
    message: str
    timestamp: str


class ChatTokenEvent(BaseModel):
    """Chat response token streaming."""

    event: Literal["chat_token"] = "chat_token"
    run_id: str
    session_id: str
    token: str


class ChatCompletedEvent(BaseModel):
    """Chat response completed."""

    event: Literal["chat_completed"] = "chat_completed"
    run_id: str
    session_id: str
    content: str
    timestamp: str


class ChatErrorEvent(BaseModel):
    """Chat error occurred."""

    event: Literal["chat_error"] = "chat_error"
    run_id: str
    session_id: str
    message: str
    timestamp: str


class DeepSearchStepEvent(BaseModel):
    """Deep Search step streaming."""

    event: Literal["deep_search_step"] = "deep_search_step"
    run_id: str
    session_id: str
    step_type: str  # "started" | "perspective" | "researcher" | "critic" | "synthesis"
    role: str  # e.g., "Leader", "Researcher 1/3: 재료공학", "Critic"
    content: str
    status: str  # "thinking" | "done"
    perspective_index: int | None = None  # 0-based index for researcher
    total_perspectives: int | None = None


AgentEvent = Union[
    RunStartedEvent,
    NodeEnteredEvent,
    ReasoningTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    DiagnosisEvent,
    ReportGeneratedEvent,
    WorkOrderGeneratedEvent,
    RunCompletedEvent,
    ErrorEvent,
    ChatTokenEvent,
    ChatCompletedEvent,
    ChatErrorEvent,
    DeepSearchStepEvent,
]
