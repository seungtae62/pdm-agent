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
]
