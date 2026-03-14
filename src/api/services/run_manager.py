"""Run manager for tracking agent execution state."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from api.models.stream import AgentEvent


class RunStatus(str, Enum):
    """Agent run status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunInfo:
    """Information about a single agent run."""

    run_id: str
    event_id: str
    status: RunStatus = RunStatus.PENDING
    created_at: str = ""
    completed_at: str | None = None
    error: str | None = None
    event_queue: asyncio.Queue[AgentEvent | None] = field(
        default_factory=asyncio.Queue
    )

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class RunManager:
    """In-memory run state manager (PoC).

    Manages run lifecycle and event queues for SSE streaming.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunInfo] = {}

    def create_run(self, event_id: str) -> RunInfo:
        """Create a new run and return its info."""
        run_id = str(uuid.uuid4())
        run_info = RunInfo(run_id=run_id, event_id=event_id)
        self._runs[run_id] = run_info
        return run_info

    def get_run(self, run_id: str) -> RunInfo | None:
        """Get run info by run_id."""
        return self._runs.get(run_id)

    def set_status(self, run_id: str, status: RunStatus) -> None:
        """Update run status."""
        run_info = self._runs.get(run_id)
        if run_info:
            run_info.status = status
            if status in (RunStatus.COMPLETED, RunStatus.FAILED):
                run_info.completed_at = datetime.now(timezone.utc).isoformat()

    async def emit_event(self, run_id: str, event: AgentEvent) -> None:
        """Push an event to the run's queue."""
        run_info = self._runs.get(run_id)
        if run_info:
            await run_info.event_queue.put(event)

    async def end_stream(self, run_id: str) -> None:
        """Signal end of stream by pushing None sentinel."""
        run_info = self._runs.get(run_id)
        if run_info:
            await run_info.event_queue.put(None)
