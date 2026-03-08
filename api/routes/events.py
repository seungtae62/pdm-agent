"""POST /api/events endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from api.dependencies import get_agent_runner, get_run_manager
from api.models.events import EventPayload
from api.services.agent_runner import AgentRunner
from api.services.run_manager import RunManager

router = APIRouter(prefix="/api", tags=["events"])


@router.post("/events")
async def receive_event(
    payload: EventPayload,
    run_manager: RunManager = Depends(get_run_manager),
    agent_runner: AgentRunner = Depends(get_agent_runner),
) -> dict:
    """Receive an event payload from Edge systems.

    Creates a new run and triggers agent execution in the background.

    Returns:
        run_id and accepted status.
    """
    run_info = run_manager.create_run(event_id=payload.event_id)

    asyncio.create_task(
        agent_runner.run(payload, run_manager, run_info.run_id)
    )

    return {"run_id": run_info.run_id, "status": "accepted"}
