"""GET /api/agent/stream/{run_id} SSE endpoint."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.dependencies import get_run_manager
from api.services.run_manager import RunManager

router = APIRouter(prefix="/api/agent", tags=["stream"])


async def _event_generator(
    run_manager: RunManager, run_id: str
) -> AsyncGenerator[dict, None]:
    """Generate SSE events from the run's event queue."""
    run_info = run_manager.get_run(run_id)
    if not run_info:
        return

    while True:
        event = await run_info.event_queue.get()
        if event is None:
            break
        yield {
            "event": event.event,
            "data": event.model_dump_json(),
        }


@router.get("/stream/{run_id}")
async def stream_agent(
    run_id: str,
    run_manager: RunManager = Depends(get_run_manager),
) -> EventSourceResponse:
    """Stream agent execution events via SSE.

    Args:
        run_id: The run ID returned from POST /api/events.

    Returns:
        SSE event stream.

    Raises:
        HTTPException: 404 if run_id not found.
    """
    if run_manager.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return EventSourceResponse(_event_generator(run_manager, run_id))
