"""Chat API endpoints."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.dependencies import get_chat_runner, get_run_manager
from api.models.chat import ChatRequest, ChatResponse
from api.services.chat_runner import ChatRunner
from api.services.run_manager import ChatEvent, RunManager

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
async def submit_chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
    run_manager: RunManager = Depends(get_run_manager),
    chat_runner: ChatRunner = Depends(get_chat_runner),
) -> ChatResponse:
    """Submit a chat message and get a session_id for streaming.

    If session_id is provided, reuses the existing session.
    Otherwise, creates a new session. run_id is optional.
    """
    # Verify run exists if run_id provided
    if req.run_id:
        run_info = run_manager.get_run(req.run_id)
        if not run_info:
            raise HTTPException(status_code=404, detail="Run not found")

    # Get or create session
    if req.session_id:
        session = run_manager.get_chat_session(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        session = run_manager.create_chat_session(req.run_id or "", req.user_id)

    # Store user message in history
    session.message_history.append({"role": "user", "content": req.message})

    # Run chat in background
    background_tasks.add_task(
        chat_runner.run,
        req.run_id or "",
        session.session_id,
        req.message,
        run_manager,
    )

    return ChatResponse(session_id=session.session_id, status="accepted")


async def _chat_event_generator(
    run_manager: RunManager, session_id: str
) -> AsyncGenerator[dict, None]:
    """Generate SSE events from the chat session's event queue."""
    session = run_manager.get_chat_session(session_id)
    if not session:
        return

    while True:
        event: ChatEvent | None = await session.event_queue.get()
        if event is None:
            break
        yield {
            "event": event.event,
            "data": event.model_dump_json(),
        }


@router.get("/stream/{session_id}")
async def stream_chat(
    session_id: str,
    run_manager: RunManager = Depends(get_run_manager),
) -> EventSourceResponse:
    """Stream chat response events via SSE."""
    if run_manager.get_chat_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    return EventSourceResponse(_chat_event_generator(run_manager, session_id))
