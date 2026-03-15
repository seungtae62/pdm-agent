"""Pydantic models for chat requests and responses."""

from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat message submission."""

    run_id: str | None = None
    session_id: str | None = None
    message: str
    user_id: str = "default"


class ChatResponse(BaseModel):
    """Chat submission response."""

    session_id: str
    status: str
