"""FastAPI dependency injection."""

from __future__ import annotations

from fastapi import Request

from api.services.agent_runner import AgentRunner
from api.services.run_manager import RunManager


def get_run_manager(request: Request) -> RunManager:
    """Get RunManager from app state."""
    return request.app.state.run_manager


def get_agent_runner(request: Request) -> AgentRunner:
    """Get AgentRunner from app state."""
    return request.app.state.agent_runner
