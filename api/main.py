"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import events, stream
from api.services.agent_runner import MockAgentRunner
from api.services.run_manager import RunManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize services."""
    app.state.run_manager = RunManager()
    app.state.agent_runner = MockAgentRunner()
    yield


app = FastAPI(
    title="PDM Agent API",
    description="Predictive Maintenance Agent API for Edge event processing",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events.router)
app.include_router(stream.router)
