"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from api.routes import chat, events, stream
from api.services.agent_runner import LangGraphAgentRunner, MockAgentRunner
from api.services.chat_runner import MockChatRunner
from api.services.run_manager import RunManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize services."""
    app.state.run_manager = RunManager()
    app.state.chat_runner = MockChatRunner()

    runner_mode = os.getenv("AGENT_RUNNER_MODE", "mock")
    if runner_mode == "langgraph":
        from agent.config import AgentConfig

        app.state.agent_runner = LangGraphAgentRunner(AgentConfig.from_env())
    else:
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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events.router)
app.include_router(stream.router)
app.include_router(chat.router)
