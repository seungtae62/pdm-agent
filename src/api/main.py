"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize services."""
    app.state.run_manager = RunManager()

    runner_mode = os.getenv("AGENT_RUNNER_MODE", "mock")
    logger.info("AgentRunner mode: %s", runner_mode)
    if runner_mode == "langgraph":
        from agent.config import AgentConfig

        config = AgentConfig.from_env()
        app.state.agent_runner = LangGraphAgentRunner(config)
    else:
        config = None
        app.state.agent_runner = MockAgentRunner()

    chat_mode = os.getenv("CHAT_RUNNER_MODE", "llm")
    if chat_mode == "llm" and config is not None:
        from api.services.llm_chat_runner import LLMChatRunner

        app.state.chat_runner = LLMChatRunner(config)
    elif chat_mode == "llm":
        # langgraph 모드가 아니어도 LLM 챗은 독립 사용 가능
        from agent.config import AgentConfig
        from api.services.llm_chat_runner import LLMChatRunner

        app.state.chat_runner = LLMChatRunner(AgentConfig.from_env())
    else:
        app.state.chat_runner = MockChatRunner()

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
