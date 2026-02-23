"""에이전트 설정 및 LLM 팩토리.

환경변수에서 LLM 모델, 임베딩 모델, 인프라 설정을 로드한다.
LangChain ChatModel 팩토리로 다수 provider 모델을 지원한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "gpt-4o"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_TOOL_CALLS = 10
DEFAULT_RAG_TOP_K = 3


@dataclass
class AgentConfig:
    """PdM Agent 설정."""

    # LLM
    llm_model: str = DEFAULT_LLM_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    openai_api_key: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "pdm_agent"
    postgres_user: str = "pdm_agent"
    postgres_password: str = ""

    # Agent
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    rag_top_k: int = DEFAULT_RAG_TOP_K

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """환경변수에서 설정 로드."""
        return cls(
            llm_model=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_db=os.getenv("POSTGRES_DB", "pdm_agent"),
            postgres_user=os.getenv("POSTGRES_USER", "pdm_agent"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
            max_tool_calls=int(os.getenv("PDM_AGENT_MAX_TOOL_CALLS", str(DEFAULT_MAX_TOOL_CALLS))),
            rag_top_k=int(os.getenv("PDM_RAG_TOP_K", str(DEFAULT_RAG_TOP_K))),
        )

    @property
    def postgres_dsn(self) -> str:
        """PostgreSQL 연결 문자열."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# ---------------------------------------------------------------------------
# LLM 팩토리
# ---------------------------------------------------------------------------


def create_chat_model(config: AgentConfig | None = None):
    """LangChain ChatModel 생성.

    환경변수 LLM_MODEL 값에 따라 적절한 provider의 ChatModel을 반환한다.

    지원 형식:
    - "gpt-4o", "gpt-4o-mini" → ChatOpenAI
    - "anthropic/claude-3-opus" → ChatAnthropic (langchain-anthropic 필요)
    - 기본값: ChatOpenAI

    Args:
        config: 에이전트 설정. None이면 환경변수에서 로드.

    Returns:
        LangChain BaseChatModel 인스턴스.
    """
    if config is None:
        config = AgentConfig.from_env()

    model_id = config.llm_model

    if model_id.startswith("anthropic/") or model_id.startswith("claude"):
        model_name = model_id.replace("anthropic/", "")
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name)
        except ImportError:
            raise ImportError(
                "langchain-anthropic 패키지가 필요합니다: "
                "pip install langchain-anthropic"
            )

    # 기본: OpenAI
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model_id,
        api_key=config.openai_api_key,
    )
