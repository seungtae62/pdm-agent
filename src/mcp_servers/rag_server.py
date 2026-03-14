"""RAG 검색 MCP Server.

Qdrant Vector DB에서 정비 이력, 설비 매뉴얼, 분석 이력을 검색하는
3개 Tool을 제공한다. LangGraph Agent가 MCP Client로 호출한다.

PoC에서는 MCP 프로토콜 없이 직접 함수 호출로 사용하며,
추후 MCP stdio/SSE 서버로 래핑 가능하다.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 3
VECTOR_SIZE = 1536


@dataclass
class RAGServerConfig:
    """RAG Server 설정."""

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = DEFAULT_TOP_K

    @classmethod
    def from_env(cls) -> "RAGServerConfig":
        """환경변수에서 설정 로드."""
        return cls(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            top_k=int(os.getenv("PDM_RAG_TOP_K", str(DEFAULT_TOP_K))),
        )


# ---------------------------------------------------------------------------
# RAG Server
# ---------------------------------------------------------------------------


class RAGServer:
    """Qdrant 기반 RAG 검색 서버.

    3개 Collection에서 의미적 검색을 수행한다:
    - maintenance_history: 과거 고장/정비 이력
    - equipment_manual: 설비 매뉴얼, FMEA, 절차서
    - analysis_history: 에이전트 과거 분석 판단 이력

    Args:
        config: RAG Server 설정. None이면 환경변수에서 로드.
        qdrant_client: Qdrant 클라이언트. None이면 config 기반 생성.
        openai_client: OpenAI 클라이언트. None이면 기본 생성.
    """

    def __init__(
        self,
        config: RAGServerConfig | None = None,
        qdrant_client: QdrantClient | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        self.config = config or RAGServerConfig.from_env()
        self.qdrant = qdrant_client or QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
        )
        self.openai = openai_client or OpenAI()

    def _embed(self, text: str) -> list[float]:
        """텍스트를 임베딩 벡터로 변환."""
        response = self.openai.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _search(
        self,
        collection_name: str,
        query: str,
        top_k: int | None = None,
        filters: list[tuple[str, str]] | None = None,
    ) -> list[dict]:
        """Qdrant Collection 검색.

        Args:
            collection_name: 검색 대상 Collection.
            query: 검색 쿼리 텍스트.
            top_k: 반환할 결과 수.
            filters: (field, value) 필터 조건 리스트.

        Returns:
            검색 결과 리스트. 각 항목은 score, text, metadata 포함.
        """
        top_k = top_k or self.config.top_k
        query_vector = self._embed(query)

        # 필터 구성
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters
                if value is not None
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        results = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            {
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {
                    k: v for k, v in (hit.payload or {}).items() if k != "text"
                },
            }
            for hit in results
        ]

    # -------------------------------------------------------------------
    # Tool 1: search_maintenance_history
    # -------------------------------------------------------------------

    def search_maintenance_history(
        self,
        query: str,
        equipment_id: str | None = None,
        bearing_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """과거 고장/정비 이력을 의미적으로 검색.

        유사 결함 사례의 진행 경과, 근본 원인, 고장까지 소요 시간 등을 참조.

        Args:
            query: 검색 쿼리 (예: "내륜 결함 급속 열화 사례").
            equipment_id: 설비 ID 필터 (선택).
            bearing_id: 베어링 ID 필터 (선택).
            top_k: 반환할 결과 수 (기본: 3).

        Returns:
            유사 정비 이력 문서 리스트 (유사도 순).
        """
        filters = []
        if equipment_id:
            filters.append(("equipment_id", equipment_id))
        if bearing_id:
            filters.append(("bearing_id", bearing_id))

        logger.info(
            f"[RAG] search_maintenance_history: query='{query[:50]}...', "
            f"equipment_id={equipment_id}, bearing_id={bearing_id}"
        )

        return self._search(
            collection_name="maintenance_history",
            query=query,
            top_k=top_k,
            filters=filters or None,
        )

    # -------------------------------------------------------------------
    # Tool 2: search_equipment_manual
    # -------------------------------------------------------------------

    def search_equipment_manual(
        self,
        query: str,
        doc_type: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """설비 매뉴얼, FMEA 문서, 정비 절차서를 검색.

        설비 사양, 결함 메커니즘, 급속 열화 조건, 교체 절차 등을 참조.

        Args:
            query: 검색 쿼리 (예: "외륜 결함 급속 열화 조건").
            doc_type: 문서 유형 필터 (선택). 예: "spec", "fault_guide",
                      "procedure", "fmea".
            top_k: 반환할 결과 수 (기본: 3).

        Returns:
            관련 매뉴얼 문서 리스트 (유사도 순).
        """
        filters = []
        if doc_type:
            filters.append(("doc_type", doc_type))

        logger.info(
            f"[RAG] search_equipment_manual: query='{query[:50]}...', "
            f"doc_type={doc_type}"
        )

        return self._search(
            collection_name="equipment_manual",
            query=query,
            top_k=top_k,
            filters=filters or None,
        )

    # -------------------------------------------------------------------
    # Tool 3: search_analysis_history
    # -------------------------------------------------------------------

    def search_analysis_history(
        self,
        query: str,
        equipment_id: str | None = None,
        bearing_id: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """에이전트의 과거 분석 판단 이력을 의미적으로 검색.

        유사한 패턴의 과거 판단과 결과를 참조하여 일관성 유지.

        Args:
            query: 검색 쿼리 (예: "BPFI 상승 내륜 결함 2단계").
            equipment_id: 설비 ID 필터 (선택).
            bearing_id: 베어링 ID 필터 (선택).
            top_k: 반환할 결과 수 (기본: 3).

        Returns:
            과거 분석 결과 리스트 (유사도 순).
        """
        filters = []
        if equipment_id:
            filters.append(("equipment_id", equipment_id))
        if bearing_id:
            filters.append(("bearing_id", bearing_id))

        logger.info(
            f"[RAG] search_analysis_history: query='{query[:50]}...', "
            f"equipment_id={equipment_id}, bearing_id={bearing_id}"
        )

        return self._search(
            collection_name="analysis_history",
            query=query,
            top_k=top_k,
            filters=filters or None,
        )
