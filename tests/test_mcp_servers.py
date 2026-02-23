"""MCP Server 테스트."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mcp_servers.notification_server import NotificationServer, NotificationResult
from mcp_servers.rag_server import RAGServer, RAGServerConfig


# ---------------------------------------------------------------------------
# NotificationServer
# ---------------------------------------------------------------------------


class TestNotificationServer:
    def test_notify_returns_success(self):
        server = NotificationServer()
        result = server.notify_maintenance_staff(
            message="베어링 결함 감지",
            risk_level="warning",
            equipment_id="IMS-TESTRIG-01",
        )
        assert result.success is True
        assert "IMS-TESTRIG-01" in result.message

    def test_notify_result_type(self):
        server = NotificationServer()
        result = server.notify_maintenance_staff(
            message="test", risk_level="critical", equipment_id="EQ-001"
        )
        assert isinstance(result, NotificationResult)
        assert result.timestamp  # 비어있지 않음


# ---------------------------------------------------------------------------
# RAGServer (mock 기반)
# ---------------------------------------------------------------------------


class TestRAGServer:
    @pytest.fixture
    def mock_rag_server(self):
        """Qdrant + OpenAI를 mock한 RAGServer."""
        config = RAGServerConfig(
            qdrant_host="localhost",
            qdrant_port=6333,
            embedding_model="text-embedding-3-small",
            top_k=3,
        )

        mock_qdrant = MagicMock()
        mock_openai = MagicMock()

        # 임베딩 mock
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai.embeddings.create.return_value = mock_embedding_response

        # Qdrant 검색 mock
        mock_hit = MagicMock()
        mock_hit.score = 0.95
        mock_hit.payload = {
            "text": "내륜 결함 사례: 35일 운전 후 발생",
            "equipment_id": "IMS-TESTRIG-01",
            "fault_type": "inner_race",
        }
        mock_qdrant.search.return_value = [mock_hit]

        server = RAGServer(
            config=config,
            qdrant_client=mock_qdrant,
            openai_client=mock_openai,
        )
        return server, mock_qdrant, mock_openai

    def test_search_maintenance_history(self, mock_rag_server):
        server, mock_qdrant, _ = mock_rag_server
        results = server.search_maintenance_history(
            query="내륜 결함 사례",
            equipment_id="IMS-TESTRIG-01",
        )

        assert len(results) == 1
        assert results[0]["score"] == 0.95
        assert "내륜 결함" in results[0]["text"]
        assert results[0]["metadata"]["fault_type"] == "inner_race"

        # Qdrant search 호출 확인
        mock_qdrant.search.assert_called_once()
        call_kwargs = mock_qdrant.search.call_args
        assert call_kwargs.kwargs["collection_name"] == "maintenance_history"

    def test_search_equipment_manual(self, mock_rag_server):
        server, mock_qdrant, _ = mock_rag_server
        results = server.search_equipment_manual(
            query="외륜 결함 메커니즘",
            doc_type="fault_guide",
        )

        assert len(results) == 1
        mock_qdrant.search.assert_called_once()
        call_kwargs = mock_qdrant.search.call_args
        assert call_kwargs.kwargs["collection_name"] == "equipment_manual"

    def test_search_analysis_history(self, mock_rag_server):
        server, mock_qdrant, _ = mock_rag_server
        results = server.search_analysis_history(
            query="BPFI 상승 패턴",
        )

        assert len(results) == 1
        mock_qdrant.search.assert_called_once()
        call_kwargs = mock_qdrant.search.call_args
        assert call_kwargs.kwargs["collection_name"] == "analysis_history"

    def test_search_with_no_filter(self, mock_rag_server):
        server, mock_qdrant, _ = mock_rag_server
        server.search_maintenance_history(query="test")

        call_kwargs = mock_qdrant.search.call_args
        assert call_kwargs.kwargs.get("query_filter") is None

    def test_search_with_filter(self, mock_rag_server):
        server, mock_qdrant, _ = mock_rag_server
        server.search_maintenance_history(
            query="test",
            equipment_id="EQ-001",
        )

        call_kwargs = mock_qdrant.search.call_args
        assert call_kwargs.kwargs.get("query_filter") is not None

    def test_embedding_called(self, mock_rag_server):
        server, _, mock_openai = mock_rag_server
        server.search_maintenance_history(query="test query")

        mock_openai.embeddings.create.assert_called_once()
        call_kwargs = mock_openai.embeddings.create.call_args
        assert call_kwargs.kwargs["input"] == "test query"
