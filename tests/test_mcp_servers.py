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

    def test_empty_search_results(self, mock_rag_server):
        """빈 검색 결과 처리."""
        server, mock_qdrant, _ = mock_rag_server
        mock_qdrant.search.return_value = []

        results = server.search_maintenance_history(query="존재하지 않는 쿼리")
        assert results == []

    def test_embed_uses_configured_model(self, mock_rag_server):
        """_embed가 config의 embedding_model을 사용하는지 확인."""
        server, _, mock_openai = mock_rag_server
        server.config.embedding_model = "text-embedding-3-large"
        server.search_maintenance_history(query="test")

        call_kwargs = mock_openai.embeddings.create.call_args
        assert call_kwargs.kwargs["model"] == "text-embedding-3-large"

    def test_config_from_env(self):
        """from_env()가 환경변수를 올바르게 읽는지 확인."""
        env = {
            "QDRANT_HOST": "remote-host",
            "QDRANT_PORT": "7777",
            "EMBEDDING_MODEL": "custom-model",
            "PDM_RAG_TOP_K": "10",
        }
        with patch.dict("os.environ", env):
            config = RAGServerConfig.from_env()

        assert config.qdrant_host == "remote-host"
        assert config.qdrant_port == 7777
        assert config.embedding_model == "custom-model"
        assert config.top_k == 10

    def test_custom_top_k_override(self, mock_rag_server):
        """쿼리별 top_k 오버라이드."""
        server, mock_qdrant, _ = mock_rag_server
        server.search_maintenance_history(query="test", top_k=7)

        call_kwargs = mock_qdrant.search.call_args
        assert call_kwargs.kwargs["limit"] == 7


# ---------------------------------------------------------------------------
# NotificationServer - 추가 테스트
# ---------------------------------------------------------------------------


class TestNotificationServerExtended:
    @pytest.mark.parametrize("risk_level", ["watch", "warning", "critical"])
    def test_risk_levels(self, risk_level):
        """다양한 위험도 레벨 처리."""
        server = NotificationServer()
        result = server.notify_maintenance_staff(
            message="테스트 알림",
            risk_level=risk_level,
            equipment_id="EQ-001",
        )
        assert result.success is True
        assert risk_level in result.message

    def test_timestamp_iso_format(self):
        """타임스탬프가 ISO 형식인지 확인."""
        server = NotificationServer()
        result = server.notify_maintenance_staff(
            message="test",
            risk_level="normal",
            equipment_id="EQ-001",
        )
        # ISO format 파싱 가능해야 함
        from datetime import datetime

        parsed = datetime.fromisoformat(result.timestamp)
        assert isinstance(parsed, datetime)
