"""Deep Group Search 단위 테스트.

Sub-Graph 스키마, 프롬프트 형식, 그래프 빌드를 검증한다.
"""

from __future__ import annotations

import pytest

from agent.deep_search.state import (
    DeepSearchState,
    SearchResult,
    CriticFeedback,
    Citation,
)
from agent.deep_search.prompts import (
    DECOMPOSITION_PROMPT,
    RESEARCH_PROMPT,
    CRITIC_PROMPT,
    SYNTHESIS_PROMPT,
)


class TestDeepSearchState:
    """DeepSearchState 스키마 검증."""

    def test_state_has_required_fields(self):
        """모든 필수 필드가 정의되어 있는지 확인."""
        annotations = DeepSearchState.__annotations__
        required_fields = [
            "original_query",
            "reasoning_context",
            "perspectives",
            "search_results",
            "critic_feedback",
            "synthesis",
            "citations",
            "iteration_count",
            "max_iterations",
        ]
        for field in required_fields:
            assert field in annotations, f"Missing field: {field}"

    def test_search_result_schema(self):
        """SearchResult TypedDict 필드 확인."""
        annotations = SearchResult.__annotations__
        assert "perspective" in annotations
        assert "confidence" in annotations
        assert "sources" in annotations
        assert "needs_revision" in annotations

    def test_critic_feedback_schema(self):
        """CriticFeedback TypedDict 필드 확인."""
        annotations = CriticFeedback.__annotations__
        assert "perspective" in annotations
        assert "passed" in annotations
        assert "feedback" in annotations
        assert "unsourced_claims" in annotations

    def test_citation_schema(self):
        """Citation TypedDict 필드 확인."""
        annotations = Citation.__annotations__
        assert "source_type" in annotations
        assert "doc_id" in annotations
        assert "title" in annotations
        assert "url" in annotations
        assert "relevance_score" in annotations


class TestPromptFormats:
    """프롬프트 템플릿 검증."""

    def test_decomposition_prompt_has_placeholders(self):
        """DECOMPOSITION_PROMPT에 필수 placeholder가 있는지 확인."""
        assert "{original_query}" in DECOMPOSITION_PROMPT
        assert "{reasoning_context}" in DECOMPOSITION_PROMPT

    def test_research_prompt_has_placeholders(self):
        """RESEARCH_PROMPT에 필수 placeholder가 있는지 확인."""
        assert "{perspective}" in RESEARCH_PROMPT
        assert "{sub_query}" in RESEARCH_PROMPT
        assert "{agent_role}" in RESEARCH_PROMPT
        assert "{original_query}" in RESEARCH_PROMPT

    def test_critic_prompt_has_placeholders(self):
        """CRITIC_PROMPT에 필수 placeholder가 있는지 확인."""
        assert "{original_query}" in CRITIC_PROMPT
        assert "{search_results_text}" in CRITIC_PROMPT

    def test_synthesis_prompt_has_placeholders(self):
        """SYNTHESIS_PROMPT에 필수 placeholder가 있는지 확인."""
        assert "{original_query}" in SYNTHESIS_PROMPT
        assert "{all_results_text}" in SYNTHESIS_PROMPT
        assert "{critic_feedback_text}" in SYNTHESIS_PROMPT

    def test_decomposition_prompt_renders(self):
        """DECOMPOSITION_PROMPT가 정상적으로 렌더링되는지 확인."""
        rendered = DECOMPOSITION_PROMPT.format(
            original_query="이 베어링이 왜 빨리 열화됐어?",
            reasoning_context="최근 5일간 Watch→Warning 전환",
        )
        assert "이 베어링이 왜 빨리 열화됐어?" in rendered
        assert "최근 5일간" in rendered


class TestResearcherHelpers:
    """Research Agent 헬퍼 함수 테스트."""

    def test_calculate_confidence_empty(self):
        """빈 결과의 confidence는 0.0."""
        from agent.deep_search.nodes.researcher import _calculate_confidence

        assert _calculate_confidence("") == 0.0
        assert _calculate_confidence("검색 결과가 없습니다.") == 0.0

    def test_calculate_confidence_json_results(self):
        """JSON 리스트 결과의 confidence 계산."""
        from agent.deep_search.nodes.researcher import _calculate_confidence
        import json

        # 1건
        assert _calculate_confidence(json.dumps([{"id": "1"}])) == 0.4
        # 3건
        assert (
            _calculate_confidence(json.dumps([{"id": "1"}, {"id": "2"}, {"id": "3"}]))
            == 0.7
        )
        # 5건
        assert (
            _calculate_confidence(json.dumps([{"id": str(i)} for i in range(5)])) == 0.9
        )

    def test_extract_sources_rag(self):
        """RAG 결과에서 출처 추출."""
        from agent.deep_search.nodes.researcher import _extract_sources
        import json

        raw = json.dumps(
            [
                {"id": "doc-001", "title": "정비 이력 #1", "score": 0.85},
                {"id": "doc-002", "title": "정비 이력 #2", "score": 0.72},
            ]
        )
        sources = _extract_sources(raw, "internal_rag")
        assert len(sources) == 2
        assert sources[0]["source_type"] == "internal_rag"
        assert sources[0]["doc_id"] == "doc-001"

    def test_extract_sources_web(self):
        """웹 검색 결과에서 출처 추출."""
        from agent.deep_search.nodes.researcher import _extract_sources

        raw = (
            "[1] Bearing Fault Detection Methods\n"
            "    URL: https://example.com/article1\n"
            "    Content about bearing faults...\n\n"
            "[2] Vibration Analysis Guide\n"
            "    URL: https://example.com/article2\n"
            "    Content about vibration..."
        )
        sources = _extract_sources(raw, "external_web")
        assert len(sources) == 2
        assert sources[0]["url"] == "https://example.com/article1"
        assert sources[0]["source_type"] == "external_web"


class TestLeaderHelpers:
    """Leader Agent 헬퍼 함수 테스트."""

    def test_extract_json_from_codeblock(self):
        """코드블록 내 JSON 추출."""
        from agent.deep_search.nodes.leader import _extract_json

        text = '```json\n[{"perspective": "정비 엔지니어"}]\n```'
        result = _extract_json(text)
        assert result == [{"perspective": "정비 엔지니어"}]

    def test_extract_json_raw(self):
        """raw JSON 추출."""
        from agent.deep_search.nodes.leader import _extract_json

        text = '[{"perspective": "설비 전문가"}]'
        result = _extract_json(text)
        assert result == [{"perspective": "설비 전문가"}]

    def test_extract_json_invalid(self):
        """잘못된 JSON은 None 반환."""
        from agent.deep_search.nodes.leader import _extract_json

        assert _extract_json("이것은 JSON이 아닙니다") is None


@pytest.mark.asyncio
class TestGraphBuild:
    """Sub-Graph 빌드 테스트."""

    async def test_deep_search_graph_compiles(self):
        """Deep Search Sub-Graph가 에러 없이 컴파일되는지 확인."""
        from unittest.mock import MagicMock

        from agent.deep_search.graph import build_deep_search_graph
        from agent.config import AgentConfig

        # Mock config
        config = AgentConfig(
            llm_model="gpt-4o",
            openai_api_key="test-key",
        )

        # Mock tools
        mock_tool = MagicMock()
        mock_tool.name = "search_maintenance_history"

        graph = await build_deep_search_graph(
            config=config,
            tools=[mock_tool],
        )

        assert graph is not None

    async def test_deep_search_graph_has_expected_nodes(self):
        """Sub-Graph에 필요한 노드가 모두 있는지 확인."""
        from unittest.mock import MagicMock

        from agent.deep_search.graph import build_deep_search_graph
        from agent.config import AgentConfig

        config = AgentConfig(
            llm_model="gpt-4o",
            openai_api_key="test-key",
        )

        mock_tool = MagicMock()
        mock_tool.name = "search_maintenance_history"

        graph = await build_deep_search_graph(
            config=config,
            tools=[mock_tool],
        )

        # CompiledGraph의 노드 확인
        node_names = set(graph.nodes.keys())
        expected_nodes = {"decompose", "research", "review", "synthesize"}
        # __start__ / __end__ 노드는 LangGraph 내부 노드
        assert expected_nodes.issubset(
            node_names
        ), f"Missing nodes: {expected_nodes - node_names}"
