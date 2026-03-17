"""Web Search Action Skill 테스트."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.skills.actions.web_search import (
    get_web_search_tools,
    search_web,
    _get_client,
)


def test_get_web_search_tools_returns_search_web():
    """get_web_search_tools가 search_web tool을 반환한다."""
    tools = get_web_search_tools()
    assert len(tools) == 1
    assert tools[0].name == "search_web"


def test_search_web_tool_has_description():
    """search_web tool에 docstring 기반 description이 설정되어 있다."""
    tools = get_web_search_tools()
    desc = tools[0].description
    assert "RAG" in desc or "웹" in desc


@patch("agent.skills.actions.web_search._tavily_client", None)
@patch("agent.skills.actions.web_search.os.getenv", return_value=None)
def test_get_client_raises_without_api_key(mock_getenv):
    """TAVILY_API_KEY가 없으면 RuntimeError가 발생한다."""
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        _get_client()


@patch("agent.skills.actions.web_search._tavily_client", None)
@patch("agent.skills.actions.web_search.os.getenv", return_value="tvly-test-key")
def test_get_client_initializes_with_api_key(mock_getenv):
    """TAVILY_API_KEY가 있으면 TavilyClient가 초기화된다."""
    with patch("agent.skills.actions.web_search.TavilyClient", create=True) as MockClass:
        # tavily 모듈의 lazy import를 모킹
        import agent.skills.actions.web_search as ws
        with patch.dict("sys.modules", {"tavily": MagicMock(TavilyClient=MockClass)}):
            client = _get_client()
            MockClass.assert_called_once_with(api_key="tvly-test-key")


@patch("agent.skills.actions.web_search._get_client")
def test_search_web_formats_results(mock_get_client):
    """검색 결과가 포맷팅되어 반환된다."""
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Bearing Fault Diagnosis",
                "url": "https://example.com/bearing",
                "content": "BPFO frequency indicates outer race fault.",
            },
            {
                "title": "Vibration Analysis Guide",
                "url": "https://example.com/vibration",
                "content": "Envelope analysis is used for bearing diagnostics.",
            },
        ]
    }
    mock_get_client.return_value = mock_client

    result = search_web.invoke({"query": "bearing fault BPFO"})

    mock_client.search.assert_called_once_with("bearing fault BPFO", max_results=5)
    assert "[1] Bearing Fault Diagnosis" in result
    assert "[2] Vibration Analysis Guide" in result
    assert "https://example.com/bearing" in result
    assert "BPFO frequency" in result


@patch("agent.skills.actions.web_search._get_client")
def test_search_web_no_results(mock_get_client):
    """검색 결과가 없으면 안내 메시지를 반환한다."""
    mock_client = MagicMock()
    mock_client.search.return_value = {"results": []}
    mock_get_client.return_value = mock_client

    result = search_web.invoke({"query": "nonexistent query"})
    assert "검색 결과가 없습니다" in result
