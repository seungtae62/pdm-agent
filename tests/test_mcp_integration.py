"""MCP 서버 통합 테스트 — FastMCP 서버 도구 검색 확인."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

PROJECT_ROOT = Path(__file__).parent.parent

EXPECTED_TOOLS = {
    "search_maintenance_history",
    "search_equipment_manual",
    "search_analysis_history",
    "notify_maintenance_staff",
}


def _make_server_config() -> dict:
    return {
        "rag-server": {
            "command": sys.executable,
            "args": [str(PROJECT_ROOT / "mcp_servers" / "rag_mcp.py")],
            "transport": "stdio",
        },
        "notification-server": {
            "command": sys.executable,
            "args": [str(PROJECT_ROOT / "mcp_servers" / "notification_mcp.py")],
            "transport": "stdio",
        },
    }


@pytest.mark.asyncio
async def test_mcp_tool_discovery():
    """MCP 서버에서 4개 Tool이 검색되는지 확인."""
    client = MultiServerMCPClient(_make_server_config())
    tools = await client.get_tools()

    tool_names = {t.name for t in tools}
    assert tool_names == EXPECTED_TOOLS, (
        f"검색된 Tool: {tool_names}, 기대: {EXPECTED_TOOLS}"
    )


@pytest.mark.asyncio
async def test_mcp_tools_have_descriptions():
    """MCP Tool들이 description을 가지고 있는지 확인."""
    client = MultiServerMCPClient(_make_server_config())
    tools = await client.get_tools()

    for tool in tools:
        assert tool.description, f"Tool '{tool.name}'에 description이 없음"
