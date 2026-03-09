"""PdM Agent StateGraph 빌드.

LangGraph StateGraph 기반으로 7개 노드를 등록하고,
조건부 엣지로 ReAct 추론 루프를 구성한다.
MCP 서버에서 동적으로 Tool을 검색하여 사용한다.

그래프 흐름:
    START → load_memory → reasoning → (조건부 분기)
                                        ├─ call_tool → tool_executor → reasoning
                                        ├─ continue_reasoning → reasoning
                                        └─ generate_report → parse_diagnosis
                                            → generate_report → generate_work_order
                                            → save_memory → END
"""

from __future__ import annotations

import logging
import sys
from functools import partial

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END

from agent.config import AgentConfig, create_chat_model
from agent.memory.store import MemoryStore
from agent.state import PdMAgentState
from agent.nodes.load_memory import load_memory
from agent.nodes.reasoning import reasoning
from agent.nodes.tool_executor import create_tool_executor
from agent.nodes.parse_diagnosis import parse_diagnosis
from agent.nodes.generate_report import generate_report
from agent.nodes.generate_work_order import generate_work_order
from agent.nodes.save_memory import save_memory

logger = logging.getLogger(__name__)


def _route_after_reasoning(state: PdMAgentState) -> str:
    """reasoning 노드 후 조건부 분기.

    Returns:
        다음 노드 이름.
    """
    next_action = state.get("next_action", "generate_report")
    tool_count = state.get("tool_calls_count", 0)
    max_calls = 10  # 안전장치

    # 안전장치: Tool 호출 횟수 초과 → 강제 종료
    if tool_count > max_calls:
        logger.warning(
            f"[route] Tool 호출 횟수 초과 ({tool_count} > {max_calls}), "
            f"강제 parse_diagnosis"
        )
        return "parse_diagnosis"

    if next_action == "call_tool":
        return "tool_executor"
    elif next_action == "continue_reasoning":
        return "reasoning"
    else:
        return "parse_diagnosis"


def _route_after_report(state: PdMAgentState) -> str:
    """generate_report 후 조건부 분기.

    Warning/Critical → generate_work_order
    Normal/Watch → save_memory (작업지시서 건너뜀)

    LLM 응답의 risk_level 대소문자 정규화 + Edge health_state fallback으로
    비결정적 LLM 출력에 대한 안정성을 보장한다.
    """
    diagnosis = state.get("diagnosis_result", {})
    risk_level = diagnosis.get("risk_level", "normal").lower().strip()

    if risk_level in ("warning", "critical"):
        return "generate_work_order"

    # fallback: Edge의 원본 health_state 확인
    payload = state.get("event_payload", {})
    if isinstance(payload, dict):
        adr = payload.get("anomaly_detection_result", {})
        health = adr.get("health_state", "normal").lower().strip()
        if health in ("warning", "critical"):
            logger.info(
                f"[route] risk_level='{risk_level}' 이지만 "
                f"health_state='{health}' fallback으로 work_order 생성"
            )
            return "generate_work_order"

    return "save_memory"


def _build_mcp_server_config(config: AgentConfig) -> dict:
    """MCP 서버 연결 설정 dict 생성."""
    return {
        "rag-server": {
            "command": sys.executable,
            "args": [config.rag_mcp_server_path],
            "transport": "stdio",
        },
        "notification-server": {
            "command": sys.executable,
            "args": [config.notification_mcp_server_path],
            "transport": "stdio",
        },
    }


async def build_graph(
    config: AgentConfig | None = None,
    *,
    memory_store: MemoryStore | None = None,
) -> tuple:
    """PdM Agent StateGraph를 빌드.

    MCP 서버에서 동적으로 Tool을 검색하여 그래프에 바인딩한다.

    Args:
        config: 에이전트 설정. None이면 환경변수에서 로드.
        memory_store: PostgreSQL Memory. None이면 Memory 없이 실행.

    Returns:
        (compiled_graph, mcp_client) 튜플.
    """
    if config is None:
        config = AgentConfig.from_env()

    llm = create_chat_model(config)

    # MCP 서버 연결 및 Tool 검색
    mcp_client = None
    tools = []
    try:
        mcp_client = MultiServerMCPClient(_build_mcp_server_config(config))
        tools = await mcp_client.get_tools()
        logger.info(f"[build_graph] MCP Tool {len(tools)}개 검색 완료: {[t.name for t in tools]}")
    except Exception as e:
        logger.warning(f"[build_graph] MCP 연결 실패, 빈 도구 목록으로 진행: {e}")
        mcp_client = None
        tools = []

    # 노드 함수 (의존성 주입)
    load_memory_fn = partial(load_memory, store=memory_store)
    reasoning_fn = partial(reasoning, llm=llm, tools=tools)
    tool_executor_fn = create_tool_executor(tools)
    generate_report_fn = partial(generate_report, llm=llm)
    generate_work_order_fn = partial(generate_work_order, llm=llm)
    save_memory_fn = partial(save_memory, store=memory_store)

    # 그래프 빌드
    graph = StateGraph(PdMAgentState)

    graph.add_node("load_memory", load_memory_fn)
    graph.add_node("reasoning", reasoning_fn)
    graph.add_node("tool_executor", tool_executor_fn)
    graph.add_node("parse_diagnosis", parse_diagnosis)
    graph.add_node("generate_report", generate_report_fn)
    graph.add_node("generate_work_order", generate_work_order_fn)
    graph.add_node("save_memory", save_memory_fn)

    # 엣지
    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "reasoning")

    # reasoning → 조건부 분기
    graph.add_conditional_edges(
        "reasoning",
        _route_after_reasoning,
        {
            "tool_executor": "tool_executor",
            "reasoning": "reasoning",
            "parse_diagnosis": "parse_diagnosis",
        },
    )

    # tool_executor → reasoning (루프)
    graph.add_edge("tool_executor", "reasoning")

    # parse_diagnosis → generate_report
    graph.add_edge("parse_diagnosis", "generate_report")

    # generate_report → 조건부 (Warning/Critical → work_order, else → save)
    graph.add_conditional_edges(
        "generate_report",
        _route_after_report,
        {
            "generate_work_order": "generate_work_order",
            "save_memory": "save_memory",
        },
    )

    # generate_work_order → save_memory
    graph.add_edge("generate_work_order", "save_memory")

    # save_memory → END
    graph.add_edge("save_memory", END)

    return graph.compile(), mcp_client


async def run_agent(
    event_payload: dict,
    config: AgentConfig | None = None,
    **kwargs,
) -> PdMAgentState:
    """PdM Agent 실행.

    Args:
        event_payload: Edge 이벤트 페이로드 dict.
        config: 에이전트 설정.
        **kwargs: build_graph에 전달할 추가 인자.

    Returns:
        최종 State.
    """
    graph, mcp_client = await build_graph(config, **kwargs)

    initial_state: PdMAgentState = {
        "event_payload": event_payload,
        "memory_context": {},
        "messages": [],
        "diagnosis_result": {},
        "tool_calls_count": 0,
        "deep_research_activated": False,
        "report": "",
        "work_order": "",
        "next_action": "",
    }

    result = await graph.ainvoke(initial_state)
    return result
