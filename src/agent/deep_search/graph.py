"""Deep Group Search Sub-Graph 빌드.

Leader → Research → Critic → (조건부) → Synthesize 파이프라인을 구성한다.
Critic 검증 실패 시 Research로 돌아가는 Review-Revise 루프를 포함한다.

그래프 흐름:
    START → decompose → research → review → (조건부 분기)
                                               ├─ 재검색 필요 → research
                                               └─ 통과 → synthesize → END
"""

from __future__ import annotations

import logging
from functools import partial

from langgraph.graph import StateGraph, END

from agent.config import AgentConfig, create_chat_model
from agent.deep_search.state import DeepSearchState
from agent.deep_search.nodes.leader import decompose, synthesize
from agent.deep_search.nodes.researcher import research
from agent.deep_search.nodes.critic import review

logger = logging.getLogger(__name__)

# 기본 최대 반복 횟수 (Research-Critic 루프)
DEFAULT_MAX_ITERATIONS = 3


def _route_after_review(state: DeepSearchState) -> str:
    """Critic 검증 후 조건부 분기.

    모든 관점이 통과했거나 최대 반복 횟수에 도달하면 synthesize로,
    그렇지 않으면 research로 돌아간다.

    Returns:
        다음 노드 이름 ("research" 또는 "synthesize").
    """
    critic_feedback = state.get("critic_feedback", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    # 안전장치: 최대 반복 횟수 초과
    if iteration_count >= max_iterations:
        logger.warning(
            "[deep_search:route] 최대 반복 횟수 도달 (%d >= %d), "
            "합성으로 진행",
            iteration_count,
            max_iterations,
        )
        return "synthesize"

    # 모든 관점이 통과했는지 확인
    all_passed = all(
        fb.get("passed", True) for fb in critic_feedback
    )

    if all_passed:
        logger.info("[deep_search:route] 모든 관점 통과, 합성 진행")
        return "synthesize"
    else:
        failed = [
            fb.get("perspective", "?")
            for fb in critic_feedback
            if not fb.get("passed", True)
        ]
        logger.info(
            "[deep_search:route] %d개 관점 미통과 (%s), 재검색",
            len(failed),
            failed,
        )
        return "research"


async def build_deep_search_graph(
    config: AgentConfig | None = None,
    tools: list | None = None,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
):
    """Deep Group Search Sub-Graph를 빌드.

    Args:
        config: 에이전트 설정. None이면 환경변수에서 로드.
        tools: 기존 Action Skills Tool 리스트. None이면 자동 로드.
        max_iterations: Research-Critic 루프 최대 반복 횟수.

    Returns:
        compiled sub-graph.
    """
    if config is None:
        config = AgentConfig.from_env()

    llm = create_chat_model(config)

    if tools is None:
        from agent.skills.actions.rag_search import get_action_tools
        from agent.skills.actions.web_search import get_web_search_tools

        tools = get_action_tools() + get_web_search_tools()

    logger.info(
        "[deep_search:graph] Sub-Graph 빌드 시작, "
        "tools=%d, max_iterations=%d",
        len(tools),
        max_iterations,
    )

    # 노드 함수 (의존성 주입)
    decompose_fn = partial(decompose, llm=llm)
    research_fn = partial(research, llm=llm, tools=tools)
    review_fn = partial(review, llm=llm)
    synthesize_fn = partial(synthesize, llm=llm)

    # 그래프 빌드
    graph = StateGraph(DeepSearchState)

    graph.add_node("decompose", decompose_fn)
    graph.add_node("research", research_fn)
    graph.add_node("review", review_fn)
    graph.add_node("synthesize", synthesize_fn)

    # 엣지
    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "research")
    graph.add_edge("research", "review")

    # review → 조건부 분기
    graph.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "research": "research",
            "synthesize": "synthesize",
        },
    )

    # synthesize → END
    graph.add_edge("synthesize", END)

    compiled = graph.compile()

    logger.info("[deep_search:graph] Sub-Graph 빌드 완료")
    return compiled


async def run_deep_search(
    original_query: str,
    reasoning_context: str = "",
    config: AgentConfig | None = None,
    tools: list | None = None,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> DeepSearchState:
    """Deep Group Search를 실행.

    Args:
        original_query: 사용자의 심층 분석 질문.
        reasoning_context: Main Agent에서 전달한 추론 맥락.
        config: 에이전트 설정.
        tools: 기존 Action Skills Tool 리스트.
        max_iterations: 최대 반복 횟수.

    Returns:
        최종 DeepSearchState.
    """
    graph = await build_deep_search_graph(
        config, tools, max_iterations=max_iterations
    )

    initial_state: DeepSearchState = {
        "original_query": original_query,
        "reasoning_context": reasoning_context,
        "perspectives": [],
        "search_results": [],
        "critic_feedback": [],
        "synthesis": "",
        "citations": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
    }

    return await graph.ainvoke(initial_state)
