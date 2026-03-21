"""Deep Group Search State 스키마.

Leader-Research-Critic-Synthesis 파이프라인의 상태를 정의한다.
Main Agent의 PdMAgentState와 독립적으로 관리된다.
"""

from __future__ import annotations

from typing import TypedDict


class SearchResult(TypedDict, total=False):
    """개별 관점(perspective)의 검색 결과."""

    perspective: str
    sub_query: str
    agent_role: str
    results: str
    confidence: float  # 0.0 ~ 1.0
    sources: list[dict]  # [{source_type, doc_id, title, url, relevance_score}]
    needs_revision: bool


class CriticFeedback(TypedDict, total=False):
    """Critic Agent의 검증 결과."""

    perspective: str
    passed: bool
    feedback: str
    unsourced_claims: list[str]


class Citation(TypedDict, total=False):
    """출처 인용 정보."""

    source_type: str  # internal_rag / external_web
    doc_id: str
    title: str
    url: str
    relevance_score: float


class DeepSearchState(TypedDict, total=False):
    """Deep Group Search Sub-Graph 상태.

    Attributes:
        original_query: 사용자의 원본 심층 분석 질문.
        reasoning_context: Main Agent Memory에서 전달받은 추론 맥락.
        perspectives: STORM 스타일 다관점 분해 결과.
        search_results: 각 관점별 Research Agent 검색 결과.
        critic_feedback: Critic Agent 검증 결과.
        synthesis: 교차 참조 합성 최종 답변.
        citations: 전체 출처 목록.
        iteration_count: 현재 반복 횟수 (Research-Critic 루프).
        max_iterations: 최대 반복 횟수 (안전장치).
    """

    original_query: str
    reasoning_context: str
    perspectives: list[dict]
    search_results: list[SearchResult]
    critic_feedback: list[CriticFeedback]
    synthesis: str
    citations: list[Citation]
    iteration_count: int
    max_iterations: int
