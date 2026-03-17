---
name: 문서-구현 불일치 팔로우업 항목
description: 문서 vs 코드 비교에서 발견된 미구현/불일치 사항. 향후 팔로우업 필요
type: project
---

2026-03-15 문서-코드 비교에서 도출된 팔로우업 항목:

### 1. ~~Web Search MCP 서버 미구현~~ → 해결됨 (2026-03-15)
- `src/agent/skills/actions/web_search.py`에 Tavily 기반 `search_web` 도구 구현 완료
- MCP 서버가 아닌 LangChain Tool로 직접 구현됨

### 2. Sparse Hybrid Search 미구현
- design.md에 "Dense + Sparse(BM25 + kiwipiepy) → RRF + Reranker" 명세
- 실제로는 Dense 임베딩(OpenAI text-embedding-3-small)만 구현
- **How to apply:** BM25/kiwipiepy/Reranker 구현 또는 design.md를 현행(Dense only)에 맞게 수정

### 3. Prompt Optimization 미구현
- design.md에 대화 이력 압축, 슬라이딩 윈도우(최근 2~3턴 원문 + 이전 요약) 명세
- 대화형 상호작용(SC-005) 관련 기능 전체 미구현
- **How to apply:** 대화형 API 구현 시 함께 구현

### 4. LangGraph PostgreSQL Checkpointer 미연동
- CLAUDE.md에 "Memory: PostgreSQL (langgraph-checkpoint-postgres)" 기술
- `graph.compile()`에 checkpointer 미전달 → 실행 중단 시 복구 불가
- **How to apply:** `langgraph-checkpoint-postgres` 패키지로 checkpointer 연동 또는 문서 수정

### 5. design.md 명칭 불일치 (경미)
- design.md: "response-template" Skill 단일 언급
- 실제 코드: `response_normal.md` + `response_alert.md` 2개로 분리
- **How to apply:** design.md의 Agent Skills 명세 테이블에서 response-template → response-normal / response-alert로 수정
