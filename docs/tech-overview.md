# PdM Agent 기술 개요

## 핵심 기술 (현재 구현)

이 프로젝트의 에이전트 기술은 2가지로 압축된다.

### 1. ReAct 패턴 (Reasoning + Acting)

- LLM이 추론(Thought)과 행동(Tool 호출)을 반복하는 Agentic Loop
- LangGraph StateGraph로 `reasoning ↔ tool_executor` 루프 구현
- 학습 단계에서 Raw API → LangChain → LangGraph로 단계적으로 익힌 것과 동일한 구조

### 2. MCP (Model Context Protocol)

- Tool을 별도 서버(프로세스)로 분리하는 프로토콜
- 현재는 직접 함수 호출로 구현 (MCP 프로토콜 미적용)
- 추후 stdio/SSE 기반 MCP 서버로 전환 예정

### 나머지는 기존 기술의 조합

| 구현 | 실체 |
|------|------|
| StateGraph 7개 노드 | LangGraph 노드/엣지 연결 |
| Memory | PostgreSQL CRUD |
| RAG | Qdrant + OpenAI 임베딩 |
| parse_diagnosis | JSON 파싱 |
| generate_report / work_order | LLM 프롬프트 호출 |

에이전트만의 딥한 신기술은 없다. 기존 기술들을 도메인에 맞게 배치한 것.

---

## 진짜 차별화 포인트: 도메인 지식

이 프로젝트에서 기술보다 중요한 것은 **시스템 프롬프트에 담길 도메인 지식**이다.

- 베어링 결함 유형 식별 (내륜, 외륜, 전동체, 케이지)
- P-F 곡선 기반 결함 진행 단계 판정 (Stage 0~4)
- 진동 주파수 해석 (BPFI, BPFO, BSF, FTF)
- 열화 속도 판단 및 RUL 평가
- 위험도 분류 체계 (Normal → Watch → Warning → Critical)
- 정비 권고 생성 로직

CLAUDE.md 아키텍처 원칙이 이를 명시한다:
- **수치 계산 금지** — Edge가 산출한 값을 읽고 해석만 한다
- **도메인 지식 기반 해석** — 시스템 프롬프트의 전문 지식으로 판단한다
- **에이전트 추론 능력 극대화** — Tool 호출을 최소화하고 추론에 집중한다

RAG에 적재할 데이터의 품질(정비 이력, 매뉴얼, FMEA)도 에이전트 성능을 좌우한다.

---

## 기술 심화 후보 (검토 필요)

현재 뼈대에 추가할 수 있는 딥한 기술적 요소들.

### A. 에이전트 추론 고도화

- **Structured Output** — LLM 응답을 JSON Schema로 강제하여 parse_diagnosis의 정규식 파싱 제거
- **Self-Reflection** — 에이전트가 자신의 판단을 검증하는 2차 추론 루프
- **Confidence Calibration** — 불확실성을 정량화하여 판단 신뢰도를 조절

### B. Memory 고도화

- **Episodic vs Semantic Memory** — 이벤트별 기억과 패턴화된 지식을 분리 관리
- **Memory Retrieval 전략** — 단순 최근 N건이 아닌 유사도 기반 관련 이력 검색
- **Forgetting Mechanism** — 오래된/무관한 기억의 가중치 감소

### C. RAG 고도화

- **Hybrid Search** — 벡터 검색 + 키워드 검색 결합
- **Reranker** — 검색 결과를 Cross-Encoder로 재순위화
- **Query Decomposition** — 복합 쿼리를 분해하여 다중 검색 후 종합

### D. 관측 가능성 (Observability)

- **LangSmith / LangFuse** — 추론 과정 트레이싱, 토큰 사용량 모니터링
- **판단 품질 평가 체계** — 에이전트 판단 vs 실제 결과 비교 피드백 루프

### E. 실시간 처리

- **Streaming 응답** — Streamlit UI에서 토큰 단위 실시간 출력
- **비동기 이벤트 처리** — 다수 Edge 이벤트 동시 처리
