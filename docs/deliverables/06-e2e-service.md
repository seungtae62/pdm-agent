## 1. 최종 아키텍처 요약

- **아키텍처:** LangGraph 단일 에이전트(ReAct) + Agent Skills(Knowledge 5종 + Action 4종) + FastAPI SSE + Streamlit UI
- **산출물:** Edge 이벤트 수신 → 결함 진단 → 리포트/작업지시서 생성 → 대화형 상호작용
- **Agent 흐름:** `load_memory` → `reasoning` ↔ `tool_executor` → `parse_diagnosis` → `generate_report` → (조건부) `generate_work_order` → `save_memory`
- **Skills 이원 구조:** Knowledge Skills(md, 프롬프트 주입)이 추론 품질을 제어하고, Action Skills(Python @tool, 서버 직접 호출)이 외부 데이터 조회를 수행

## 2. KPI 달성도 (Plan vs Actual)

| **KPI** | **목표** | **실제** | **비고** |
| --- | --- | --- | --- |
| 고장 유형 진단 정확도 | 85%+ | 주요 결함 구간에서 목표 충족 | 전이 초기는 "의심" 보수적 처리 |
| 추론 근거 설명 품질 | 4/5+ | 평균 4.2/5 | Warning/Critical에서 특히 높음 |
| Tool 호출 효율성 | 정상 0회, 이상 1~2회 | 달성 | "최소 Function Call" 원칙 준수 |
| 처리 시간 | 기존 대비 90%+ 단축 | 수 분 내 완료 | 기존 수동 수 시간 대비 |

## 3. 창출된 핵심 가치

**비즈니스:**

- 이상 감지 → 정비 권고 처리시간 90%+ 단축
- 결함 진행 단계 + RUL 평가로 비계획 다운타임 사전 방지
- Warning/Critical 시 작업지시서 자동 생성으로 현장 정비 준비 시간 단축

**기술:**

- Edge(수치 계산) / Agent(해석) 역할 분리로 독립적 발전 가능
- Skills 이원 구조 — Knowledge Skills(도메인 지식 모듈화) + Action Skills(실행형 도구). 새 설비 확장 시 Skill 추가만으로 대응
- Action Skills가 RAGServer를 in-process 직접 호출하여 프로토콜 오버헤드 제거
- LangGraph → FastAPI SSE → Streamlit 실시간 스트리밍 파이프라인

## 4. 운영 및 보안 고려 사항

- **안전장치:** tool_calls_count > 10 강제 종료, 수치 계산 금지 (프롬프트 레벨), asyncio.timeout(300)
- **Memory:** PostgreSQL 영구 저장 (정상 포함), 최근 5건 컨텍스트 로드, Qdrant analysis_history 벡터 적재
- **Action Skills:** RAGServer/NotificationServer 지연 초기화 (싱글턴). 프로덕션에서 MCP transport로 전환하여 분리 배포 가능
- **에러 처리:** LLM 실패 시 graceful degradation, SSE 종료 시 Queue 정리, JSON 파싱 실패 시 fallback 기본값

## 5. 회고 및 향후 확장

### 기술적 한계

- IMS Bearing 오픈 데이터셋 기반 검증 — 실제 산업 환경 다양한 조건 미검증
- Mock Edge 시뮬레이션 — 실시간 센서 스트리밍 미구현
- GPT-4o 단일 모델 의존 — 모델 변경 시 프롬프트 재최적화 필요
- Dense-only 검색 — 설계된 Hybrid Search(BM25) 미적용

### Next Step (v1.0 비전)

| **확장 영역** | **내용** |
| --- | --- |
| **Self-Evolving Skills** | 분석 결과 vs 실제 고장 대조로 Skill 해석 규칙 자동 보정. 정적 SKILL.md → 피드백 기반 자기 진화 |
| **Deep Search Engine (Group Agent)** | 복수 검색 에이전트 병렬 탐색 (내부 RAG + 외부 웹 + 논문 DB). PdM Agent 본체는 단일 유지 |
| **Personalized Skills** | `skills/users/{user_id}/`에 사용자별 Skill 배치. 설비 담당자별 특화 지식 자동 로드 |
| **Skills Store** | core/(공통) + users/(개인화) 분리 관리. 검증 파이프라인으로 users → core 승격 |
