# 06. E2E 서비스 통합

> PdM Agent — 예지보전 AI 에이전트

## 최종 아키텍처 요약

- **아키텍처:** LangGraph 단일 에이전트(ReAct) + Agent Skills(Core Skills 5종 + Action Skills 5종 + User Skills) + Deep Search Engine(STORM 스타일 서브그래프) + FastAPI SSE + Streamlit UI
- **산출물:** Edge 이벤트 수신 → 결함 진단 → 리포트/작업지시서 생성 → 대화형 상호작용 (Deep Search 포함)
- **Agent 흐름:** `load_memory` → `reasoning` ↔ `tool_executor` → `parse_diagnosis` → `generate_report` → (조건부) `generate_work_order` → `save_memory`
- **Deep Search 흐름:** `decompose` → `research` (3개 병렬) → `review` (Critic Pass/Revise) → `synthesize` (confidence score 가중치 투표)
- **Skills 3원 구조:** Action Skills(외부 데이터 조회), Core Skills(도메인 판단 지침, md 프롬프트 주입), User Skills(사용자 개인화)로 구성. Core Skills가 추론 품질을 제어하고, Action Skills(Python @tool, 서버 직접 호출)가 외부 데이터 조회를 수행

## KPI 달성도 (Plan vs Actual)

| **KPI** | **목표** | **실제** | **비고** |
| --- | --- | --- | --- |
| KPI 1: 이상 감지 → 정비 권고 처리시간 단축률 | 90% 이상 단축 | 수 분 내 완료 (기존 수동 수 시간 대비 90%+ 단축) | 이벤트 수신 ~ 작업지시서 생성까지 E2E 측정 |
| KPI 2: Skills 도입 토큰 효율성 개선율 | 정상 이벤트 60% 절감 | 정상 ~6K (도입 전 ~15K 대비 60% 절감), 이상 ~12K | Core Skills 조건부 로딩으로 달성 |
| KPI 3: Deep Search 다관점 분석 품질 | Critic Pass Rate 80%+, Confidence Score 평균 0.7+ | Critic Pass Rate 목표 충족, KB 기반 Confidence Score 0.7+ | 병렬 실행으로 순차 대비 응답 시간 단축 |

## 창출된 핵심 가치

**비즈니스:**

- 이상 감지 → 정비 권고 처리시간 90%+ 단축
- 결함 진행 단계 + RUL 평가로 비계획 다운타임 사전 방지
- Warning/Critical 시 작업지시서 자동 생성으로 현장 정비 준비 시간 단축

**기술:**

- Edge(수치 계산) / Agent(해석) 역할 분리로 독립적 발전 가능
- Skills 3원 구조 — Action Skills(외부 데이터 조회) + Core Skills(도메인 판단 지침) + User Skills(사용자 개인화). Core Skills는 컴포넌트 단위로 모듈화되어 있어, 베어링 외에 모터·펌프 등 다른 설비 유형의 Skill을 추가하면 동일 에이전트가 복합 장비를 진단할 수 있다. 컴포넌트 간 상관 분석 Skill(예: 모터 전류 불균형 → 구동측 베어링 편심 하중 영향)을 추가하면 개별 부품이 아닌 장비 전체 맥락에서의 진단이 가능
- Action Skills가 RAGServer를 in-process 직접 호출하여 프로토콜 오버헤드 제거
- LangGraph → FastAPI SSE → Streamlit 실시간 스트리밍 파이프라인

## 운영 및 보안 고려 사항

- **안전장치:** tool_calls_count > 10 강제 종료, 수치 계산 금지 (프롬프트 레벨), asyncio.timeout(300)
- **Memory:** PostgreSQL 영구 저장 (정상 포함), 최근 5건 컨텍스트 로드, Qdrant analysis_history 벡터 적재
- **Action Skills:** RAGServer/NotificationServer 지연 초기화 (싱글턴). 프로덕션에서 MCP transport로 전환하여 분리 배포 가능
- **에러 처리:** LLM 실패 시 graceful degradation, SSE 종료 시 Queue 정리, JSON 파싱 실패 시 fallback 기본값

## Self-Evolving Skills Engine

### 개념

Agent가 운영 과정에서 Skills를 자동으로 확장·개인화하는 자기 진화 메커니즘. 현재는 User Skills 자동 생성과 Action Skills 확장이 핵심이며, Core Skills 갱신은 추후 방향성으로 설계되어 있다.

### 아키텍처

**User Feedback Loop (User Skills 자동 생성)**

현재 구현된 Self-Evolving의 핵심 기능. 대화 완료 시 `SkillEvolver`가 사용자의 반복 요청 패턴을 자동으로 감지하여 User Skills를 생성한다.

- 감지 대상: 반복적으로 요청하는 분석 관점, 선호하는 리포트 형식, 자주 참조하는 설비 조건 등
- 생성 프로세스: 대화 패턴 분석 → 사용자 선호 추출 → YAML frontmatter 기반 Skill 파일 자동 생성 → `skills/users/{user_id}/`에 배치
- 다음 세션부터 자동 로드되어 담당자별 맥락에 맞는 개인화된 분석 품질 제공

**Action Skills 자동 확장**

새로운 외부 데이터 소스나 도구가 필요할 때, Action Skills를 추가하여 Agent의 실행 능력을 확장할 수 있다. Python `@tool` 데코레이터 기반 함수를 `skills/actions/` 디렉토리에 추가하는 방식으로 기존 Skills에 영향 없이 독립적으로 관리된다.

**Core Skills 갱신 방향성 (추후)**

Core Skills는 현재 자동 보정 대상이 아니다. 추후 다음과 같은 단계적 접근을 계획하고 있다:

- Deep Search Engine을 활용하여 월 1회 정도 최신 논문/기술 문헌을 자동 탐색, Core Skills에 추가 가능한 새로운 도메인 지식을 파악
- 파악된 내용은 Human-in-the-Loop으로 도메인 전문가가 검토 후 Core Skills에 반영
- 자동 갱신이 아닌 전문가 승인 기반으로 운영하여 안전성 확보

**Skills Store 구조**

```
skills/
├── actions/                 # Action Skills (외부 데이터 조회, Python @tool)
│   ├── rag_search.py
│   ├── web_search.py
│   └── notification.py
├── core/                    # Core Skills (도메인 판단 지침, md)
│   ├── fault-diagnosis.md
│   ├── feature-interpret.md
│   ├── deep-research.md
│   ├── response-normal.md
│   └── response-alert.md
└── users/                   # User Skills (사용자 개인화, md)
    ├── user-001/
    │   ├── preferred-report-format.md
    │   ├── equipment-notes.md
    │   └── analysis-focus.md
    └── user-002/
        └── ...
```

### 기대 효과

- **개인화된 분석 품질**: User Skills 자동 생성으로 담당자별 맥락을 반영하여 개인화된 분석 경험 제공
- **확장 가능한 실행 능력**: Action Skills 추가로 새로운 외부 데이터 소스·도구를 유연하게 통합
- **조직 노하우의 명시적 자산화**: 암묵지(경험 기반 판단)가 Skill 파일로 코드화되어 인력 이동에도 지식 유실 방지
- **안전한 도메인 지식 갱신**: Core Skills는 Human-in-the-Loop으로 검증 후 반영하여 신뢰성 확보

---

## Deep Search Engine (STORM 스타일 다관점 분석)

### 개념

복잡한 분석 요청 시 **STORM 스타일 다관점 분석**을 수행하는 Group Agent 아키텍처. PdM Agent 본체는 단일 ReAct 에이전트를 유지하면서, 고난도 분석이 필요한 경우에만 Deep Search Engine을 조건부 호출한다. Leader가 질의를 분해하고, 3개 고정 Research Agent가 각자의 전문 관점에서 병렬 검색을 수행하며, Critic의 Review-Revise 루프를 거쳐 confidence score 기반 가중치 투표로 최종 합성한다.

### 아키텍처

**Leader-Research-Critic-Synthesize 구조**

| **컴포넌트** | **역할** | **구현** |
| --- | --- | --- |
| **Leader** | 사용자 질의를 분해(Decompose)하여 3개 Research Agent에 검색 지시 배포 | LangGraph 서브그래프 `decompose` 노드 |
| **Maintenance Engineer** | 정비 이력 관점에서 유사 고장 사례, 정비 경과, 근본 원인 검색 | agent_role: `maintenance_history`, tools: `search_maintenance_history` |
| **Senior Analyst** | 분석 이력 관점에서 과거 유사 패턴의 에이전트 판단 및 결과 검색 | agent_role: `analysis_history`, tools: `search_analysis_history` |
| **Equipment Specialist** | 설비 매뉴얼 + 외부 기술 자료 관점에서 결함 메커니즘, 정비 절차 검색 | agent_role: `equipment_manual`, tools: `search_equipment_manual` + `search_web` |
| **Critic** | 각 Research Agent 결과를 검증하여 Pass/Revise 판정 | LangGraph `review` 노드 |
| **Synthesizer** | confidence score 기반 가중치 투표로 합성 | LangGraph `synthesize` 노드 |

**실행 흐름**

```
START → decompose → research (3개 병렬, asyncio.gather) → review → (조건부)
                                                                      ├─ Revise → research (실패한 perspective만)
                                                                      └─ Pass → synthesize → END
```

- 3개 Research Agent는 `asyncio.gather()`로 동시에 실행
- Critic은 각 결과를 독립적으로 Pass/Revise 판정, Revise 시 해당 perspective만 재검색 (최대 3회)
- confidence score(0.0~1.0): 검색 결과 양/품질 기반 휴리스틱으로 산출
- 합성 시 각 전문가의 confidence score에 비례하여 기여도(가중치) 결정

### 트리거 조건

| **조건** | **설명** | **판단 기준** | **구현 상태** |
| --- | --- | --- | --- |
| **사용자 명시적 요청** | UI에서 Deep Search 토글 On/Off로 사용자가 직접 활성화 | 사용자 설정 기반 | 현재 구현 |
| **RAG 검색 결과 불충분 + 유사 사례 부족** | 내부 RAG 검색으로 충분한 근거 확보 실패 시, 에이전트가 자동으로 Deep Search를 발동 | confidence score < threshold + 유사 사례 < 2건 | 추후 확장 |

### 구현 설계

**`deep_search_graph.py`**

LangGraph StateGraph 기반 서브그래프:

| **노드** | **동작** |
| --- | --- |
| `decompose` | Leader가 사용자 질의를 분석하여 3개 Research Agent에 대한 검색 지시(sub-queries) 생성 |
| `research` | 3개 Research Agent가 `asyncio.gather()`로 병렬 실행. 각 perspective별 도구로 검색 수행 후 confidence score 산출 |
| `review` | Critic이 각 Research Agent 결과를 Pass/Revise 판정. Revise 시 해당 perspective만 재검색 (최대 3회) |
| `synthesize` | confidence score 기반 가중치 투표로 3개 관점의 결과를 종합하여 구조화된 근거 세트 생성 |

**PdM Agent 연동**

- Synthesizer가 종합한 결과를 PdM Agent의 `reasoning` 노드에 구조화된 형태로 반환
- 반환 형식: 전문가별 근거 목록(출처, confidence score, 요약) + 가중치 투표 결과 + 종합 판단
- 안전장치: Review-Revise 최대 3회 + 전체 서브그래프 타임아웃

### 기대 효과

- **단일 RAG 검색 한계 극복**: 3개 전문가 관점에서 동시에 탐색하여 근거 보강
- **다관점 교차 검증**: 정비 이력, 분석 이력, 설비 매뉴얼/외부 기술 자료의 다관점 교차 검증으로 분석 근거 강화
- **품질 보장**: Critic의 Review-Revise 루프로 검색 결과 품질을 보장하고, confidence score로 신뢰도 정량화
- **병렬 실행 효율**: `asyncio.gather()` 병렬 실행으로 순차 검색 대비 응답 시간 단축

---

## 회고 및 향후 확장

### 기술적 한계

- IMS Bearing 오픈 데이터셋 기반 검증 — 실제 산업 환경 다양한 조건 미검증
- Mock Edge 시뮬레이션 — 실시간 센서 스트리밍 미구현
- GPT-4o 단일 모델 의존 — 모델 변경 시 프롬프트 재최적화 필요
- Dense-only 검색 — 설계된 Hybrid Search(BM25) 미적용
