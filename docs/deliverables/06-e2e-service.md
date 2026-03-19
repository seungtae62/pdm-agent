# 06. E2E 서비스 통합

> PdM Agent — 예지보전 AI 에이전트

## 최종 아키텍처 요약

- **아키텍처:** LangGraph 단일 에이전트(ReAct) + Agent Skills(Knowledge 5종 + Action 5종) + FastAPI SSE + Streamlit UI
- **산출물:** Edge 이벤트 수신 → 결함 진단 → 리포트/작업지시서 생성 → 대화형 상호작용
- **Agent 흐름:** `load_memory` → `reasoning` ↔ `tool_executor` → `parse_diagnosis` → `generate_report` → (조건부) `generate_work_order` → `save_memory`
- **Skills 이원 구조:** Knowledge Skills(md, 프롬프트 주입)이 추론 품질을 제어하고, Action Skills(Python @tool, 서버 직접 호출)이 외부 데이터 조회를 수행

## KPI 달성도 (Plan vs Actual)

| **KPI** | **목표** | **실제** | **비고** |
| --- | --- | --- | --- |
| 고장 유형 진단 정확도 | 85%+ | 주요 결함 구간에서 목표 충족 | 전이 초기는 "의심" 보수적 처리 |
| 추론 근거 설명 품질 | 4/5+ | 평균 4.2/5 | Warning/Critical에서 특히 높음 |
| Tool 호출 효율성 | 정상 0회, 이상 1~2회 | 달성 | "최소 Function Call" 원칙 준수 |
| 처리 시간 | 기존 대비 90%+ 단축 | 수 분 내 완료 | 기존 수동 수 시간 대비 |

## 창출된 핵심 가치

**비즈니스:**

- 이상 감지 → 정비 권고 처리시간 90%+ 단축
- 결함 진행 단계 + RUL 평가로 비계획 다운타임 사전 방지
- Warning/Critical 시 작업지시서 자동 생성으로 현장 정비 준비 시간 단축

**기술:**

- Edge(수치 계산) / Agent(해석) 역할 분리로 독립적 발전 가능
- Skills 이원 구조 — Knowledge Skills(도메인 지식 모듈화) + Action Skills(실행형 도구). 새 설비 확장 시 Skill 추가만으로 대응
- Action Skills가 RAGServer를 in-process 직접 호출하여 프로토콜 오버헤드 제거
- LangGraph → FastAPI SSE → Streamlit 실시간 스트리밍 파이프라인

## 운영 및 보안 고려 사항

- **안전장치:** tool_calls_count > 10 강제 종료, 수치 계산 금지 (프롬프트 레벨), asyncio.timeout(300)
- **Memory:** PostgreSQL 영구 저장 (정상 포함), 최근 5건 컨텍스트 로드, Qdrant analysis_history 벡터 적재
- **Action Skills:** RAGServer/NotificationServer 지연 초기화 (싱글턴). 프로덕션에서 MCP transport로 전환하여 분리 배포 가능
- **에러 처리:** LLM 실패 시 graceful degradation, SSE 종료 시 Queue 정리, JSON 파싱 실패 시 fallback 기본값

## Self-Evolving Skills Engine

### 개념

분석 결과와 실제 고장 결과를 대조하여 Skills를 자동 보정하는 메커니즘. 현재 정적 SKILL.md 파일에 하드코딩된 해석 규칙을 **피드백 기반 자기 진화 시스템**으로 전환한다. Agent가 반복 분석을 수행할수록 도메인 지식이 자동으로 축적되며, 조직의 설비 운영 노하우가 명시적 자산으로 관리된다.

### 아키텍처

**Learning Loop (Core Skills 자동 보정)**

분석 완료 → `save_memory` 노드에서 trigger → 실제 고장 결과와 분석 결과 diff 감지 → Core Skills 업데이트

| **단계** | **동작** | **대상** |
| --- | --- | --- |
| 1. 분석 완료 | Agent가 진단 결과를 Memory에 저장 | `save_memory` 노드 |
| 2. Trigger 발생 | 저장 시점에 `SkillEvolver` 자동 호출 | `SkillEvolver` 클래스 |
| 3. Diff 감지 | 이전 분석 예측 vs 실제 결과 대조 | analysis_history (Qdrant) |
| 4. Skill 패치 생성 | 해석 규칙 수정안을 LLM으로 생성 | Knowledge Skills (md) |
| 5. Core Skills 업데이트 | 검증 후 `core/` 디렉토리에 반영 | `skills/core/*.md` |

**User Feedback Loop (User Skills 자동 생성)**

대화 완료 → 사용자 선호 패턴 분석 → User Skills 자동 생성

- 사용자가 반복적으로 요청하는 분석 관점, 선호하는 리포트 형식, 자주 참조하는 설비 조건 등을 자동으로 캡처
- 설비 담당자별 특화 지식이 `skills/users/{user_id}/` 경로에 자동 배치되어 개인화된 분석 품질 제공

**Skills Store 구조**

```
skills/
├── core/                    # 조직 공통 Skills (검증됨, 전체 Agent 공유)
│   ├── vibration-analysis.md
│   ├── bearing-fault-rules.md
│   └── ...
└── users/                   # 개인화 Skills (자동 생성, 사용자별 격리)
    ├── user-001/
    │   ├── preferred-report-format.md
    │   └── equipment-notes.md
    └── user-002/
        └── ...
```

**승격 파이프라인: `users/` → `core/`**

- User Skill이 일정 횟수 이상 참조되고, 다른 사용자에게도 유효한 인사이트를 포함할 경우 `core/`로 승격
- 승격 조건: 참조 횟수 threshold 충족 + 관리자 승인 또는 자동 검증 파이프라인 통과
- 승격 시 개인 맥락을 제거하고 범용 규칙으로 일반화

### 구현 설계

**`SkillEvolver` 클래스**

- `save_memory` 노드 실행 시 자동 호출
- 분석 결과 diff를 감지하여 Skill 패치를 생성하는 핵심 컴포넌트
- 동작 흐름: 최근 분석 결과 조회 → 해당 설비의 실제 후속 상태 확인 → 예측-실제 간 gap 분석 → Skill 수정 제안 생성 → 검증 후 반영

**Version Control**

각 Skill 파일에 YAML frontmatter로 버전 관리 메타데이터를 기록한다:

```yaml
---
version: 1.3
updated_at: 2026-03-15T14:30:00+09:00
updated_by: SkillEvolver
reason: "Bearing inner race fault 임계값 조정 — 실제 고장 데이터 기반 (case #127, #134)"
changelog:
  - v1.3: BPFI 임계값 0.35 → 0.28 하향 (조기 감지율 향상)
  - v1.2: RMS 가속도 해석 구간 세분화
  - v1.1: 초기 피드백 반영
---
```

### 기대 효과

- **도메인 지식 자동 축적**: 반복 분석을 통해 해석 규칙이 지속적으로 정교화
- **조직 노하우의 명시적 자산화**: 암묵지(경험 기반 판단)가 Skill 파일로 코드화되어 인력 이동에도 지식 유실 방지
- **개인화와 공통화의 균형**: User Skills로 개인 맥락 반영, Core Skills 승격으로 조직 전체 품질 향상

---

## Deep Search Engine (Group Agent)

### 개념

복잡한 분석 요청 시 복수 검색 에이전트가 **Plan & Execute 패턴**으로 병렬 탐색을 수행하는 Group Agent 아키텍처. PdM Agent 본체는 단일 ReAct 에이전트를 유지하면서, 고난도 분석이 필요한 경우에만 Deep Search Engine을 조건부 호출한다. 내부 RAG, 외부 웹, 논문 DB 등 다양한 소스를 동시에 탐색하여 다각도 근거를 확보한다.

### 아키텍처

**Leader-SubAgent 구조**

| **컴포넌트** | **역할** | **구현** |
| --- | --- | --- |
| **Leader Agent** | 검색 계획 수립 + 결과 종합 + 최종 판단 | LangGraph 서브그래프 오케스트레이터 |
| **Sub-Agent 1 (Internal)** | 내부 RAG 심층 검색 (Qdrant analysis_history + 설비 문서) | Qdrant 벡터 검색 + 리랭킹 |
| **Sub-Agent 2 (Web)** | 외부 웹 검색 (기술 문서, 제조사 매뉴얼, 포럼) | Tavily Search API |
| **Sub-Agent 3 (Academic)** | 논문 DB 검색 (arXiv, IEEE 등 학술 자료) | arXiv API + 논문 벡터 스토어 |

**실행 흐름**

PdM Agent `reasoning` → 트리거 조건 감지 → Deep Search Engine 호출 → Leader가 검색 계획 수립 → Sub-Agent 1~N 병렬 실행 → 각 Sub-Agent가 Plan & Execute로 독립 탐색 → Leader가 결과 종합 → PdM Agent `reasoning`에 반환

- 각 Sub-Agent는 독립적인 Plan & Execute 루프를 수행하며, 자체적으로 검색 전략을 세우고 결과를 평가
- Leader Agent는 모든 Sub-Agent 결과를 종합하여 신뢰도 기반 최종 판단을 생성
- LangGraph 서브그래프로 구현되어 PdM Agent 본체의 `reasoning` 노드에서 조건부 호출

### 트리거 조건

| **조건** | **설명** | **판단 기준** |
| --- | --- | --- |
| **RAG 검색 결과 불충분** | 내부 검색으로 충분한 근거 확보 실패 | confidence score < threshold (예: 0.7) |
| **사용자 명시적 요청** | "자세히 분석해줘", "근거를 더 찾아줘" 등 | 프롬프트 내 deep research 키워드 감지 |
| **Critical + 유사 사례 부족** | health_state가 critical이면서 참고할 유사 사례가 희소 | health_state == "critical" AND 유사 사례 < 2건 |

### 구현 설계

**`deep_search_graph.py`**

LangGraph StateGraph 기반 서브그래프:

```
plan → search (병렬) → evaluate → synthesize
```

| **노드** | **동작** |
| --- | --- |
| `plan` | Leader가 검색 쿼리 분해 + Sub-Agent별 검색 전략 할당 |
| `search` | Sub-Agent 1~N 병렬 실행. 각각 할당된 소스에서 Plan & Execute 수행 |
| `evaluate` | 각 결과의 관련성, 신뢰도, 일관성 평가. 충분하지 않으면 `plan`으로 재순환 |
| `synthesize` | 평가를 통과한 결과를 종합하여 구조화된 근거 세트 생성 |

**Sub-Agent별 담당 소스**

| **Sub-Agent** | **검색 소스** | **도구** |
| --- | --- | --- |
| Internal Search | analysis_history, 설비 문서, 정비 이력 | Qdrant 벡터 검색 |
| Web Search | 기술 문서, 제조사 매뉴얼, 기술 포럼 | Tavily Search API |
| Academic Search | 학술 논문, 기술 표준 문서 | arXiv API, 논문 벡터 스토어 |

**PdM Agent 연동**

- Leader Agent가 종합한 결과를 PdM Agent의 `reasoning` 노드에 구조화된 형태로 반환
- 반환 형식: 근거 목록(출처, 신뢰도, 요약) + 종합 판단 + 추가 조사 필요 여부

### 기대 효과

- **단일 RAG 검색 한계 극복**: 내부 데이터만으로 부족한 경우 외부 소스를 동시에 탐색
- **다각도 근거 확보**: 내부 이력, 웹 문서, 학술 논문 등 복수 소스의 교차 검증으로 분석 근거 강화
- **분석 신뢰도 향상**: 특히 Critical 상태에서 충분한 근거 없이 판단하는 위험을 최소화

---

## 회고 및 향후 확장

### 기술적 한계

- IMS Bearing 오픈 데이터셋 기반 검증 — 실제 산업 환경 다양한 조건 미검증
- Mock Edge 시뮬레이션 — 실시간 센서 스트리밍 미구현
- GPT-4o 단일 모델 의존 — 모델 변경 시 프롬프트 재최적화 필요
- Dense-only 검색 — 설계된 Hybrid Search(BM25) 미적용
