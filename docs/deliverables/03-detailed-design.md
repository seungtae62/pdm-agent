# 03. 상세 설계

> PdM Agent — 예지보전 AI 에이전트

## Agent 페르소나 및 시스템 프롬프트 (Identity)

Agent의 정체성, 역할, 그리고 답변의 톤앤매너를 정의합니다. 실제 LLM의 System Prompt에 들어갈 핵심 내용입니다.

| **항목** | **정의 내용** |
| --- | --- |
| **Agent 이름** | PdM Agent |
| **주요 역할** | Edge 시스템에서 전달받은 이벤트 페이로드(이상감지 결과, 진동 특징량, 추세 데이터, ML RUL 예측값)를 전문가 관점에서 해석하여 결함 유형 식별, 결함 진행 단계 판정, RUL 맥락 평가, 위험도 종합 판단, 정비 권고 생성을 수행하는 베어링 예지보전 전문 AI 에이전트 |
| **핵심 목표** | 베어링 설비의 이상 이벤트에 대해, 도메인 지식 기반의 전문가 수준 해석과 근거 있는 정비 의사결정을 자동화하여, 비계획 정지를 사전에 방지하고 정비 리드타임을 확보하는 것 |
| **톤앤매너** | 전문가적 어조로 근거 기반 판단을 제시하며, 불확실성이 있을 경우 그 수준과 원인을 투명하게 고지함. 정상 상태에서는 간결하게, 심각한 이상 상황에서는 상세하고 구조화된 분석을 제공함. 추측이나 일반론이 아닌 도메인 지식에 근거한 판단을 우선함 |
| **제약 사항** | (1) 수치 계산 금지: RMS 산출, FFT 분석, 통계 연산, 추세선 기울기 계산 등 정량적 연산은 Edge 시스템이 담당하며 에이전트는 산출된 값을 해석만 수행 (2) 추측 금지: 데이터가 부족하거나 판단이 불확실할 때 확신 없는 결론을 생성하지 않으며, 불확실성의 수준과 원인을 명시 (3) 단일 에이전트 제약: 역할별 멀티 에이전트 분리 없이 하나의 에이전트로 모든 분석 수행 |

**시스템 프롬프트 핵심 구성:**

에이전트의 시스템 프롬프트는 다음 요소들로 구성됩니다.

| **구성 요소** | **상주/온디맨드** | **설명** |
| --- | --- | --- |
| 역할 및 원칙 정의 | 상주 (시스템 프롬프트) | 에이전트의 정체성, 5개 핵심 원칙 (수치 계산 금지, 도메인 지식 기반 해석, 추론 깊이 자율 조절, 불확실성 투명 고지, 능동적 정보 획득) |
| 추론 절차 | 상주 (시스템 프롬프트) | 5단계 Thought 구조 (초기 판별 → 단계 판정 → 열화 평가 → RUL 평가 → 종합 판단) 및 분기 기준 |
| Action Skills 사용 규칙 | 상주 (시스템 프롬프트) | 5개 Action Skills의 호출 조건 및 효율성 원칙 |
| 도메인 지식 | **온디맨드 (Knowledge Skills)** | fault-diagnosis, feature-interpret 등 SKILL.md로 분리. 추론 과정에서 필요 시에만 로드 |
| 위험도 기준 및 응답 양식 | **온디맨드 (Knowledge Skills)** | response-normal · response-alert Skill로 분리. 위험도 판정 후 로드 |
| Deep Research 절차 | **온디맨드 (Knowledge Skills)** | deep-research Skill로 분리. 대화형 상호작용에서 사용자 요청 시에만 로드 |
| 대화형 상호작용 규칙 | 상주 (시스템 프롬프트) | 분석 맥락 유지, Prompt Optimization 적용 조건 |

## 워크플로우 및 오케스트레이션 (Workflow & Logic)

이벤트 페이로드 수신부터 최종 정비 권고 및 리포트 생성까지 Agent의 사고 과정과 행동 순서를 기술합니다.

### 처리 로직

- **Step 1 (Input Analysis — 이벤트 수신 및 맥락 로드):**
    - Edge 시스템에서 이벤트 페이로드(JSON)를 수신
    - 페이로드에서 설비 ID, 베어링 ID를 추출하여 Long-term Memory(PostgreSQL)에서 해당 설비의 이전 분석 판단 이력(최근 5건)을 조회
    - 이전 이력이 있으면 자연어로 요약하여 추론 컨텍스트에 주입 (예: "5일 전 정상 판정, 15일 전 Watch 판정 이력 존재")

- **Step 2 (ReAct Reasoning — 5단계 추론 및 Tool 선택):**
    - Thought 1: `anomaly_detected` 확인. false이면 특징량 교차 확인 후 정상 판정 → 조기 종료. true이면 주파수 영역에서 지배적 결함 주파수 식별 → 결함 유형 판별
    - Thought 2: 시간/주파수 영역 특징량을 종합하여 P-F 곡선 상 결함 진행 단계(1~4단계) 판정. Memory 이전 이력 대비 변화 확인
    - Thought 3: Edge 산출 추세 데이터(slope, trend_direction, acceleration_detected) 해석. 비정상 가속 시 `search_equipment_manual` Tool 호출 가능
    - Thought 4: ML RUL 예측값과 신뢰구간을 에이전트 판정과 대조. 가속 열화 시 신뢰구간 하한(보수적 추정) 채택
    - Thought 5: 위험도 종합 판정(Normal/Watch/Warning/Critical). 필요시 `search_maintenance_history`로 유사 사례 참조. Watch 이상 시 `notify_maintenance_staff` 호출
    - **Tool 선택 기준**: 추론 과정에서 근거 보강이 필요하다고 판단될 때만 호출. 정상 상태에서는 Tool 미호출. 이벤트 분석에서는 일반 RAG 활용(1~2회)으로 충분

    - **Deep Search (분석적 심층 조사 — Deep Search Engine 활용):**
        - 일반 RAG 활용(정보 조회)과 구분되는 분석적 조사 행동 패턴. **Deep Search Engine(Group Agent)을 조건부 호출하여 병렬 탐색을 수행**
        - 발동 조건: 대화형 상호작용에서 사용자의 분석적 질문 시에만 발동 ("근본 원인 분석해줘", "유사 사례 있어?", "왜 급속 열화인가?"), 또는 RAG confidence < threshold, health_state == critical AND 유사 사례 < 2건
        - 이벤트 자동 분석에서는 발동하지 않음 (Critical 포함). 이벤트 분석은 도메인 지식 + 일반 RAG로 빠르게 처리
        - 실행 구조: PdM Agent `reasoning` → 트리거 조건 감지 → **Deep Search Engine 호출** → Leader Agent가 검색 계획 수립 → Sub-Agent 1(Internal: Qdrant 심층 검색) + Sub-Agent 2(Web: Tavily) + Sub-Agent 3(Academic: arXiv) **병렬 실행** → Leader가 결과 종합 → PdM Agent `reasoning`에 구조화된 근거 세트 반환
        - 기존 순차 탐색(가설→RAG→외부검색→해석→재검색 루프)을 Leader + Sub-Agents 병렬 구조로 전환하여 탐색 깊이와 병렬성을 동시에 확보
        - 종료: Leader가 충분한 근거 확보를 판단하여 자율 종료 또는 안전장치 (max_iterations = 3)

- **Step 3 (Output Generation — 결과 구조화 및 리포트 생성):**
    - ReAct 루프 완료 후 LLM 응답에서 `diagnosis_result`를 구조화된 JSON으로 파싱 (결함 유형, 단계, 열화 속도, RUL 평가, 위험도, 권고, 불확실성 고지)
    - Warning/Critical 위험도: 분석 리포트(추론 체인 + Memory 이력 + RAG 검색 결과) 및 작업지시서(정비 절차, 필요 자원, 권장 일정, 안전 주의사항) 생성
    - Normal/Watch 위험도: 리포트 생성 생략, 간결한 상태 확인 또는 모니터링 강화 권고 응답
    - 모든 판단 결과를 Long-term Memory(PostgreSQL)에 저장하고, analysis_history VDB에 벡터 적재

### 상태 관리

LangGraph StateGraph 기반으로 워크플로우 상태를 관리합니다.

- **State 주요 필드:**
    - `event_payload`: Edge에서 수신한 이벤트 데이터
    - `memory_context`: Long-term Memory에서 조회한 이전 판단 이력
    - `messages`: ReAct 추론 과정의 메시지 누적 (Thought-Action-Observation)
    - `diagnosis_result`: 구조화된 진단 결과 (결함 유형, 단계, 위험도, RUL 평가 등)
    - `next_action`: 워크플로우 분기 제어 (continue_reasoning / call_tool / generate_report / end)
    - `tool_calls_count`: Tool 호출 횟수 추적 (안전장치용)

- **Node/Edge 흐름:**

```
START → load_memory → reasoning → (조건부 분기)
                                    ├─ call_tool → tool_executor → reasoning
                                    ├─ continue_reasoning → reasoning
                                    └─ generate_report → parse_diagnosis → generate_report
                                                          → generate_work_order → save_memory → END
```

- **안전장치**: `tool_calls_count > 10`이면 강제로 `parse_diagnosis`로 전이하여 무한 루프 방지
- **조건부 실행**: `generate_work_order`는 Warning/Critical 위험도에서만 실행, Normal/Watch에서는 건너뜀

## 도구(Tools) 및 지식 관리 명세 (Capability)

Agent의 능력은 **Action Skills(실행형 도구)**와 **Knowledge Skills(도메인 지식)**의 이원 구조로 구성됩니다.

- **Action Skills**: 외부 데이터 소스 및 알림 시스템과의 실제 상호작용을 담당합니다. LangChain `@tool`로 구현되며, RAGServer/NotificationServer를 in-process 직접 호출합니다.
- **Knowledge Skills**: 도메인 지식을 SKILL.md 파일로 모듈화하여, 필요 시에만 점진적으로 로드합니다. 시스템 프롬프트에는 최소한의 페르소나와 추론 구조만 유지합니다.

**Action Skills 명세:**

| **도구명 (Function Name)** | **기능 설명 (Description)** | **입력 파라미터 (Input Schema)** | **출력 데이터 (Output)** |
| --- | --- | --- | --- |
| search_maintenance_history | 과거 고장/정비 이력을 의미적으로 검색. 유사 결함 사례의 진행 경과, 근본 원인, 고장까지 소요 시간 등을 참조 | query: str, equipment_id?: str, bearing_id?: str, top_k?: int | 유사 정비 이력 문서 리스트 (유사도 순) |
| search_equipment_manual | 설비 매뉴얼, FMEA 문서, 정비 절차서를 검색. 설비 사양, 결함 메커니즘, 급속 열화 조건, 교체 절차 등 참조 | query: str, doc_type?: str, top_k?: int | 관련 매뉴얼 문서 리스트 (유사도 순) |
| search_analysis_history | 에이전트의 과거 분석 판단 이력을 의미적으로 검색. 유사 패턴의 과거 판단과 결과를 참조하여 일관성 유지 | query: str, equipment_id?: str, bearing_id?: str, top_k?: int | 과거 분석 결과 리스트 (유사도 순) |
| notify_maintenance_staff | 정비 담당자에게 분석 결과 및 정비 권고 알림을 전송 | message: str, risk_level: str, equipment_id: str | 전송 성공/실패 상태 |
| search_web | 외부 인터넷에서 베어링/설비 관련 기술 문헌, 논문, 산업 리포트를 검색. Deep Research에서만 사용하며, 내부 RAG 검색을 보완하는 외부 지식 획득용 | query: str | 검색 결과 리스트 (제목, 요약, URL). "외부 참고 자료 (검증 필요)" 태그 포함 |

**Action Skills 호출 원칙:**
- 정상 상태에서는 Action Skills를 호출하지 않음 (Knowledge Skills로 로드된 도메인 지식으로 판단 완료)
- **일반 RAG 활용 (이벤트 분석)**: 결함 확인 시 근거 보강이 필요할 때 선택적 호출 (정보 조회 목적, 1~2회)
- **Deep Research (대화형에서만)**: 사용자가 분석적 질문 시 발동. 내부 RAG + 외부 웹 검색의 반복적 탐색-해석-후속질문-재검색 루프
- search_web은 Deep Research에서만 사용. 이벤트 자동 분석에서는 호출하지 않음
- Tool 호출 효율성은 KPI로 평가됨

**Knowledge Skills 명세:**

| **Skill 이름** | **로드 조건** | **내용** |
| --- | --- | --- |
| fault-diagnosis | Thought 1에서 anomaly_detected = true 시 로드 | 베어링 결함 주파수 해석 (BPFO, BPFI, BSF, FTF), P-F 곡선 4단계 정의, 고조파/사이드밴드 해석 기준 |
| feature-interpret | Thought 2~3에서 특징량 복합 해석 시 로드 | Kurtosis + RMS 복합 패턴, Crest Factor 전이 패턴, 고주파 에너지 초기 지표 등 특징량 해석 규칙 |
| deep-research | 대화형 상호작용에서 사용자의 분석적 질문 시 로드 | 가설 수립 → 내부 RAG → 외부 검색 → 해석 → 재검색의 탐색 절차, 외부 자료 신뢰도 태깅 규칙 |
| response-normal | Thought 5 위험도 판정 후 Normal/Watch 시 로드 | Normal/Watch 위험도별 응답 양식, 간결 요약 구조 |
| response-alert | Thought 5 위험도 판정 후 Warning/Critical 시 로드 | Warning/Critical 위험도별 응답 양식, 리포트 구조, 작업지시서 포함 기준 |

**User Skills (Personalized Skills) 명세:**

| **Skill 유형** | **저장 경로** | **매칭 조건** | **설명** |
| --- | --- | --- | --- |
| 리포트 형식 선호 | `skills/users/{id}/preferred-report-format.md` | `chat` | 사용자가 선호하는 리포트 구조/상세 수준 |
| 설비 메모 | `skills/users/{id}/equipment-notes.md` | `always` | 담당 설비 특이사항, 과거 경험 메모 |
| 분석 관점 | `skills/users/{id}/analysis-focus.md` | `anomaly_detected` | 사용자가 중시하는 분석 항목 (예: 윤활 상태 우선) |

- YAML frontmatter 기반 Skill 파일, 컨텍스트별 매칭 (`always`, `chat`, `agent`, `anomaly_detected`, `alert`)
- CRUD API: `save_user_skill()`, `delete_user_skill()`, `list_user_skills()`
- 자동 생성: `SkillEvolver`가 대화 패턴에서 사용자 선호를 감지하여 자동 배치

**Skills 점진적 로딩 예시:**
- **정상 이벤트**: 시스템 프롬프트(페르소나 + 추론 구조) → Thought 1 조기 종료 → Skills 미로드 (토큰 절감)
- **이상 이벤트**: 시스템 프롬프트 → fault-diagnosis 로드 → feature-interpret 로드 → response-normal 또는 response-alert 로드
- **대화형 Deep Research**: 위 + deep-research 로드 → Deep Search Engine 조건부 호출
- **Self-Evolving**: save_memory 완료 → SkillEvolver가 분석 결과 diff 감지 → Core Skills 자동 패치 (비동기)
- **대화형 개인화**: 대화 완료 → 사용자 선호 패턴 감지 → User Skills 자동 생성 → 다음 세션부터 자동 로드

## 지식 베이스 및 메모리 전략 (Context & Memory)

LLM이 참조할 외부 지식과 대화/분석 이력의 관리 전략을 수립합니다.

### RAG 전략

- **참조 데이터 소스:**

| **Collection** | **문서 수** | **내용** | **용도** |
| --- | --- | --- | --- |
| maintenance_history | 80건 (PDF) — 완료보고서 40건 + 작업지시서 40건 | 과거 고장/정비 이력. 설비 정보, 고장 원인, 조치 내용, 자재/공구, ISO 14224 코드 포함 | 유사 결함 사례 비교, 근본 원인 추론 보강 |
| equipment_manual | 7건 (PDF) | 베어링 사양서, 결함 유형별 가이드(외륜/내륜/전동체/보지기), 교체 정비 절차서, FMEA | 결함 메커니즘 확인, 급속 열화 조건 참조, 정비 절차 참조, 위험 우선순위 참조 |
| analysis_history | 동적 증가 | 에이전트 과거 분석 판단 결과 (save_memory 시 PostgreSQL + VDB 동시 적재). reasoning_summary + diagnosis_result 임베딩 | 과거 유사 분석 사례의 의미적 검색. PostgreSQL Memory의 SQL 조회와 상호 보완 |

- **Indexing Pipeline:**

    - **청킹(Chunking) 전략:**

    | **Collection** | **청킹 방식** | **설명** |
    | --- | --- | --- |
    | maintenance_history | Document-level (1문서 = 1청크) | 완료보고서·작업지시서 모두 1~2페이지 분량으로, 분할 시 맥락 손실 우려 |
    | equipment_manual | Recursive Chunking (헤더 기반 분할) | 매뉴얼 내 섹션 구조(H2/H3)를 기준으로 재귀 분할. FMEA는 고장 모드별 분할 |
    | analysis_history | Record-level (1레코드 = 1청크) | 에이전트 판단 결과 1건을 하나의 청크로 적재 |

    - **임베딩 전략:** Dense + Sparse 이원 구조

    | **유형** | **모델/방식** | **차원** | **역할** |
    | --- | --- | --- | --- |
    | Dense | OpenAI text-embedding-3-small | 1536 | 의미적 유사도 검색 (Semantic Search) |
    | Sparse | BM25 + 한국어 형태소 분석 (kiwipiepy) | — | 키워드 매칭 (설비 코드, ISO 14224 코드, 고유명사 등) |

    - **Metadata 추출:** PDF 파싱 시 아래 필드를 자동 추출하여 벡터와 함께 저장

        - `equipment_id`, `bearing_id` — 설비/베어링 식별자
        - `doc_type` — 문서 유형 (completion_report, work_order, manual, fmea 등)
        - `fault_type`, `fault_location` — 결함 유형 및 위치 (해당 시)
        - `date` — 문서 작성일

- **Retrieval Pipeline:**

    | **단계** | **방법** | **설명** |
    | --- | --- | --- |
    | 1단계 — Metadata Filtering | Qdrant Payload Filter | `equipment_id`, `bearing_id`, `doc_type` 등으로 후보 문서 사전 필터링 |
    | 2단계 — Hybrid Search | Dense + Sparse → RRF 결합 | Dense 유사도와 Sparse BM25 점수를 Reciprocal Rank Fusion으로 결합하여 최종 스코어 산출 |
    | 3단계 — Reranker | Cross-encoder (모델 선정중) | Top-k 후보를 Cross-encoder로 재정렬하여 정밀도 향상 |
    | 결과 | Top-k 반환 (기본 k=3) | 최종 상위 k건을 Agent 컨텍스트에 주입 |

- **Vector DB:** Qdrant (Docker 자체 호스팅, Dense + Sparse 듀얼 인덱스 지원)

### 대화 메모리

- **메모리 유형:** Short-term Memory + Long-term Memory 이원 구조

| **구분** | **저장소** | **내용** | **용도** |
| --- | --- | --- | --- |
| Short-term | LangGraph State (`messages`) | 현재 분석 세션의 ReAct 추론 과정 (Thought-Action-Observation 체인) | 단일 이벤트 분석 내 맥락 유지 |
| Long-term | PostgreSQL (`pdm_agent_memory` 테이블) | 모든 이벤트(정상 포함)에 대한 판단 결과. 결함 유형, 단계, 위험도, RUL 평가, 권고, Tool 사용 기록, 후속 조치 추적 | 동일 설비의 시계열 변화 추적, "언제부터 이상 시작" 파악, 후속 분석 시 이전 맥락 참조 |

- **저장 전략:**
    - Short-term: 이벤트 분석 완료 시 소멸 (State 수명 = 단일 워크플로우 실행)
    - Long-term: 모든 판단 결과를 영구 저장. 조회 시 설비/베어링 ID 기준 최근 5건 로드
    - 정상 판정도 저장하여, 이상 시작 시점 추적에 활용

- **Prompt Optimization (대화형 상호작용 시):**

| **전략** | **설명** | **적용 시점** |
| --- | --- | --- |
| Skills 점진적 로딩 | 도메인 지식을 Knowledge Skills로 분리. 세션 시작 시 메타데이터(~300토큰)만 로드하고, 추론 시 필요한 Skill만 온디맨드 로드(~5K토큰/Skill). 정상 이벤트에서는 결함 진단 Skill 미로드 | 매 턴 |
| 분석 맥락 구조화 | Memory에서 전체 추론 체인 대신, 핵심 필드(결함 유형, 단계, 위험도, RUL, 핵심 근거)만 추출하여 컨텍스트에 주입 | 대화 세션 시작 시 |
| 대화 이력 압축 | 이전 턴 질문-응답 쌍을 핵심 결론 중심으로 요약 | 매 턴 누적 시 |
| 슬라이딩 윈도우 | 최근 2~3턴은 원문 유지, 이전 턴은 요약으로 대체하여 컨텍스트 윈도우 효율적 활용 | 대화 3턴 이상 시 |

## 핵심 에이전트 기술 스택

에이전트의 추론 품질과 검색 정확도를 제어하기 위한 핵심 기술적 의사결정입니다.

### 추론 전략

| **기술** | **구현** | **선정 사유** |
| --- | --- | --- |
| ReAct 추론 | 5단계 Thought 구조 (결함 식별 → 단계 판정 → 열화 평가 → RUL 평가 → 종합 판단) | Thought-Action-Observation 교차 수행으로 단계적 심화 추론 구현. 각 단계에서 분기/조기종료/Tool 호출을 자율 결정하여, 정상 상태는 Thought 1 조기 종료, 이상 상태는 전체 추론 수행 |
| Deep Search Engine | Deep Search Engine(Group Agent)을 조건부 호출하여 Leader + Sub-Agents 병렬 탐색 | 트리거 조건 충족 시 Leader Agent가 검색 계획을 수립하고 Sub-Agent 1(Internal: Qdrant) + Sub-Agent 2(Web: Tavily) + Sub-Agent 3(Academic: arXiv)이 병렬 탐색 수행. 대화형 상호작용에서 사용자 요청 시 또는 RAG confidence 부족/Critical 상태에서 자동 발동. 외부 자료는 "외부 참고 (검증 필요)" 태그로 소스 신뢰도를 구분 |

### RAG 검색 파이프라인

| **기술** | **구현** | **선정 사유** |
| --- | --- | --- |
| Hybrid Search | Dense (OpenAI text-embedding-3-small, 1536dim) + Sparse (BM25 + kiwipiepy) → RRF (Reciprocal Rank Fusion) 결합 | 의미적 유사도(Dense)와 키워드 매칭(Sparse)을 RRF로 결합. 한국어 형태소 분석(kiwipiepy)으로 설비 코드, ISO 14224 코드 등 도메인 고유명사 검색 정밀도 확보 |
| Metadata Filtering | Qdrant Payload Filter (equipment_id, bearing_id, doc_type) | 검색 전 후보 문서를 설비/베어링/문서유형 기준으로 사전 필터링하여 검색 범위 축소. 불필요한 문서 노이즈 제거 및 정밀도 향상 |
| Reranker | Cross-encoder (모델 선정 중) | Hybrid Search Top-k 후보를 query-document 쌍 단위로 정밀 재정렬. 1차 검색의 recall과 Reranker의 precision을 결합하여 최종 검색 품질 극대화 |
| Embedding 이원 구조 | Dense: OpenAI text-embedding-3-small (1536dim) / Sparse: BM25 + kiwipiepy 형태소 분석 | Dense로 의미적 유사도를, Sparse로 키워드 정확 매칭을 분담. Qdrant의 Dense + Sparse 듀얼 인덱스로 단일 Collection 내에서 두 방식 동시 지원 |

### Tool 호출 아키텍처

| **기술** | **구현** | **선정 사유** |
| --- | --- | --- |
| Tool 구현 | LangChain `@tool` 데코레이터 기반 Action Skills | RAGServer/NotificationServer를 in-process 직접 호출하여 프로토콜 오버헤드 제거. LangGraph의 `bind_tools`로 에이전트에 바인딩 |
| Action Skills 구성 | RAG 검색 3종 (maintenance_history · equipment_manual · analysis_history) · web_search (외부 검색) · notification (알림) | 기능별 분리로 독립 관리 가능. search_web은 Deep Research 전용, notify_maintenance_staff는 Watch 이상 시 호출 |

### 도메인 지식 관리 (Skills)

> Agent Skills 오픈 표준([agentskills.io](https://agentskills.io))을 채택하여, 시스템 프롬프트에 고정되던 도메인 지식을 모듈화합니다.

| **기술** | **구현** | **선정 사유** |
| --- | --- | --- |
| Knowledge Skills | SKILL.md 기반 도메인 지식 모듈화. 점진적 로딩(Progressive Disclosure) 적용 | 세션 시작 시 메타데이터(Skill 이름 + 설명)만 로드하고, 추론 과정에서 필요한 Skill만 온디맨드 로드. 정상 이벤트(Thought 1 조기 종료)에서 불필요한 도메인 지식 로딩을 제거하여 토큰 효율성 확보 |
| Skill 구성 | fault-diagnosis (결함 주파수 + P-F 곡선) · feature-interpret (특징량 복합 해석) · deep-research (심화 조사 절차) · response-normal (Normal/Watch 응답 양식) · response-alert (Warning/Critical 응답 양식) | 추론 단계별로 필요한 지식만 선택 로드. 새 설비 유형(모터, 펌프 등) 확장 시 Skill 파일 추가만으로 대응 가능 |
| Action Skills-Knowledge Skills 역할 분리 | Knowledge Skills = 도메인 지식(뇌), Action Skills = 외부 실행(근육) | Knowledge Skills는 에이전트의 추론 품질을 제어하는 지식/지침, Action Skills는 외부 데이터 소스와의 실제 상호작용을 담당. 관심사 분리로 각 레이어의 독립적 확장 가능 |

## Self-Evolving Skills Engine

분석 결과와 실제 고장 결과를 대조하여 Skills를 자동 보정하는 자기 진화 메커니즘입니다. 정적 SKILL.md 파일에 고정된 해석 규칙을 **피드백 기반 자기 진화 시스템**으로 전환하여, Agent가 반복 분석을 수행할수록 도메인 지식이 자동으로 축적되고, 조직의 설비 운영 노하우가 명시적 자산으로 관리됩니다.

### Learning Loop (Core Skills 자동 보정)

분석 완료 시 `save_memory` 노드에서 `SkillEvolver`를 자동 호출하여, 이전 분석 예측과 실제 결과를 대조하고 Skill 패치를 생성합니다.

| **단계** | **동작** | **대상 컴포넌트** |
| --- | --- | --- |
| 1. 분석 완료 | Agent가 진단 결과를 Memory에 저장 | `save_memory` 노드 |
| 2. Trigger 발생 | 저장 시점에 `SkillEvolver` 자동 호출 (비동기) | `SkillEvolver` 클래스 |
| 3. Diff 감지 | 이전 분석 예측 vs 실제 결과(후속 상태) 대조 | analysis_history (Qdrant) + PostgreSQL Memory |
| 4. Skill 패치 생성 | 예측-실제 간 gap을 분석하여 해석 규칙 수정안을 LLM으로 생성 | Knowledge Skills (md) |
| 5. Core Skills 업데이트 | 검증 후 `skills/core/` 디렉토리에 반영 | `skills/core/*.md` |

**`SkillEvolver` 클래스 설계:**
- `save_memory` 노드 실행 시 자동 호출되는 핵심 컴포넌트
- 동작 흐름: 최근 분석 결과 조회 → 해당 설비의 실제 후속 상태 확인 → 예측-실제 간 gap 분석 → Skill 수정 제안 생성 → 검증 후 반영
- 비동기 실행으로 분석 워크플로우의 응답 지연에 영향을 주지 않음

### User Feedback Loop (User Skills 자동 생성)

대화 완료 시 사용자의 선호 패턴을 분석하여 User Skills를 자동 생성합니다.

- 사용자가 반복적으로 요청하는 분석 관점, 선호하는 리포트 형식, 자주 참조하는 설비 조건 등을 자동 캡처
- 설비 담당자별 특화 지식이 `skills/users/{user_id}/` 경로에 자동 배치되어 개인화된 분석 품질 제공
- `SkillEvolver`가 대화 패턴에서 사용자 선호를 감지하여 User Skill 파일을 자동 생성/갱신

### Skills Store 구조

Knowledge Skills를 `core/`(조직 공통)과 `users/`(개인화)로 계층화하여 관리합니다.

```
skills/
├── core/                    # 조직 공통 Skills (검증됨, 전체 Agent 공유)
│   ├── fault-diagnosis.md
│   ├── feature-interpret.md
│   ├── deep-research.md
│   ├── response-normal.md
│   └── response-alert.md
└── users/                   # 개인화 Skills (자동 생성, 사용자별 격리)
    ├── user-001/
    │   ├── preferred-report-format.md
    │   ├── equipment-notes.md
    │   └── analysis-focus.md
    └── user-002/
        └── ...
```

### 승격 파이프라인: `users/` → `core/`

User Skill이 일정 기준을 충족하면 Core Skills로 승격하여, 개인 노하우를 조직 표준으로 확산합니다.

| **단계** | **조건** | **동작** |
| --- | --- | --- |
| 1. 참조 횟수 모니터링 | User Skill 참조 횟수가 threshold 초과 | 승격 후보로 플래깅 |
| 2. 범용성 검증 | 개인 맥락 의존도 평가 + 타 사용자 적용 가능성 확인 | 관리자 승인 또는 자동 검증 파이프라인 |
| 3. 일반화 | 개인 맥락을 제거하고 범용 규칙으로 변환 | LLM 기반 일반화 + 사람 검토 |
| 4. Core 반영 | `skills/core/`에 신규 Skill 또는 기존 Skill 패치로 반영 | 버전 업데이트 + changelog 기록 |

### Version Control

각 Skill 파일에 YAML frontmatter로 버전 관리 메타데이터를 기록합니다.

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

- `version`: 현재 버전 번호 (SemVer 단순화)
- `updated_at`: 최종 수정 시각 (ISO 8601)
- `updated_by`: 수정 주체 (`SkillEvolver` / 관리자 이름)
- `reason`: 수정 사유 (실제 사례 번호 포함)
- `changelog`: 주요 변경 이력 역순 기록

### 기대 효과

| **효과** | **설명** |
| --- | --- |
| **도메인 지식 자동 축적** | 반복 분석을 통해 해석 규칙이 지속적으로 정교화. 분석 건수 증가에 비례하여 진단 정확도 향상 |
| **조직 노하우의 명시적 자산화** | 암묵지(경험 기반 판단)가 Skill 파일로 코드화되어 인력 이동에도 지식 유실 방지 |
| **개인화-공통화 균형** | User Skills로 담당자별 맥락을 반영하고, 승격 파이프라인을 통해 검증된 지식을 조직 전체로 확산 |

---

## Deep Search Engine (Group Agent)

복잡한 분석 요청 시 복수 검색 에이전트가 **Plan & Execute 패턴**으로 병렬 탐색을 수행하는 Group Agent 아키텍처입니다. PdM Agent 본체는 단일 ReAct 에이전트를 유지하면서, 고난도 분석이 필요한 경우에만 Deep Search Engine을 조건부 호출합니다.

### Leader-SubAgent 구조

| **컴포넌트** | **역할** | **구현** |
| --- | --- | --- |
| **Leader Agent** | 검색 계획 수립 + 결과 종합 + 최종 판단 | LangGraph 서브그래프 오케스트레이터 |
| **Sub-Agent 1 (Internal)** | 내부 RAG 심층 검색 (Qdrant analysis_history + 설비 문서 + 정비 이력) | Qdrant 벡터 검색 + 리랭킹 |
| **Sub-Agent 2 (Web)** | 외부 웹 검색 (기술 문서, 제조사 매뉴얼, 기술 포럼) | Tavily Search API |
| **Sub-Agent 3 (Academic)** | 논문 DB 검색 (arXiv, IEEE 등 학술 자료) | arXiv API + 논문 벡터 스토어 |

### 실행 흐름

```
PdM Agent reasoning → 트리거 조건 감지 → Deep Search Engine 호출
  → Leader: 검색 계획 수립 (쿼리 분해 + Sub-Agent별 전략 할당)
    → Sub-Agent 1~N: 병렬 실행 (각각 Plan & Execute 독립 탐색)
  → Leader: 결과 종합 (신뢰도 기반 판단)
→ PdM Agent reasoning에 구조화된 근거 세트 반환
```

- 각 Sub-Agent는 독립적인 Plan & Execute 루프를 수행하며, 자체적으로 검색 전략을 수립하고 결과를 평가
- Leader Agent는 모든 Sub-Agent 결과를 종합하여 신뢰도 기반 최종 판단을 생성
- LangGraph 서브그래프로 구현되어 PdM Agent 본체의 `reasoning` 노드에서 조건부 호출

### 트리거 조건

| **조건** | **설명** | **판단 기준** |
| --- | --- | --- |
| **RAG 검색 결과 불충분** | 내부 검색으로 충분한 근거 확보 실패 | confidence score < threshold (예: 0.7) |
| **사용자 명시적 요청** | "자세히 분석해줘", "근거를 더 찾아줘" 등 | 프롬프트 내 deep search 키워드 감지 |
| **Critical + 유사 사례 부족** | health_state가 critical이면서 참고할 유사 사례가 희소 | health_state == "critical" AND 유사 사례 < 2건 |

### 구현 설계: `deep_search_graph.py`

LangGraph StateGraph 기반 서브그래프로 구현합니다.

| **노드** | **동작** | **분기 조건** |
| --- | --- | --- |
| `plan` | Leader가 검색 쿼리 분해 + Sub-Agent별 검색 전략 할당 | 항상 `search`로 전이 |
| `search` | Sub-Agent 1~N 병렬 실행. 각각 할당된 소스에서 Plan & Execute 수행 | 항상 `evaluate`로 전이 |
| `evaluate` | 각 결과의 관련성, 신뢰도, 일관성 평가 | 충분 → `synthesize` / 불충분 → `plan` 재순환 |
| `synthesize` | 평가를 통과한 결과를 종합하여 구조화된 근거 세트 생성 | 종료 → PdM Agent에 반환 |

**Sub-Agent별 담당 소스:**

| **Sub-Agent** | **검색 소스** | **도구** |
| --- | --- | --- |
| Internal Search | analysis_history, 설비 문서, 정비 이력 | Qdrant 벡터 검색 (Hybrid Search) |
| Web Search | 기술 문서, 제조사 매뉴얼, 기술 포럼 | Tavily Search API |
| Academic Search | 학술 논문, 기술 표준 문서 | arXiv API + 논문 벡터 스토어 |

**PdM Agent 연동:**
- Leader Agent가 종합한 결과를 PdM Agent의 `reasoning` 노드에 구조화된 형태로 반환
- 반환 형식: 근거 목록(출처, 신뢰도, 요약) + 종합 판단 + 추가 조사 필요 여부
- 안전장치: Deep Search Engine 내부에도 최대 재순환 횟수 제한 (max_iterations = 3)

### 기대 효과

| **효과** | **설명** |
| --- | --- |
| **단일 RAG 한계 극복** | 내부 데이터만으로 부족한 경우 외부 소스를 동시에 탐색하여 근거 보강 |
| **다각도 근거 확보** | 내부 이력, 웹 문서, 학술 논문 등 복수 소스의 교차 검증으로 분석 근거 강화 |
| **분석 신뢰도 향상** | 특히 Critical 상태에서 충분한 근거 없이 판단하는 위험을 최소화 |


