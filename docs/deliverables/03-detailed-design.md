# 03. 상세 설계

> PdM Agent — 예지보전 AI 에이전트

## Agent 페르소나 및 시스템 프롬프트 (Identity)

| **항목** | **정의 내용** |
| --- | --- |
| **Agent 이름** | PdM Agent |
| **주요 역할** | Edge 시스템에서 전달받은 이벤트 페이로드(이상감지 결과, 진동 특징량, 추세 데이터, ML RUL 예측값)를 전문가 관점에서 해석하여 결함 유형 식별, 결함 진행 단계 판정, RUL 맥락 평가, 위험도 종합 판단, 정비 권고 생성을 수행하는 베어링 예지보전 전문 AI 에이전트 |
| **핵심 목표** | 베어링 설비의 이상 이벤트에 대해, 도메인 지식 기반의 전문가 수준 해석과 근거 있는 정비 의사결정을 자동화하여, 비계획 정지를 사전에 방지하고 정비 리드타임을 확보하는 것 |
| **톤앤매너** | 전문가적 어조로 근거 기반 판단을 제시하며, 불확실성이 있을 경우 그 수준과 원인을 투명하게 고지함. 정상 상태에서는 간결하게, 심각한 이상 상황에서는 상세하고 구조화된 분석을 제공함. 추측이나 일반론이 아닌 도메인 지식에 근거한 판단을 우선함 |
| **제약 사항** | (1) 수치 계산 금지: RMS 산출, FFT 분석, 통계 연산, 추세선 기울기 계산 등 정량적 연산은 Edge 시스템이 담당하며 에이전트는 산출된 값을 해석만 수행 (2) 추측 금지: 데이터가 부족하거나 판단이 불확실할 때 확신 없는 결론을 생성하지 않으며, 불확실성의 수준과 원인을 명시 (3) 단일 에이전트 제약: 역할별 멀티 에이전트 분리 없이 하나의 에이전트로 모든 분석 수행 |

**시스템 프롬프트 핵심 구성:**

| **구성 요소** | **위치** | **설명** |
| --- | --- | --- |
| 역할 및 원칙 정의 | 시스템 프롬프트 | 에이전트의 정체성, 5개 핵심 원칙 (수치 계산 금지, 도메인 지식 기반 해석, 추론 깊이 자율 조절, 불확실성 투명 고지, 능동적 정보 획득) |
| 추론 절차 | 시스템 프롬프트 | 5단계 Thought 구조 (초기 판별 → 단계 판정 → 열화 평가 → RUL 평가 → 종합 판단) 및 분기 기준 |
| Action Skills 사용 규칙 | 시스템 프롬프트 | 5개 Action Skills의 호출 조건 및 효율성 원칙 |
| 도메인 지식 | Core Skills | fault-diagnosis, feature-interpret 등 SKILL.md로 분리. 추론 과정에서 필요 시에만 조건부 로딩 |
| 위험도 기준 및 응답 양식 | Core Skills | response-normal · response-alert Skill로 분리. 위험도 판정 후 로드 |
| Deep Research 절차 | Core Skills | deep-research Skill로 분리. 대화형 상호작용에서 사용자 요청 시에만 로드 |
| 대화형 상호작용 규칙 | 시스템 프롬프트 | 분석 맥락 유지, Prompt Optimization 적용 조건 |

## 워크플로우 및 오케스트레이션 (Workflow & Logic)

### 처리 로직

- **Step 1 (Input Analysis — 이벤트 수신 및 맥락 로드):**
    - Edge 시스템에서 이벤트 페이로드(JSON)를 수신
    - 페이로드에서 설비 ID, 베어링 ID를 추출하여 Long-term Memory(PostgreSQL)에서 해당 설비의 이전 분석 판단 이력(최근 5건)을 조회
    - 이전 이력이 있으면 자연어로 요약하여 추론 컨텍스트에 주입

- **Step 2 (ReAct Reasoning — 5단계 추론 및 Skills 선택):**
    - Thought 1: `anomaly_detected` 확인. false이면 정상 판정 → 조기 종료. true이면 결함 주파수 식별 → 결함 유형 판별
    - Thought 2: 시간/주파수 영역 특징량을 종합하여 P-F 곡선 상 결함 진행 단계(1~4단계) 판정
    - Thought 3: Edge 산출 추세 데이터 해석. 비정상 가속 시 Skills 호출로 근거 보강
    - Thought 4: ML RUL 예측값과 신뢰구간을 에이전트 판정과 대조
    - Thought 5: 위험도 종합 판정(Normal/Watch/Warning/Critical). Watch 이상 시 알림 호출
    - **Skills 선택 기준**: 추론 과정에서 근거 보강이 필요할 때만 호출. 정상 상태에서는 미호출
    - **Deep Search**: 트리거 조건 충족 시 Deep Search Engine 호출 (상세는 Deep Search Engine 섹션 참조)

- **Step 3 (Output Generation — 결과 구조화 및 리포트 생성):**
    - ReAct 루프 완료 후 `diagnosis_result`를 구조화된 JSON으로 파싱 (결함 유형, 단계, 열화 속도, RUL 평가, 위험도, 권고, 불확실성 고지)
    - Warning/Critical 위험도: 분석 리포트 및 작업지시서 생성
    - Normal/Watch 위험도: 간결한 상태 확인 또는 모니터링 강화 권고 응답
    - 모든 판단 결과를 Long-term Memory(PostgreSQL)에 저장하고, analysis_history VDB에 벡터 적재

### 상태 관리

LangGraph StateGraph 기반으로 워크플로우 상태를 관리합니다.

- **State 주요 필드:**
    - `event_payload`: Edge에서 수신한 이벤트 데이터
    - `messages`: ReAct 추론 과정의 메시지 누적 (Thought-Action-Observation)
    - `diagnosis_result`: 구조화된 진단 결과 (결함 유형, 단계, 위험도, RUL 평가 등)
    - `next_action`: 워크플로우 분기 제어 (continue_reasoning / call_tool / generate_report / end)

- **안전장치**: Skills 호출 횟수가 10회 초과 시 강제로 결과 파싱 단계로 전이하여 무한 루프 방지
- **조건부 실행**: 작업지시서 생성은 Warning/Critical 위험도에서만 실행, Normal/Watch에서는 건너뜀

## 도구(Tools) 및 지식 관리 명세 (Capability)

Agent의 능력은 **Action Skills(실행형 도구)**와 **Core Skills(도메인 지식)**의 이원 구조로 구성됩니다.

- **Action Skills**: 외부 데이터 소스 및 알림 시스템과의 실제 상호작용을 담당
- **Core Skills**: 도메인 지식을 SKILL.md 파일로 모듈화하여, 조건부 로딩으로 필요 시에만 점진적으로 로드

**Action Skills 명세:**

| **도구명 (Function Name)** | **기능 설명 (Description)** | **입력 파라미터 (Input Schema)** | **출력 데이터 (Output)** |
| --- | --- | --- | --- |
| search_maintenance_history | 과거 고장/정비 이력을 의미적으로 검색. 유사 결함 사례의 진행 경과, 근본 원인, 고장까지 소요 시간 등을 참조 | query: str, equipment_id?: str, bearing_id?: str, top_k?: int | 유사 정비 이력 문서 리스트 (유사도 순) |
| search_equipment_manual | 설비 매뉴얼, FMEA 문서, 정비 절차서를 검색. 설비 사양, 결함 메커니즘, 급속 열화 조건, 교체 절차 등 참조 | query: str, doc_type?: str, top_k?: int | 관련 매뉴얼 문서 리스트 (유사도 순) |
| search_analysis_history | 에이전트의 과거 분석 판단 이력을 의미적으로 검색. 유사 패턴의 과거 판단과 결과를 참조하여 일관성 유지 | query: str, equipment_id?: str, bearing_id?: str, top_k?: int | 과거 분석 결과 리스트 (유사도 순) |
| notify_maintenance_staff | 정비 담당자에게 분석 결과 및 정비 권고 알림을 전송 | message: str, risk_level: str, equipment_id: str | 전송 성공/실패 상태 |
| search_web | 외부 인터넷에서 베어링/설비 관련 기술 문헌, 논문, 산업 리포트를 검색. Deep Research에서만 사용하며, 내부 RAG 검색을 보완하는 외부 지식 획득용 | query: str | 검색 결과 리스트 (제목, 요약, URL). "외부 참고 자료 (검증 필요)" 태그 포함 |

**Core Skills 명세:**

| **Skill 이름** | **로드 조건** | **내용** |
| --- | --- | --- |
| fault-diagnosis | 이상 감지 시 자동 로드 | 베어링 결함 주파수 해석 (BPFO, BPFI, BSF, FTF), P-F 곡선 4단계 정의, 고조파/사이드밴드 해석 기준 |
| feature-interpret | 이상 감지 시 자동 로드 | Kurtosis + RMS 복합 패턴, Crest Factor 전이 패턴, 고주파 에너지 초기 지표 등 특징량 해석 규칙 |
| deep-research | Deep Search 발동 시 로드 | 가설 수립 → 내부 RAG → 외부 검색 → 해석 → 재검색의 탐색 절차, 외부 자료 신뢰도 태깅 규칙 |
| response-normal | 정상/관찰 판정 시 로드 | Normal/Watch 위험도별 응답 양식, 간결 요약 구조 |
| response-alert | 경고/위험 판정 시 로드 | Warning/Critical 위험도별 응답 양식, 리포트 구조, 작업지시서 포함 기준 |

**User Skills (Personalized Skills) 명세:**

| **Skill 유형** | **저장 경로** | **매칭 조건** | **설명** |
| --- | --- | --- | --- |
| 리포트 형식 선호 | `skills/users/{id}/preferred-report-format.md` | `chat` | 사용자가 선호하는 리포트 구조/상세 수준 |
| 설비 메모 | `skills/users/{id}/equipment-notes.md` | `always` | 담당 설비 특이사항, 과거 경험 메모 |
| 분석 관점 | `skills/users/{id}/analysis-focus.md` | `anomaly_detected` | 사용자가 중시하는 분석 항목 (예: 윤활 상태 우선) |

- YAML frontmatter 기반 Skill 파일, 컨텍스트별 매칭 (`always`, `chat`, `agent`, `anomaly_detected`, `alert`)
- CRUD API: `save_user_skill()`, `delete_user_skill()`, `list_user_skills()`
- 자동 생성: `SkillEvolver`가 대화 패턴에서 사용자 선호를 감지하여 자동 배치

**Skills 조건부 로딩 예시:**
- **정상 이벤트**: 시스템 프롬프트(페르소나 + 추론 구조) → Thought 1 조기 종료 → Core Skills 미로드 (토큰 절감)
- **이상 이벤트**: 시스템 프롬프트 → fault-diagnosis 로드 → feature-interpret 로드 → response-normal 또는 response-alert 로드
- **대화형 Deep Research**: 위 + deep-research 로드 → Deep Search Engine 조건부 호출
- **Self-Evolving**: save_memory 완료 → SkillEvolver가 분석 결과 diff 감지 → Core Skills 자동 패치 (비동기)
- **대화형 개인화**: 대화 완료 → 사용자 선호 패턴 감지 → User Skills 자동 생성 → 다음 세션부터 자동 로드

## 지식 베이스 및 메모리 전략 (Context & Memory)

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
| Skills 조건부 로딩 | 도메인 지식을 Core Skills로 분리. 세션 시작 시 메타데이터(~300토큰)만 로드하고, 추론 시 필요한 Skill만 조건부 로딩(~5K토큰/Skill). 정상 이벤트에서는 결함 진단 Skill 미로드 | 매 턴 |
| 분석 맥락 구조화 | Memory에서 전체 추론 체인 대신, 핵심 필드(결함 유형, 단계, 위험도, RUL, 핵심 근거)만 추출하여 컨텍스트에 주입 | 대화 세션 시작 시 |
| 대화 이력 압축 | 이전 턴 질문-응답 쌍을 핵심 결론 중심으로 요약 | 매 턴 누적 시 |
| 슬라이딩 윈도우 | 최근 2~3턴은 원문 유지, 이전 턴은 요약으로 대체하여 컨텍스트 윈도우 효율적 활용 | 대화 3턴 이상 시 |

## 핵심 에이전트 기술 스택

### 추론 전략

| **기술** | **구현** | **선정 사유** |
| --- | --- | --- |
| ReAct 추론 | 5단계 Thought 구조 (결함 식별 → 단계 판정 → 열화 평가 → RUL 평가 → 종합 판단) | Thought-Action-Observation 교차 수행으로 단계적 심화 추론 구현. 각 단계에서 분기/조기종료/Tool 호출을 자율 결정하여, 정상 상태는 Thought 1 조기 종료, 이상 상태는 전체 추론 수행 |
| Deep Search Engine (STORM) | STORM(Survey of Tailored Research on Open-domain Modeling) 기법 기반 다관점 분석 엔진을 조건부 호출하여 Leader-Research-Critic-Synthesize 구조로 병렬 탐색 | 트리거 조건 충족 시 Leader가 질의를 분해하고, 3개 고정 Research Agent가 각각 독립적으로 검색을 수행. Critic이 각 결과를 Pass/Revise 검증하고, Review-Revise 루프(최대 3회)를 거쳐 confidence score 기반 가중치 투표로 합성 |

### RAG 검색 파이프라인

| **기술** | **구현** | **선정 사유** |
| --- | --- | --- |
| Hybrid Search | Dense (OpenAI text-embedding-3-small, 1536dim) + Sparse (BM25 + kiwipiepy) → RRF (Reciprocal Rank Fusion) 결합 | 의미적 유사도(Dense)와 키워드 매칭(Sparse)을 RRF로 결합. 한국어 형태소 분석(kiwipiepy)으로 설비 코드, ISO 14224 코드 등 도메인 고유명사 검색 정밀도 확보 |
| Metadata Filtering | Qdrant Payload Filter (equipment_id, bearing_id, doc_type) | 검색 전 후보 문서를 설비/베어링/문서유형 기준으로 사전 필터링하여 검색 범위 축소. 불필요한 문서 노이즈 제거 및 정밀도 향상 |
| Reranker | Cross-encoder (모델 선정 중) | Hybrid Search Top-k 후보를 query-document 쌍 단위로 정밀 재정렬. 1차 검색의 recall과 Reranker의 precision을 결합하여 최종 검색 품질 극대화 |
| Embedding 이원 구조 | Dense: OpenAI text-embedding-3-small (1536dim) / Sparse: BM25 + kiwipiepy 형태소 분석 | Dense로 의미적 유사도를, Sparse로 키워드 정확 매칭을 분담. Qdrant의 Dense + Sparse 듀얼 인덱스로 단일 Collection 내에서 두 방식 동시 지원 |

### Skills 호출 아키텍처

| **기술** | **구현** | **선정 사유** |
| --- | --- | --- |
| Action Skills 구성 | RAG 검색 3종 (maintenance_history · equipment_manual · analysis_history) · web_search (외부 검색) · notification (알림) | 기능별 분리로 독립 관리 가능. search_web은 Deep Research 전용, notify_maintenance_staff는 Watch 이상 시 호출 |

### 도메인 지식 관리 (Skills)

> Agent Skills 오픈 표준([agentskills.io](https://agentskills.io))을 채택하여, 시스템 프롬프트에 고정되던 도메인 지식을 모듈화합니다.

| **기술** | **구현** | **선정 사유** |
| --- | --- | --- |
| Core Skills | SKILL.md 기반 도메인 지식 모듈화. 조건부 로딩(Progressive Disclosure) 적용 | 세션 시작 시 메타데이터(Skill 이름 + 설명)만 로드하고, 추론 과정에서 필요한 Skill만 조건부 로딩. 정상 이벤트(Thought 1 조기 종료)에서 불필요한 도메인 지식 로딩을 제거하여 토큰 효율성 확보 |
| Skill 구성 | fault-diagnosis (결함 주파수 + P-F 곡선) · feature-interpret (특징량 복합 해석) · deep-research (심화 조사 절차) · response-normal (Normal/Watch 응답 양식) · response-alert (Warning/Critical 응답 양식) | 추론 단계별로 필요한 지식만 선택 로드. 새 설비 유형(모터, 펌프 등) 확장 시 Skill 파일 추가만으로 대응 가능 |
| Action Skills-Core Skills 역할 분리 | Core Skills = 도메인 지식(뇌), Action Skills = 외부 실행(근육) | Core Skills는 에이전트의 추론 품질을 제어하는 지식/지침, Action Skills는 외부 데이터 소스와의 실제 상호작용을 담당. 관심사 분리로 각 레이어의 독립적 확장 가능 |

## Self-Evolving Skills Engine

분석 결과와 실제 고장 결과를 대조하여 Core Skills를 자동 보정하는 자기 진화 메커니즘입니다.

### Learning Loop (Core Skills 자동 보정)

| **단계** | **동작** | **대상 컴포넌트** |
| --- | --- | --- |
| 1. 분석 완료 | Agent가 진단 결과를 Memory에 저장 | `save_memory` 노드 |
| 2. Trigger 발생 | 저장 시점에 `SkillEvolver` 자동 호출 (비동기) | `SkillEvolver` |
| 3. Diff 감지 | 이전 분석 예측 vs 실제 결과(후속 상태) 대조 | analysis_history + PostgreSQL Memory |
| 4. Skill 패치 생성 | 예측-실제 간 gap 분석 → 해석 규칙 수정안 LLM 생성 | Core Skills (md) |
| 5. Core Skills 업데이트 | 검증 후 `skills/core/` 디렉토리에 반영 | `skills/core/*.md` |

### User Feedback Loop

대화 완료 시 사용자의 반복 요청 패턴(분석 관점, 리포트 형식, 설비 조건 등)을 감지하여 `skills/users/{user_id}/` 경로에 User Skills를 자동 생성. 다음 세션부터 자동 로드되어 개인화된 분석 품질을 제공합니다.

### Skills Store 구조

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

### 기대 효과

| **효과** | **설명** |
| --- | --- |
| **도메인 지식 자동 축적** | 반복 분석을 통해 해석 규칙이 지속적으로 정교화. 분석 건수 증가에 비례하여 진단 정확도 향상 |
| **조직 노하우의 명시적 자산화** | 암묵지(경험 기반 판단)가 Skill 파일로 코드화되어 인력 이동에도 지식 유실 방지 |
| **개인화된 분석 품질** | User Skills로 담당자별 맥락을 반영하여 개인화된 분석 경험 제공 |

---

## Deep Search Engine (STORM 스타일 다관점 분석)

복잡한 분석 요청 시 **STORM 스타일 다관점 분석**을 수행하는 Group Agent 아키텍처입니다. PdM Agent 본체는 단일 ReAct 에이전트를 유지하면서, 고난도 분석이 필요한 경우에만 Deep Search Engine을 조건부 호출합니다. Leader가 질의를 분해하고, 3개 고정 Research Agent가 각자의 전문 관점에서 병렬 검색을 수행하며, Critic의 Review-Revise 루프를 거쳐 confidence score 기반 가중치 투표로 최종 합성합니다.

### Leader-Research-Critic-Synthesize 구조

| **컴포넌트** | **역할** | **구현** |
| --- | --- | --- |
| **Leader** | 사용자 질의를 분해(Decompose)하여 3개 Research Agent에 검색 지시 배포 | LangGraph 서브그래프 `decompose` 노드 |
| **Research Agent 1: Maintenance Engineer** | 정비 이력 관점에서 유사 고장 사례, 정비 경과, 근본 원인 검색 | agent_role: `maintenance_history`, tools: `search_maintenance_history` |
| **Research Agent 2: Senior Analyst** | 분석 이력 관점에서 과거 유사 패턴의 에이전트 판단 및 결과 검색 | agent_role: `analysis_history`, tools: `search_analysis_history` |
| **Research Agent 3: Equipment Specialist** | 설비 매뉴얼 + 외부 기술 자료 관점에서 결함 메커니즘, 급속 열화 조건, 정비 절차 검색 | agent_role: `equipment_manual`, tools: `search_equipment_manual` + `search_web` |
| **Critic** | 각 Research Agent 결과의 관련성, 충분성, 정확성을 검증하여 Pass/Revise 판정 | LangGraph `review` 노드 |
| **Synthesizer** | Critic 검증을 통과한 결과를 confidence score 기반 가중치 투표로 합성 | LangGraph `synthesize` 노드 |

### 실행 흐름

PdM Agent의 추론 과정에서 트리거 조건이 충족되면 Deep Search Engine을 호출합니다. Leader가 질의를 분해(Decompose)하고, 3개 Research Agent가 병렬로 검색을 수행합니다. Critic이 각 결과를 Pass/Revise 판정하며, Revise 시 해당 perspective만 재검색합니다 (최대 3회). 모든 결과가 Pass되면 confidence score 기반 가중치 투표로 합성(Synthesize)하여 PdM Agent에 구조화된 근거 세트를 반환합니다.

### 3개 고정 Research Agent (Perspective)

| **Research Agent** | **전문 관점** | **검색 도구** | **주요 검색 대상** |
| --- | --- | --- | --- |
| **Maintenance Engineer** | 정비 이력 전문가 | `search_maintenance_history` | 유사 고장 사례, 정비 경과, 근본 원인, 고장까지 소요 시간 |
| **Senior Analyst** | 분석 이력 전문가 | `search_analysis_history` | 과거 유사 패턴의 에이전트 판단, 예측 정확도, 결과 추적 |
| **Equipment Specialist** | 설비/기술 전문가 | `search_equipment_manual` + `search_web` | 결함 메커니즘, 급속 열화 조건, 정비 절차, 외부 기술 문헌 |

### Review-Revise 루프 및 Voting 메커니즘

**Critic 검증 프로세스:**
1. 각 Research Agent의 결과를 독립적으로 평가 (관련성, 충분성, 정확성)
2. Pass 판정: 해당 perspective의 결과가 질의에 대한 충분한 근거를 제공
3. Revise 판정: 결과가 불충분하거나 질의와의 관련성이 낮음 → 해당 perspective만 재검색 지시
4. 최대 3회 재검색 후에도 Revise이면 현재 결과로 진행 (안전장치)

**Confidence Score 산출:**
- 검색 결과 건수, 유사도 점수, 문서 관련성을 종합한 휴리스틱 (0.0~1.0)
- 높음 (≥0.7): 충분한 근거 확보, 합성 시 높은 가중치
- 중간 (0.4~0.7): 부분적 근거, 보완적 참조
- 낮음 (<0.4): 근거 부족, 합성 시 낮은 가중치 또는 "근거 부족" 명시

**가중치 투표(Weighted Voting):**
- 합성(Synthesize) 단계에서 각 전문가의 confidence score에 비례하여 최종 결과에 대한 기여도 결정
- 예: Maintenance Engineer(0.85) + Senior Analyst(0.45) + Equipment Specialist(0.72) → 정비 이력 관점 우세 반영

### 트리거 조건

| **조건** | **설명** | **판단 기준** |
| --- | --- | --- |
| **RAG 검색 결과 불충분** | 내부 검색으로 충분한 근거 확보 실패 | confidence score < threshold (예: 0.7) |
| **사용자 명시적 요청** | "자세히 분석해줘", "근거를 더 찾아줘" 등 | 프롬프트 내 deep search 키워드 감지 |
| **Critical + 유사 사례 부족** | health_state가 critical이면서 참고할 유사 사례가 희소 | health_state == "critical" AND 유사 사례 < 2건 |

### 기대 효과

| **효과** | **설명** |
| --- | --- |
| **단일 RAG 한계 극복** | 3개 전문가 관점에서 동시에 탐색하여 단일 검색으로 놓칠 수 있는 근거 확보 |
| **다관점 교차 검증** | 정비 이력, 분석 이력, 설비 매뉴얼/외부 기술 자료의 다관점 교차 검증으로 분석 근거 강화 |
| **품질 보장** | Critic의 Review-Revise 루프로 검색 결과 품질을 보장하고, confidence score로 신뢰도를 정량화 |
| **병렬 실행 효율** | `asyncio.gather()` 병렬 실행으로 순차 검색 대비 응답 시간 단축 |


