### Agent 페르소나 및 시스템 프롬프트 (Identity)

Agent의 정체성, 역할, 그리고 답변의 톤앤매너를 정의합니다. 실제 LLM의 System Prompt에 들어갈 핵심 내용입니다.

| **항목**       | **정의 내용** |
| ------------ | --------- |
| **Agent 이름** | PdM Agent |
| **주요 역할**    | Edge 시스템에서 전달받은 이벤트 페이로드(이상감지 결과, 진동 특징량, 추세 데이터, ML RUL 예측값)를 전문가 관점에서 해석하여 결함 유형 식별, 결함 진행 단계 판정, RUL 맥락 평가, 위험도 종합 판단, 정비 권고 생성을 수행하는 베어링 예지보전 전문 AI 에이전트 |
| **핵심 목표**    | 베어링 설비의 이상 이벤트에 대해, 도메인 지식 기반의 전문가 수준 해석과 근거 있는 정비 의사결정을 자동화하여, 비계획 정지를 사전에 방지하고 정비 리드타임을 확보하는 것 |
| **톤앤매너**     | 전문가적 어조로 근거 기반 판단을 제시하며, 불확실성이 있을 경우 그 수준과 원인을 투명하게 고지함. 정상 상태에서는 간결하게, 심각한 이상 상황에서는 상세하고 구조화된 분석을 제공함. 추측이나 일반론이 아닌 도메인 지식에 근거한 판단을 우선함 |
| **제약 사항**    | (1) 수치 계산 금지: RMS 산출, FFT 분석, 통계 연산, 추세선 기울기 계산 등 정량적 연산은 Edge 시스템이 담당하며 에이전트는 산출된 값을 해석만 수행 (2) 추측 금지: 데이터가 부족하거나 판단이 불확실할 때 확신 없는 결론을 생성하지 않으며, 불확실성의 수준과 원인을 명시 (3) 단일 에이전트 제약: 역할별 멀티 에이전트 분리 없이 하나의 에이전트로 모든 분석 수행 |

**시스템 프롬프트 핵심 구성:**

에이전트의 시스템 프롬프트는 다음 요소들로 구성됩니다.

| **구성 요소** | **설명** |
| --------- | ------- |
| 역할 및 원칙 정의 | 에이전트의 정체성, 5개 핵심 원칙 (수치 계산 금지, 도메인 지식 기반 해석, 추론 깊이 자율 조절, 불확실성 투명 고지, 능동적 정보 획득) |
| 도메인 지식 | 베어링 결함 주파수 해석 (BPFO, BPFI, BSF, FTF), P-F 곡선과 결함 진행 4단계, 특징량 복합 해석 패턴 |
| 추론 절차 | 5단계 Thought 구조 (초기 판별 → 단계 판정 → 열화 평가 → RUL 평가 → 종합 판단) 및 분기 기준 |
| Tool 사용 규칙 | 4개 MCP Tool의 호출 조건 및 효율성 원칙 |
| 위험도 기준 | Normal / Watch / Warning / Critical 4단계 정의 및 각 단계별 응답 규칙 |
| 대화형 상호작용 규칙 | 분석 맥락 유지, Prompt Optimization 적용 조건 |

### 워크플로우 및 오케스트레이션 (Workflow & Logic)

이벤트 페이로드 수신부터 최종 정비 권고 및 리포트 생성까지 Agent의 사고 과정과 행동 순서를 기술합니다.

**2.1 처리 로직**

* **Step 1 (Input Analysis — 이벤트 수신 및 맥락 로드):**
    * Edge 시스템에서 이벤트 페이로드(JSON)를 수신
    * 페이로드에서 설비 ID, 베어링 ID를 추출하여 Long-term Memory(PostgreSQL)에서 해당 설비의 이전 분석 판단 이력(최근 5건)을 조회
    * 이전 이력이 있으면 자연어로 요약하여 추론 컨텍스트에 주입 (예: "5일 전 정상 판정, 15일 전 Watch 판정 이력 존재")

* **Step 2 (ReAct Reasoning — 5단계 추론 및 Tool 선택):**
    * Thought 1: `anomaly_detected` 확인. false이면 특징량 교차 확인 후 정상 판정 → 조기 종료. true이면 주파수 영역에서 지배적 결함 주파수 식별 → 결함 유형 판별
    * Thought 2: 시간/주파수 영역 특징량을 종합하여 P-F 곡선 상 결함 진행 단계(1~4단계) 판정. Memory 이전 이력 대비 변화 확인
    * Thought 3: Edge 산출 추세 데이터(slope, trend_direction, acceleration_detected) 해석. 비정상 가속 시 `search_equipment_manual` Tool 호출 가능
    * Thought 4: ML RUL 예측값과 신뢰구간을 에이전트 판정과 대조. 가속 열화 시 신뢰구간 하한(보수적 추정) 채택
    * Thought 5: 위험도 종합 판정(Normal/Watch/Warning/Critical). 필요시 `search_maintenance_history`로 유사 사례 참조. Watch 이상 시 `notify_maintenance_staff` 호출
    * **Tool 선택 기준**: 추론 과정에서 근거 보강이 필요하다고 판단될 때만 호출. 정상 상태에서는 Tool 미호출. 이벤트 분석에서는 일반 RAG 활용(1~2회)으로 충분

    * **Deep Research (분석적 심층 조사):**
        * 일반 RAG 활용(정보 조회)과 구분되는 분석적 조사 행동 패턴
        * 발동 조건: 대화형 상호작용에서 사용자의 분석적 질문 시에만 발동 ("근본 원인 분석해줘", "유사 사례 있어?", "왜 급속 열화인가?")
        * 이벤트 자동 분석에서는 발동하지 않음 (Critical 포함). 이벤트 분석은 도메인 지식 + 일반 RAG로 빠르게 처리
        * 탐색 전략:
            1. 가설 수립: 현재 관찰된 패턴에 기반한 가능한 원인/시나리오 도출
            2. 내부 RAG 탐색: search_maintenance_history, search_equipment_manual, search_analysis_history로 내부 근거 확보
            3. 외부 웹 검색: search_web으로 내부 지식의 부족분 보완 (메커니즘, 최신 사례 등)
            4. 결과 해석 + 후속 질문: 발견 내용을 해석하고, 추가 확인 사항이 있으면 후속 쿼리 생성
            5. 종합 분석: 내부 + 외부 결과를 교차 검증하여 결론 도출. 외부 자료는 "외부 참고 (검증 필요)" 명시
        * 종료: 충분한 근거 확보 시 자율 종료 또는 안전장치 (tool_calls_count > 10)

* **Step 3 (Output Generation — 결과 구조화 및 리포트 생성):**
    * ReAct 루프 완료 후 LLM 응답에서 `diagnosis_result`를 구조화된 JSON으로 파싱 (결함 유형, 단계, 열화 속도, RUL 평가, 위험도, 권고, 불확실성 고지)
    * Warning/Critical 위험도: 분석 리포트(추론 체인 + Memory 이력 + RAG 검색 결과) 및 작업지시서(정비 절차, 필요 자원, 권장 일정, 안전 주의사항) 생성
    * Normal/Watch 위험도: 리포트 생성 생략, 간결한 상태 확인 또는 모니터링 강화 권고 응답
    * 모든 판단 결과를 Long-term Memory(PostgreSQL)에 저장하고, analysis_history VDB에 벡터 적재

**2.2 상태 관리**

LangGraph StateGraph 기반으로 워크플로우 상태를 관리합니다.

* **State 주요 필드:**
    * `event_payload`: Edge에서 수신한 이벤트 데이터
    * `memory_context`: Long-term Memory에서 조회한 이전 판단 이력
    * `messages`: ReAct 추론 과정의 메시지 누적 (Thought-Action-Observation)
    * `diagnosis_result`: 구조화된 진단 결과 (결함 유형, 단계, 위험도, RUL 평가 등)
    * `next_action`: 워크플로우 분기 제어 (continue_reasoning / call_tool / generate_report / end)
    * `tool_calls_count`: Tool 호출 횟수 추적 (안전장치용)

* **Node/Edge 흐름:**

```
START → load_memory → reasoning → (조건부 분기)
                                    ├─ call_tool → tool_executor (MCP) → reasoning
                                    ├─ continue_reasoning → reasoning
                                    └─ generate_report → parse_diagnosis → generate_report
                                                          → generate_work_order → save_memory → END
```

* **안전장치**: `tool_calls_count > 10`이면 강제로 `parse_diagnosis`로 전이하여 무한 루프 방지
* **조건부 실행**: `generate_work_order`는 Warning/Critical 위험도에서만 실행, Normal/Watch에서는 건너뜀

### 도구(Tools) 및 함수 명세 (Capability)

Agent가 외부 데이터 소스 및 알림 시스템과 상호작용하기 위한 도구를 정의합니다. 모든 Tool은 MCP(Model Context Protocol) Server로 구현되며, Agent가 MCP Client로 호출합니다.

| **도구명 (Function Name)** | **기능 설명 (Description)** | **입력 파라미터 (Input Schema)** | **출력 데이터 (Output)** |
| ----------------------- | ----------------------- | -------------------------- | ------------------- |
| search_maintenance_history | 과거 고장/정비 이력을 의미적으로 검색. 유사 결함 사례의 진행 경과, 근본 원인, 고장까지 소요 시간 등을 참조 | query: str, equipment_id?: str, bearing_id?: str, top_k?: int | 유사 정비 이력 문서 리스트 (유사도 순) |
| search_equipment_manual | 설비 매뉴얼, FMEA 문서, 정비 절차서를 검색. 설비 사양, 결함 메커니즘, 급속 열화 조건, 교체 절차 등 참조 | query: str, doc_type?: str, top_k?: int | 관련 매뉴얼 문서 리스트 (유사도 순) |
| search_analysis_history | 에이전트의 과거 분석 판단 이력을 의미적으로 검색. 유사 패턴의 과거 판단과 결과를 참조하여 일관성 유지 | query: str, equipment_id?: str, bearing_id?: str, top_k?: int | 과거 분석 결과 리스트 (유사도 순) |
| notify_maintenance_staff | 정비 담당자에게 분석 결과 및 정비 권고 알림을 전송 | message: str, risk_level: str, equipment_id: str | 전송 성공/실패 상태 |
| search_web | 외부 인터넷에서 베어링/설비 관련 기술 문헌, 논문, 산업 리포트를 검색. Deep Research에서만 사용하며, 내부 RAG 검색을 보완하는 외부 지식 획득용 | query: str | 검색 결과 리스트 (제목, 요약, URL). "외부 참고 자료 (검증 필요)" 태그 포함 |

**Tool 호출 원칙:**
* 정상 상태에서는 Tool을 호출하지 않음 (에이전트 자체 도메인 지식으로 판단 완료)
* **일반 RAG 활용 (이벤트 분석)**: 결함 확인 시 근거 보강이 필요할 때 선택적 호출 (정보 조회 목적, 1~2회)
* **Deep Research (대화형에서만)**: 사용자가 분석적 질문 시 발동. 내부 RAG + 외부 웹 검색의 반복적 탐색-해석-후속질문-재검색 루프
* search_web은 Deep Research에서만 사용. 이벤트 자동 분석에서는 호출하지 않음
* Tool 호출 효율성은 KPI로 평가됨

### 지식 베이스 및 메모리 전략 (Context & Memory)

LLM이 참조할 외부 지식과 대화/분석 이력의 관리 전략을 수립합니다.

**4.1 RAG (검색 증강 생성) 전략**

* **참조 데이터 소스:**

| **Collection** | **문서 수** | **내용** | **용도** |
| -------------- | --------- | ------- | ------- |
| maintenance_history | 80건 (PDF) — 완료보고서 40건 + 작업지시서 40건 | 과거 고장/정비 이력. 설비 정보, 고장 원인, 조치 내용, 자재/공구, ISO 14224 코드 포함 | 유사 결함 사례 비교, 근본 원인 추론 보강 |
| equipment_manual | 7건 (PDF) | 베어링 사양서, 결함 유형별 가이드(외륜/내륜/전동체/보지기), 교체 정비 절차서, FMEA | 결함 메커니즘 확인, 급속 열화 조건 참조, 정비 절차 참조, 위험 우선순위 참조 |
| analysis_history | 동적 증가 | 에이전트 과거 분석 판단 결과 (save_memory 시 PostgreSQL + VDB 동시 적재). reasoning_summary + diagnosis_result 임베딩 | 과거 유사 분석 사례의 의미적 검색. PostgreSQL Memory의 SQL 조회와 상호 보완 |

* **Indexing Pipeline:**

    * **청킹(Chunking) 전략:**

    | **Collection** | **청킹 방식** | **설명** |
    | -------------- | ----------- | ------- |
    | maintenance_history | Document-level (1문서 = 1청크) | 완료보고서·작업지시서 모두 1~2페이지 분량으로, 분할 시 맥락 손실 우려 |
    | equipment_manual | Recursive Chunking (헤더 기반 분할) | 매뉴얼 내 섹션 구조(H2/H3)를 기준으로 재귀 분할. FMEA는 고장 모드별 분할 |
    | analysis_history | Record-level (1레코드 = 1청크) | 에이전트 판단 결과 1건을 하나의 청크로 적재 |

    * **임베딩 전략:** Dense + Sparse 이원 구조

    | **유형** | **모델/방식** | **차원** | **역할** |
    | ------- | ----------- | ------- | ------- |
    | Dense | OpenAI text-embedding-3-small | 1536 | 의미적 유사도 검색 (Semantic Search) |
    | Sparse | BM25 + 한국어 형태소 분석 (kiwipiepy) | — | 키워드 매칭 (설비 코드, ISO 14224 코드, 고유명사 등) |

    * **Metadata 추출:** PDF 파싱 시 아래 필드를 자동 추출하여 벡터와 함께 저장

        * `equipment_id`, `bearing_id` — 설비/베어링 식별자
        * `doc_type` — 문서 유형 (completion_report, work_order, manual, fmea 등)
        * `fault_type`, `fault_location` — 결함 유형 및 위치 (해당 시)
        * `date` — 문서 작성일

* **Retrieval Pipeline:**

    | **단계** | **방법** | **설명** |
    | ------- | ------- | ------- |
    | 1단계 — Metadata Filtering | Qdrant Payload Filter | `equipment_id`, `bearing_id`, `doc_type` 등으로 후보 문서 사전 필터링 |
    | 2단계 — Hybrid Search | Dense + Sparse → RRF 결합 | Dense 유사도와 Sparse BM25 점수를 Reciprocal Rank Fusion으로 결합하여 최종 스코어 산출 |
    | 3단계 — Reranker | Cross-encoder (모델 선정중) | Top-k 후보를 Cross-encoder로 재정렬하여 정밀도 향상 |
    | 결과 | Top-k 반환 (기본 k=3) | 최종 상위 k건을 Agent 컨텍스트에 주입 |

* **Vector DB:** Qdrant (Docker 자체 호스팅, Dense + Sparse 듀얼 인덱스 지원)

**4.2 대화 메모리 (Conversation History)**

* **메모리 유형:** Short-term Memory + Long-term Memory 이원 구조

| **구분** | **저장소** | **내용** | **용도** |
| ------- | -------- | ------- | ------- |
| Short-term | LangGraph State (`messages`) | 현재 분석 세션의 ReAct 추론 과정 (Thought-Action-Observation 체인) | 단일 이벤트 분석 내 맥락 유지 |
| Long-term | PostgreSQL (`pdm_agent_memory` 테이블) | 모든 이벤트(정상 포함)에 대한 판단 결과. 결함 유형, 단계, 위험도, RUL 평가, 권고, Tool 사용 기록, 후속 조치 추적 | 동일 설비의 시계열 변화 추적, "언제부터 이상 시작" 파악, 후속 분석 시 이전 맥락 참조 |

* **저장 전략:**
    * Short-term: 이벤트 분석 완료 시 소멸 (State 수명 = 단일 워크플로우 실행)
    * Long-term: 모든 판단 결과를 영구 저장. 조회 시 설비/베어링 ID 기준 최근 5건 로드
    * 정상 판정도 저장하여, 이상 시작 시점 추적에 활용

* **Prompt Optimization (대화형 상호작용 시):**

| **전략** | **설명** | **적용 시점** |
| ------- | ------- | ---------- |
| 분석 맥락 구조화 | Memory에서 전체 추론 체인 대신, 핵심 필드(결함 유형, 단계, 위험도, RUL, 핵심 근거)만 추출하여 컨텍스트에 주입 | 대화 세션 시작 시 |
| 대화 이력 압축 | 이전 턴 질문-응답 쌍을 핵심 결론 중심으로 요약 | 매 턴 누적 시 |
| 슬라이딩 윈도우 | 최근 2~3턴은 원문 유지, 이전 턴은 요약으로 대체하여 컨텍스트 윈도우 효율적 활용 | 대화 3턴 이상 시 |

### 핵심 에이전트 기술 스택

에이전트의 추론 품질과 검색 정확도를 제어하기 위한 핵심 기술적 의사결정입니다.

**① 추론 전략**

| **기술** | **구현** | **선정 사유** |
| ------- | ------- | ---------- |
| ReAct 추론 | 5단계 Thought 구조 (결함 식별 → 단계 판정 → 열화 평가 → RUL 평가 → 종합 판단) | Thought-Action-Observation 교차 수행으로 단계적 심화 추론 구현. 각 단계에서 분기/조기종료/Tool 호출을 자율 결정하여, 정상 상태는 Thought 1 조기 종료, 이상 상태는 전체 추론 수행 |
| Deep Research | 내부 RAG + 외부 웹 검색(search_web) 기반 분석적 조사 루프 | 가설 수립 → 내부 RAG 탐색 → 외부 웹 검색 → 결과 해석 → 후속 질문 → 재검색의 반복적 조사. 대화형 상호작용에서 사용자 요청 시에만 발동. 단일 에이전트 내에서 조사 깊이를 극대화하며, 외부 자료는 "외부 참고 (검증 필요)" 태그로 소스 신뢰도를 구분 |

**② RAG 검색 파이프라인**

| **기술** | **구현** | **선정 사유** |
| ------- | ------- | ---------- |
| Hybrid Search | Dense (OpenAI text-embedding-3-small, 1536dim) + Sparse (BM25 + kiwipiepy) → RRF (Reciprocal Rank Fusion) 결합 | 의미적 유사도(Dense)와 키워드 매칭(Sparse)을 RRF로 결합. 한국어 형태소 분석(kiwipiepy)으로 설비 코드, ISO 14224 코드 등 도메인 고유명사 검색 정밀도 확보 |
| Metadata Filtering | Qdrant Payload Filter (equipment_id, bearing_id, doc_type) | 검색 전 후보 문서를 설비/베어링/문서유형 기준으로 사전 필터링하여 검색 범위 축소. 불필요한 문서 노이즈 제거 및 정밀도 향상 |
| Reranker | Cross-encoder (모델 선정 중) | Hybrid Search Top-k 후보를 query-document 쌍 단위로 정밀 재정렬. 1차 검색의 recall과 Reranker의 precision을 결합하여 최종 검색 품질 극대화 |
| Embedding 이원 구조 | Dense: OpenAI text-embedding-3-small (1536dim) / Sparse: BM25 + kiwipiepy 형태소 분석 | Dense로 의미적 유사도를, Sparse로 키워드 정확 매칭을 분담. Qdrant의 Dense + Sparse 듀얼 인덱스로 단일 Collection 내에서 두 방식 동시 지원 |

**③ Tool 호출 아키텍처 (TBD)**

> Tool 호출 프로토콜은 MCP를 기준으로 설계하였으나, 최종 확정 전입니다.

| **기술** | **구현** | **선정 사유** |
| ------- | ------- | ---------- |
| Tool 프로토콜 | MCP (Model Context Protocol) — TBD | LangGraph Agent가 MCP Client, Tool 서버가 MCP Server로 동작. 표준 프로토콜 기반 Tool 호출로 서버 독립 배포 및 확장 가능 |
| MCP Server 구성 | rag_server (RAG 검색 3종) · web_search_server (외부 검색) · notification_server (알림) | 기능별 서버 분리로 독립 배포 가능. search_web은 Deep Research 전용, notify_maintenance_staff는 Watch 이상 시 호출 |
