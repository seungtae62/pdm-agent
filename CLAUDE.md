# PdM Agent - 예지보전 AI 에이전트 프로젝트

## 프로젝트 개요

제조업 베어링 설비의 예지보전(Predictive Maintenance)을 위한 AI 에이전트 시스템. Edge에서 진동 센서 기반 이상감지를 수행하고, Cloud의 PdM Agent가 이상 이벤트를 해석하여 결함 진단, RUL 평가, 정비 권고를 생성한다.

데이터셋: NASA-IMS Bearing Dataset (4개 베어링, 2000 RPM, 6000 lbs, Test Set 1~3)

---

## 기술 스택

- LLM: GPT-4o (OpenAI API)
- 에이전트 프레임워크: LangGraph (Python)
- Embedding: OpenAI text-embedding-3-small (1536차원)
- Vector DB: Qdrant (Docker 자체 호스팅)
- Memory / Checkpointer: PostgreSQL (langgraph-checkpoint-postgres)
- 인프라: On-Premise Cloud (내부망), Docker 기반 배포
- PDF 파싱: PyMuPDF 또는 pdfplumber (텍스트 기반 PDF, OCR 불필요)

---

## 에이전트 아키텍처

### 핵심 원칙

1. 단일 에이전트(PdM Agent)로 구성한다. 역할별 멀티 에이전트 분리는 하지 않는다.
2. 에이전트는 수치 계산을 하지 않는다. Edge가 산출한 정량값을 해석하고 의미를 부여한다.
3. 에이전트는 내장 도메인 지식을 기반으로 전문가 수준의 해석을 수행한다.
4. 상황에 따라 추론의 깊이와 경로를 스스로 결정한다.
5. Function Call을 최소화하고 에이전트의 추론 능력을 극대화한다.

### 기술 구성요소

- ReAct: 기본 추론 프레임워크. Thought-Action-Observation 루프.
- RAG: ReAct 내 Action으로 선택적 호출. 정비 이력 검색, 설비 매뉴얼 검색.
- Memory: Short-term(LangGraph State) + Long-term(PostgreSQL).
- Deep Research: ReAct 내 다회 RAG 탐색으로 구현.

### 핵심 기능

1. 결함 유형 식별
2. 결함 진행 단계 판정 및 열화 속도 평가
3. RUL 예측 평가
4. 종합 판단 및 정비 권고 생성
5. 분석 리포트 생성

### Tool (3개)

1. **search_maintenance_history**: 과거 고장/정비 이력 검색 (Qdrant RAG)
2. **search_equipment_manual**: 설비 매뉴얼/FMEA/절차서 검색 (Qdrant RAG)
3. **notify_maintenance_staff**: 정비 담당자 알림 전송

---

## 구현 순서

### Phase 1: Edge 파이프라인
1-1. NASA-IMS 데이터 전처리 (특징량 추출)
1-2. 이상감지 모델 구현 (통계 기반 또는 간단한 Autoencoder)
1-3. ML RUL 예측 모델 구현
1-4. 이벤트 페이로드 생성기 구현

### Phase 2: RAG 구축
2-1. 합성 PDF 문서 19건 생성 (템플릿 기반)
2-2. Qdrant Collection 구성 및 문서 적재
2-3. 검색 품질 검증

### Phase 3: PdM Agent 핵심 구현
3-1. LangGraph StateGraph 구현 (기본 골격)
3-2. 시스템 프롬프트 탑재 및 기본 추론 검증 (Tool 없이)
3-3. RAG Tool 연동
3-4. Memory 구현 (PostgreSQL)
3-5. Deep Research 로직 (다회 RAG 탐색 분기)
3-6. 알림 Tool 구현

### Phase 4: 시나리오 시뮬레이션 및 KPI 측정
4-1. 시나리오 1~5 시뮬레이션 실행
4-2. KPI 측정
4-3. 결과 분석 및 프롬프트 튜닝
4-4. 분석 리포트 생성 기능 검증

### 향후 확장
- Critic Agent, Human-in-the-Loop, Structured Output, Self-Correction

---

## 상세 설계 문서

- **StateGraph / RAG / Memory / 페이로드 / 시나리오 / KPI**: [`docs/design.md`](docs/design.md)
- **시스템 프롬프트 전문**: [`docs/system-prompt.md`](docs/system-prompt.md)

---

## 프로젝트 구조

> **참고**: 각 Phase의 디렉토리(`edge/`, `rag/`, `simulation/`, `tests/`)는 해당 Phase 작업 시 생성한다.

```
pdm-agent/
├── CLAUDE.md                          # 이 파일
├── .env.example                       # 환경변수 템플릿
├── .gitignore
├── requirements.txt
├── docs/
│   ├── design.md                      # 상세 설계 (StateGraph, RAG, Memory, 페이로드, 시나리오, KPI)
│   └── system-prompt.md               # 시스템 프롬프트 전문
├── scripts/
│   └── download_ims_dataset.py        # NASA-IMS 데이터셋 다운로드
├── data/
│   └── ims/                           # NASA-IMS 데이터셋 저장 위치
└── agent/                             # Phase 3: PdM Agent
    ├── __init__.py
    ├── graph.py                       # LangGraph StateGraph 정의
    ├── state.py                       # State 스키마
    ├── nodes/                         # 노드 구현
    │   ├── __init__.py
    │   ├── load_memory.py
    │   ├── reasoning.py
    │   ├── tool_executor.py
    │   ├── parse_diagnosis.py
    │   ├── generate_report.py
    │   └── save_memory.py
    ├── tools/                         # Tool 구현
    │   ├── __init__.py
    │   ├── search_maintenance_history.py
    │   ├── search_equipment_manual.py
    │   └── notify_maintenance_staff.py
    ├── prompts/                       # 시스템 프롬프트
    │   ├── __init__.py
    │   └── system_prompt.py
    └── memory/                        # Memory 관련
        ├── __init__.py
        ├── schema.sql                 # PostgreSQL 테이블 DDL
        └── memory_manager.py          # 조회/저장 로직
```
