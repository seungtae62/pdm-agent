# PdM Agent - 상세 설계 문서

## 구현 순서

### Phase 1: Edge 파이프라인
1-1. NASA-IMS 데이터 전처리 (특징량 추출)
1-2. 이상감지 모델 구현 (통계 기반 또는 간단한 Autoencoder)
1-3. ML RUL 예측 모델 구현
1-4. 이벤트 페이로드 생성기 구현

### Phase 2: RAG 구축
2-1. 합성 PDF 문서 19건 생성 (템플릿 기반)
2-2. Qdrant Collection 3개 구성 (maintenance_history, equipment_manual, analysis_history) 및 문서 적재
2-3. MCP Server 구현 (RAG Tool 호출용)
2-4. 검색 품질 검증

### Phase 3: PdM Agent 핵심 구현
3-1. LangGraph StateGraph 구현 (기본 골격)
3-2. 시스템 프롬프트 탑재 및 기본 추론 검증 (Tool 없이)
3-3. MCP Client 연동 (RAG Tool + 알림 Tool)
3-4. Memory 구현 (PostgreSQL)
3-5. Deep Research 로직 (다회 RAG 탐색 분기)
3-6. 작업지시서 생성 기능 구현
3-7. Prompt Optimization 구현 (대화형 상호작용용)

### Phase 4: 시나리오 시뮬레이션 및 KPI 측정
4-1. 시나리오 1~3 시뮬레이션 실행 (점진적 열화 3시점 + 급속 열화 + 대화형 상호작용)
4-2. KPI 측정 (고장 유형 진단 정확도, 추론 근거 설명 품질, Tool 호출 효율성)
4-3. 결과 분석 및 프롬프트 튜닝
4-4. 분석 리포트 및 작업지시서 생성 기능 검증

### Phase 5: Streamlit 데모 UI
5-1. 분석 결과 대시보드 (결함 유형, 단계, 위험도, RUL 시각화)
5-2. 정비 권고 검토 화면
5-3. 대화형 상호작용 인터페이스 (정비 담당자 질의응답)
5-4. 분석 리포트 / 작업지시서 다운로드

### 향후 확장
- Critic Agent, Human-in-the-Loop, Structured Output, Self-Correction

---

## LangGraph StateGraph 설계

### State 스키마

```python
from typing import TypedDict, Literal, Optional
from langchain_core.messages import BaseMessage

class PdMAgentState(TypedDict):
    # 입력 데이터
    event_payload: dict

    # Memory에서 불러온 이력
    memory_context: dict

    # ReAct 추론 과정 (누적)
    messages: list[BaseMessage]

    # 에이전트 판단 결과 (ReAct 루프 완료 후 파싱)
    diagnosis_result: dict
    # diagnosis_result 구조:
    # {
    #   "fault_type": str,           # inner_race / outer_race / rolling_element / cage / none / unknown
    #   "fault_stage": int,          # 0: 정상, 1~4: P-F 곡선 단계
    #   "degradation_speed": str,    # stable / normal / accelerating / abnormal
    #   "rul_assessment": {
    #     "ml_rul_hours": float,
    #     "agent_assessment": str,
    #     "confidence_level": str
    #   },
    #   "risk_level": str,           # normal / watch / warning / critical
    #   "recommendation": str,
    #   "uncertainty_notes": str,
    #   "reasoning_summary": str
    # }

    # Tool 호출 관련 (MCP 기반)
    tool_calls_count: int
    deep_research_activated: bool

    # 리포트 및 작업지시서
    report: str
    work_order: str

    # 워크플로우 제어
    should_continue: bool
    next_action: str  # continue_reasoning / call_tool / generate_report / generate_work_order / end
```

### Node 구성

1. **load_memory**: event_payload에서 설비/베어링 ID 추출 → PostgreSQL에서 이전 판단 이력 조회 → memory_context 저장
2. **reasoning**: 시스템 프롬프트 + event_payload + memory_context + messages → GPT-4o 추론 → next_action 결정
3. **tool_executor**: reasoning에서 Tool 호출 결정 시 MCP Client를 통해 해당 Tool 실행 → 결과를 messages에 추가 → tool_calls_count 증가
4. **parse_diagnosis**: ReAct 루프 완료 후 LLM 응답에서 diagnosis_result를 구조화된 형태로 파싱
5. **generate_report**: diagnosis_result + messages + memory_context → 분석 리포트 생성
6. **generate_work_order**: diagnosis_result + 분석 리포트 기반 → 정비 절차, 자원, 일정을 포함한 작업지시서 생성
7. **save_memory**: diagnosis_result를 PostgreSQL에 저장 + analysis_history VDB에 벡터 적재

### Edge 구성 (노드 간 전이)

```
START → load_memory → reasoning → (조건부 분기)
                                    ├─ next_action == "call_tool" → tool_executor (MCP) → reasoning
                                    ├─ next_action == "continue_reasoning" → reasoning
                                    └─ next_action == "generate_report" → parse_diagnosis → generate_report → generate_work_order → save_memory → END
```

안전장치: tool_calls_count > 10이면 강제로 parse_diagnosis로 전이하여 무한 루프 방지.
참고: generate_work_order는 Warning/Critical 위험도에서만 실행. Normal/Watch에서는 건너뜀.

---

## RAG 설계

### Qdrant Collection 구성

**Collection 1: maintenance_history**
- vector_size: 1536
- distance: Cosine
- payload_index: equipment_id, bearing_id, fault_type
- 문서 수: 12건 (PDF)
- 청크 전략: PDF 1건 = 1청크 (문서 길이가 짧으므로 분할 불필요)

**Collection 2: equipment_manual**
- vector_size: 1536
- distance: Cosine
- payload_index: doc_type, equipment_model
- 문서 수: 7건 (PDF)
- 청크 전략: 사양서/가이드/절차서는 문서당 1청크, FMEA는 고장 모드별 분할 (4청크)

**Collection 3: analysis_history**
- vector_size: 1536
- distance: Cosine
- payload_index: equipment_id, bearing_id, fault_type, risk_level
- 문서: 에이전트 과거 판단 결과 (save_memory 시 자동 적재)
- 청크 전략: 판단 결과 1건 = 1청크 (reasoning_summary + diagnosis_result 임베딩)
- 용도: 과거 유사 분석 사례의 의미적 검색 (PostgreSQL Memory의 SQL 조회와 상호 보완)

검색 설정: top-k = 3 (기본값)

### MCP Server 구성

RAG 검색 및 알림 Tool은 MCP Server로 구현하여, LangGraph Agent가 MCP Client로 호출한다.

- **rag_server**: search_maintenance_history, search_equipment_manual, search_analysis_history 3개 Tool 제공
- **notification_server**: notify_maintenance_staff Tool 제공

### 합성 PDF 문서 목록

**maintenance_history (12건):**

| 파일명 | 내용 | 유형 |
|--------|------|------|
| MH-001_IMS-TESTRIG-01_BRG-003_inner_race.pdf | Test1-Brg3, 내륜, 정상 열화, 35일 | NASA-IMS |
| MH-002_IMS-TESTRIG-01_BRG-004_rolling_element.pdf | Test1-Brg4, 전동체, 정상 열화, 35일 | NASA-IMS |
| MH-003_IMS-TESTRIG-02_BRG-001_outer_race.pdf | Test2-Brg1, 외륜, 급속 열화, 7일 | NASA-IMS |
| MH-004_IMS-TESTRIG-03_BRG-003_outer_race.pdf | Test3-Brg3, 외륜, 정상 열화, 30일 | NASA-IMS |
| MH-005_inner_race_rapid_lubrication.pdf | 내륜, 급속 열화, 윤활 부족 원인, 10일 | 합성 |
| MH-006_inner_race_slow_fatigue.pdf | 내륜, 완만 열화, 정상 피로, 45일 | 합성 |
| MH-007_outer_race_slow_light_load.pdf | 외륜, 완만 열화, 경하중 조건, 60일 | 합성 |
| MH-008_rolling_element_rapid_contamination.pdf | 전동체, 급속 열화, 이물질 오염, 12일 | 합성 |
| MH-009_rolling_element_slow_material.pdf | 전동체, 완만 열화, 재질 결함, 40일 | 합성 |
| MH-010_compound_inner_rolling.pdf | 복합(내륜+전동체), 가속 열화, 25일 | 합성 |
| MH-011_success_early_replacement.pdf | 외륜 2단계 조기 교체 성공 사례 | 합성 |
| MH-012_false_alarm_self_recovery.pdf | 오탐, 일시적 이상 후 자연 복귀 | 합성 |

**equipment_manual (7건):**

| 파일명 | 내용 |
|--------|------|
| EM-001_SPEC_Rexnord-ZA-2115.pdf | 베어링 사양서 (치수, 하중 정격, 결함 주파수, 수명, 급속 열화 조건) |
| EM-002_FAULT_outer_race.pdf | 외륜 결함 가이드 (메커니즘, 원인, 진동 특성, 진행 특성, 대응 기준) |
| EM-003_FAULT_inner_race.pdf | 내륜 결함 가이드 |
| EM-004_FAULT_rolling_element.pdf | 전동체 결함 가이드 |
| EM-005_FAULT_cage.pdf | 보지기 결함 가이드 |
| EM-006_PROC_bearing_replacement.pdf | 베어링 교체 정비 절차서 |
| EM-007_FMEA_IMS-TESTRIG.pdf | FMEA (고장 모드 영향 분석) |

### 정비 이력 보고서 PDF 템플릿

```
===========================================
         정비 이력 보고서
===========================================

문서 번호: MH-XXX
작성일: YYYY-MM-DD
작성자: 정비팀

-------------------------------------------
1. 설비 정보
-------------------------------------------
설비 ID / 설비명 / 위치 / 대상 부품 / 베어링 모델 / 운전 조건

-------------------------------------------
2. 고장 개요
-------------------------------------------
고장 일자 / 고장 유형 / 총 운전 시간 / 최초 이상 감지 일자 /
감지에서 고장까지 소요일 / 감지 방법

-------------------------------------------
3. 고장 진행 경과
-------------------------------------------
시간대별 특징량 변화 및 상태 서술 (초기~고장까지)

-------------------------------------------
4. 근본 원인 분석
-------------------------------------------
분석 결과, 외부 요인 유무

-------------------------------------------
5. 조치 내용
-------------------------------------------
조치 일자 / 조치 내용 / 부품 교체 / 소요 시간 / 투입 인력

-------------------------------------------
6. 교훈 및 권고
-------------------------------------------
모니터링 개선점, 정비 기준 제안, 수명 추정
```

### 설비 기술 문서 PDF 템플릿

**사양서 (EM-001):**
기본 사양 / 하중 정격 / 운전 조건 / 결함 특성 주파수 / 수명 및 신뢰성 / 급속 열화 위험 조건

**고장 모드 가이드 (EM-002~005):**
고장 메커니즘 / 주요 원인 / 진동 특성 / 결함 진행 특성 / 권장 대응 기준

**정비 절차서 (EM-006):**
적용 범위 / 안전 주의사항 / 필요 공구 및 자재 / 작업 절차 / 시운전 및 검증 / 소요 시간 및 인력

**FMEA (EM-007):**
시스템 개요 / 고장 모드 분석 (유형별 원인/영향/심각도/발생도/검출도/RPN) / 위험 우선순위 및 대응 전략

---

## Memory 설계

### Long-term Memory 테이블 (PostgreSQL)

```sql
CREATE TABLE pdm_agent_memory (
    memory_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    equipment_id        VARCHAR(50) NOT NULL,
    bearing_id          VARCHAR(50) NOT NULL,
    event_id            VARCHAR(100) NOT NULL,
    event_timestamp     TIMESTAMP NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 에이전트 판단 결과
    fault_type          VARCHAR(30),
    fault_stage         INTEGER,
    degradation_speed   VARCHAR(20),
    risk_level          VARCHAR(20),
    ml_rul_hours        FLOAT,
    agent_rul_assessment TEXT,

    -- 권고 및 맥락
    recommendation      TEXT,
    uncertainty_notes   TEXT,
    reasoning_summary   TEXT,

    -- Tool 사용 기록
    tools_used          JSONB,
    deep_research       BOOLEAN DEFAULT FALSE,

    -- 후속 조치 추적
    human_response      TEXT,
    action_taken        TEXT,
    resolved            BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_memory_equipment_bearing
ON pdm_agent_memory (equipment_id, bearing_id, event_timestamp DESC);
```

### Memory 조회 (load_memory 노드)

```sql
SELECT * FROM pdm_agent_memory
WHERE equipment_id = '{equipment_id}'
  AND bearing_id = '{bearing_id}'
ORDER BY event_timestamp DESC
LIMIT 5;
```

조회 결과를 자연어로 요약하여 에이전트 messages에 주입한다.

### Memory 저장 (save_memory 노드)

모든 이벤트(정상 포함)에 대해 판단 결과를 저장한다. 정상 판정도 저장하여 "언제부터 이상이 시작되었는가" 추적에 활용한다.

---

## 이벤트 페이로드 스키마

Edge에서 Cloud 에이전트로 전송하는 JSON 구조. 5개 섹션으로 구성.

### 페이로드 구조

```json
{
  "event_id": "EVT-YYYYMMDD-NNNN",
  "timestamp": "ISO 8601",
  "event_type": "periodic_monitoring | anomaly_alert",
  "edge_node_id": "EDGE-XXX",

  "equipment_meta": {
    "equipment_id": "",
    "equipment_name": "",
    "location": "",
    "shaft_rpm": 2000,
    "radial_load_lbs": 6000,
    "operation_start_date": "",
    "operation_days_elapsed": 0,
    "total_running_hours": 0,
    "bearing": {
      "bearing_id": "",
      "position": "",
      "model": "Rexnord ZA-2115",
      "type": "Double Row Bearing",
      "rolling_elements_count": 16,
      "ball_diameter_inch": 0.331,
      "pitch_diameter_inch": 2.815,
      "contact_angle_deg": 15.17,
      "install_date": "",
      "last_maintenance_date": null,
      "defect_frequencies_hz": {
        "BPFO": 236.4,
        "BPFI": 296.9,
        "BSF": 141.1,
        "FTF": 14.8
      }
    },
    "sensor_config": {
      "sensor_count": 0,
      "channels": [],
      "sensor_type": "PCB 353B33 High Sensitivity Accelerometer",
      "sampling_rate_hz": 20000,
      "samples_per_snapshot": 20480,
      "snapshot_interval_min": 10
    }
  },

  "anomaly_detection_result": {
    "model_id": "",
    "anomaly_detected": false,
    "anomaly_score": 0.0,
    "anomaly_threshold": 0.65,
    "health_state": "",
    "confidence": 0.0
  },

  "current_features": {
    "snapshot_timestamp": "",
    "time_domain": {
      "{channel}": {
        "rms": 0.0,
        "peak": 0.0,
        "peak_to_peak": 0.0,
        "crest_factor": 0.0,
        "kurtosis": 0.0,
        "skewness": 0.0,
        "standard_deviation": 0.0,
        "mean": 0.0,
        "shape_factor": 0.0
      }
    },
    "frequency_domain": {
      "{channel}": {
        "bpfo_amplitude": 0.0,
        "bpfi_amplitude": 0.0,
        "bsf_amplitude": 0.0,
        "ftf_amplitude": 0.0,
        "bpfo_harmonics_2x": 0.0,
        "bpfi_harmonics_2x": 0.0,
        "spectral_energy_total": 0.0,
        "spectral_energy_high_freq_band": 0.0,
        "dominant_frequency_hz": 0.0,
        "sideband_presence": false,
        "sideband_spacing_hz": 0.0,
        "sideband_count": 0
      }
    }
  },

  "historical_trend": {
    "period": "",
    "data_points_count": 0,
    "summary": {
      "{feature}_trend": {
        "{channel}": {
          "values_daily_avg": [],
          "dates": [],
          "slope": 0.0,
          "trend_direction": "stable | increasing | accelerating_increase",
          "percent_change_total": 0.0,
          "acceleration_detected": false,
          "acceleration_start_date": ""
        }
      }
    }
  },

  "ml_rul_prediction": {
    "model_id": "",
    "predicted_rul_hours": null,
    "confidence_interval_hours": { "lower": null, "upper": null },
    "prediction_status": "active | not_applicable",
    "reason": ""
  }
}
```

참고: 4건의 시뮬레이션용 페이로드 샘플이 별도 JSON 파일로 존재한다.
- scenario1_day05.json (시나리오1 5일차, 정상)
- scenario1_day15.json (시나리오1 15일차, 초기 열화)
- scenario1_day25.json (시나리오1 25일차, 결함 진행)
- scenario2_day04.json (시나리오2 4일차, 급속 열화)

---

## 시나리오 시뮬레이션

### 시나리오 1: 점진적 열화 감지 및 추적 (Test Set 1, Bearing 3 내륜)

각 시점 개별 분석으로 3개 서브 시나리오 구성:

**SC-001 (시나리오 1-A): 정상 구간 (5일차)**
- anomaly_detected = false, Memory 이력 없음
- 에이전트 Thought 1에서 정상 판정 후 조기 종료
- Memory에 정상 판정 이력 저장

**SC-002 (시나리오 1-B): 의심 전이 구간 (15일차)**
- anomaly_detected = true, Memory에 SC-001 이력 존재
- Thought 1~5 전체 추론: BPFI 상승 → 내륜 결함 초기 → Kurtosis 상승 시작 → 열화 속도 정상 → Watch 판정
- 모니터링 강화 권고

**SC-003 (시나리오 1-C): 결함 진행 구간 (25일차)**
- anomaly_detected = true, Memory에 SC-001 + SC-002 이력 존재
- Thought 1~5 전체 추론 + Deep Research 발동
- search_maintenance_history로 유사 내륜 결함 사례 심층 탐색
- Warning 판정, 분석 리포트 + 작업지시서 생성

### 시나리오 2: 급속 열화 대응 (Test Set 2, Bearing 1 외륜)

**SC-004: 급속 열화 (약 7일간 급속 진행)**
- anomaly_detected = true, acceleration_detected = true
- BPFO 지배적 상승, Kurtosis 감소 + RMS 급상승 (4단계 말기 패턴)
- Deep Research 필수 발동: 급속 열화 조건 검색 → 과거 유사 사례 탐색 → 원인 추론
- Critical 판정, 보수적 RUL 해석, 긴급 리포트 + 작업지시서 생성
- 외부 요인(과부하, 윤활 부족, 오염, 설치 불량) 개입 가능성 명시

### 시나리오 3: 대화형 상호작용 (정비 담당자 질의)

**SC-005: 정비 담당자 질의응답**
- SC-003 또는 SC-004 분석 완료 후, 정비 담당자가 후속 질문
- Memory에서 분석 맥락 로드 → 구조화된 요약(결함 유형, 단계, 위험도, RUL, 핵심 근거) 주입
- Prompt Optimization 적용: 분석 맥락 구조화, 대화 이력 압축, 슬라이딩 윈도우
- 새로운 정보 반영 시 분석 결과 동적 보완 및 Memory 업데이트

### Prompt Optimization 전략

| 전략 | 설명 | 적용 시점 |
|------|------|-----------|
| 분석 맥락 구조화 | Memory에서 전체 추론 체인 대신, 핵심 필드(결함 유형, 단계, 위험도, 핵심 근거)만 추출하여 컨텍스트에 주입 | 대화 세션 시작 시 |
| 대화 이력 압축 | 이전 턴 질문-응답 쌍을 핵심 결론 중심으로 요약 | 매 턴 누적 시 |
| 슬라이딩 윈도우 | 최근 2~3턴은 원문 유지, 이전 턴은 요약으로 대체하여 컨텍스트 윈도우 효율적 활용 | 대화 3턴 이상 시 |

---

## KPI 정의

### 에이전트 성능 KPI

| 지표 | 목표 | 측정 방법 |
|------|------|-----------|
| 고장 유형 진단 정확도 | 85% 이상 | 에이전트가 진단한 고장 유형(외륜, 내륜, 전동체 등)과 NASA-IMS 실제 고장 유형의 일치율 |
| 추론 근거 설명 품질 | 4점/5점 이상 | 전문가 정성 평가 - 추론 근거의 논리성, 도메인 지식 활용도, 불확실성 표현 적절성 |
| Tool 호출 효율성 | 정성 평가 | 불필요한 Tool 호출 없이 상황에 적절한 Tool만 호출했는지 평가 (정상 시 Tool 미호출, 심각 시 Deep Research 발동 등) |
