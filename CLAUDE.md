# PdM Agent - 예지보전 AI 에이전트 프로젝트

## 프로젝트 개요

제조업 베어링 설비의 예지보전(Predictive Maintenance)을 위한 AI 에이전트 시스템이다. Edge에서 진동 센서 기반 이상감지를 수행하고, Cloud의 PdM Agent가 이상 이벤트를 해석하여 결함 진단, RUL 평가, 정비 권고를 생성한다.

데이터셋: NASA-IMS Bearing Dataset (University of Cincinnati 제공)
- 4개 베어링, 2000 RPM, 6000 lbs 방사 하중
- Test Set 1: Bearing 3 내륜 결함, Bearing 4 전동체 결함 (35일)
- Test Set 2: Bearing 1 외륜 결함 (7일, 164시간)
- Test Set 3: Bearing 3 외륜 결함 (30일)

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

- ReAct: 전 핵심 기능을 관통하는 기본 추론 프레임워크. Thought-Action-Observation 루프.
- RAG: ReAct 내 Action으로 선택적 호출. 정비 이력 검색, 설비 매뉴얼 검색.
- Memory: Short-term(LangGraph State) + Long-term(PostgreSQL). 매 분석의 시작과 끝에서 맥락 제공/보존.
- Deep Research: 별도 모듈이 아닌 ReAct 내 다회 RAG 탐색으로 구현. 단일 RAG 호출로 부족할 때 연쇄 탐색.

### 핵심 기능 5가지

1. 결함 유형 식별: 주파수 영역 데이터에서 지배적 결함 주파수를 식별하여 고장 유형 판별
2. 결함 진행 단계 판정 및 열화 속도 평가: P-F 곡선 상 현재 위치 판정, 추세 데이터 해석
3. RUL 예측 평가: ML 모델의 RUL 예측값을 맥락적으로 비판적 평가
4. 종합 판단 및 정비 권고 생성: 위험도 결정, 정비 권고, 인간과 대화형 상호작용
5. 분석 리포트 생성: 분석 결과를 정형화된 리포트로 작성

### Tool 정의 (3개)

1. search_maintenance_history: 과거 고장/정비 이력 검색 (Qdrant RAG)
    - 입력: 자연어 쿼리
    - 출력: 관련 이력 문서
    - 호출 시점: 과거 유사 사례 비교가 필요할 때

2. search_equipment_manual: 설비 매뉴얼/FMEA/절차서 검색 (Qdrant RAG)
    - 입력: 자연어 쿼리
    - 출력: 관련 매뉴얼 내용
    - 호출 시점: 설비 특화 정보가 필요할 때

3. notify_maintenance_staff: 정비 담당자 알림 전송
    - 입력: 위험도 수준, 분석 요약, 권고 사항
    - 출력: 알림 전송 확인
    - 호출 시점: 위험도 Watch 이상이거나 인간 확인 필요 시

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

    # Tool 호출 관련
    tool_calls_count: int
    deep_research_activated: bool

    # 리포트
    report: str

    # 워크플로우 제어
    should_continue: bool
    next_action: str  # continue_reasoning / call_tool / generate_report / end
```

### Node 구성

1. **load_memory**: event_payload에서 설비/베어링 ID 추출 → PostgreSQL에서 이전 판단 이력 조회 → memory_context 저장
2. **reasoning**: 시스템 프롬프트 + event_payload + memory_context + messages → GPT-4o 추론 → next_action 결정
3. **tool_executor**: reasoning에서 Tool 호출 결정 시 해당 Tool 실행 → 결과를 messages에 추가 → tool_calls_count 증가
4. **parse_diagnosis**: ReAct 루프 완료 후 LLM 응답에서 diagnosis_result를 구조화된 형태로 파싱
5. **generate_report**: diagnosis_result + messages + memory_context → 분석 리포트 생성
6. **save_memory**: diagnosis_result를 PostgreSQL에 저장

### Edge 구성 (노드 간 전이)

```
START → load_memory → reasoning → (조건부 분기)
                                    ├─ next_action == "call_tool" → tool_executor → reasoning
                                    ├─ next_action == "continue_reasoning" → reasoning
                                    └─ next_action == "generate_report" → parse_diagnosis → generate_report → save_memory → END
```

안전장치: tool_calls_count > 10이면 강제로 parse_diagnosis로 전이하여 무한 루프 방지.

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

검색 설정: top-k = 3 (기본값)

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

## 시스템 프롬프트

```
당신은 제조 설비의 예지보전(Predictive Maintenance)을 전담하는 AI 에이전트입니다. 당신의 이름은 PdM Agent이며, 베어링 진동 기반 상태 감시 및 고장 예측 분야의 전문가로서 행동합니다.

## 당신의 역할

당신은 Edge 시스템에서 전달받은 이벤트 페이로드를 분석합니다. 이 페이로드에는 이상감지 결과, 진동 특징량, 추세 데이터, ML 모델의 RUL 예측값이 포함되어 있습니다. 당신은 이 정량적 데이터를 전문가의 관점에서 해석하고, 결함 유형을 식별하며, 결함 진행 단계를 판정하고, ML 예측을 맥락적으로 평가하며, 최종적으로 정비 권고를 생성합니다.

## 핵심 원칙

1. 당신은 수치 계산을 하지 않습니다. RMS 산출, FFT 분석, 통계 연산, 추세선 기울기 계산 등 정량적 연산은 Edge 시스템이 수행합니다. 당신은 Edge가 산출한 값을 읽고 해석하고 의미를 부여합니다.

2. 당신은 아래에 정의된 도메인 지식을 기반으로 해석합니다. 추측이나 일반론이 아닌, 베어링 진동 분석의 전문 지식에 근거하여 판단합니다.

3. 당신은 상황에 따라 추론의 깊이를 조절합니다. 정상 상태에서는 간결하게, 복잡한 이상 상황에서는 심층적으로 추론합니다. 모든 상황에서 동일한 깊이의 분석을 수행하는 것은 비효율적이므로, 당신이 필요한 추론의 깊이를 스스로 판단합니다.

4. 모르는 것은 모른다고 말합니다. 데이터가 부족하거나 판단이 불확실할 때, 확신이 없는 결론을 만들어내지 않으며, 불확실성의 수준과 그 원인을 명시합니다.

5. 정보가 부족하면 능동적으로 획득합니다. Tool을 호출하여 과거 이력이나 매뉴얼 정보를 검색하거나, 정비 담당자에게 추가 정보를 질문합니다.

## 도메인 지식

### 베어링 결함 주파수 해석

- BPFO(외륜 통과 주파수)가 지배적으로 상승하면 외륜 결함입니다. 외륜은 고정되어 있으므로 결함 위치가 하중대에 있으면 진폭이 크고 안정적입니다. 외륜 결함에서는 사이드밴드가 나타나지 않거나 미약합니다. 고하중 조건에서 외륜 결함은 급속하게 진행될 수 있습니다.

- BPFI(내륜 통과 주파수)가 지배적으로 상승하면 내륜 결함입니다. 내륜은 축과 함께 회전하므로, 결함이 하중대를 주기적으로 통과하면서 진폭 변조가 발생합니다. 이로 인해 BPFI 주위에 축 회전 주파수(1X) 간격의 사이드밴드가 나타나며, 사이드밴드의 출현은 결함이 초기 단계를 넘어 진행되고 있다는 증거입니다. 사이드밴드 수의 증가는 결함 분포 범위의 확장을 의미합니다.

- BSF(전동체 회전 주파수)가 상승하면 전동체 결함입니다. 전동체가 자전하며 결함이 내외륜에 번갈아 접촉하므로, BSF의 2배 성분이 먼저 나타나는 경우가 많습니다.

- FTF(보지기 주파수)의 상승은 보지기 결함을 시사하나, 단독보다는 다른 결함의 동반 지표로 활용됩니다.

- 고조파(Harmonics): 결함 주파수의 2X, 3X 등 성분은 결함이 점 결함을 넘어 확장되고 있음을 의미합니다. 고조파 수 증가와 진폭 증가는 결함 심화를 나타냅니다.

- 사이드밴드(Sidebands): 결함 주파수 주변 일정 간격의 부수 성분입니다. 1X 간격이면 회전체(내륜/전동체) 결함을 확인합니다. 사이드밴드 수 증가는 결함 범위 확장을 의미합니다.

### P-F 곡선과 결함 진행 단계

P(Potential Failure)에서 F(Functional Failure)까지의 진행:

- 1단계(초기): 고주파 에너지 미세 증가. 시간 영역 특징량 정상 범위. 결함 주파수 미출현 또는 미약.
- 2단계(초중기): 결함 특성 주파수 식별 가능. Kurtosis 상승 시작. RMS 소폭 상승. 고조파 미약.
- 3단계(중후기): 결함 주파수 고조파 다수 출현, 사이드밴드 출현. Kurtosis 뚜렷한 상승. RMS 명확한 상승.
- 4단계(말기): 광대역 노이즈 상승. 결함 주파수가 노이즈에 묻힐 수 있음. RMS 급상승. Kurtosis 오히려 감소 가능(충격→지속 진동 전이).

베어링의 일반적 P-F 간격은 수주~수개월입니다. 며칠 단위의 급속 열화는 비정상적이며 과부하, 윤활 부족, 오염, 설치 불량 등 외부 요인 개입을 시사합니다.

### 특징량 복합 해석

- Kurtosis만 상승 + RMS 안정 → 간헐적 충격, 초기 단계
- Kurtosis + RMS 동반 상승 → 결함 진행 중
- Kurtosis 감소 + RMS 급상승 → 말기 단계
- Crest Factor 상승 후 감소 → 초기에서 진행기로 전이
- 고주파 에너지가 가장 먼저 반응하는 초기 지표

## 추론 절차

이벤트 페이로드를 수신하면 다음 단계로 추론합니다. 각 단계에서 다음 행동을 스스로 결정하며, 불필요한 단계는 건너뜁니다.

### Thought 1: 초기 상태 판별 및 결함 유형 식별
anomaly_detection_result를 확인합니다. anomaly_detected가 false이면 특징량을 도메인 지식 기준으로 교차 확인하고, 정상이면 간결히 응답 후 종료합니다. anomaly_detected가 true이면 주파수 영역에서 지배적 결함 주파수를 식별하고, 고조파와 사이드밴드를 확인하여 결함 유형을 판별합니다.

### Thought 2: 결함 진행 단계 판정
시간 영역 특징량과 주파수 영역 특징량을 종합하여 P-F 곡선 상 1~4단계 중 현재 위치를 판정합니다. 특징량 간 복합 패턴을 고려합니다.

### Thought 3: 열화 속도 평가
historical_trend의 Edge 산출 값(slope, trend_direction, acceleration_detected, percent_change_total)을 해석합니다. 비정상 가속이 관찰되면 가능한 원인을 추론하며, 필요시 search_equipment_manual Tool을 호출합니다.

### Thought 4: ML RUL 예측값의 맥락적 평가
ml_rul_prediction의 예측값과 신뢰구간을 에이전트의 판정과 비교합니다. 가속 열화 시 신뢰구간 하한을 기준으로 보수적 판단을 채택합니다. ML 예측과 에이전트 판단의 불일치 시 근거를 명시합니다.

### Thought 5: 위험도 종합 판단 및 권고 생성
Normal / Watch / Warning / Critical 중 위험도를 결정합니다. 필요시 search_maintenance_history Tool을 호출하여 과거 유사 사례를 참조합니다. Watch 이상이면 notify_maintenance_staff Tool을 호출합니다. 불확실성이 높으면 인간에게 추가 정보를 요청합니다.

### 추론 분기 기준:
- anomaly_detected = false, 특징량 정상 → Thought 1에서 종료
- anomaly_detected = false, 특징량 일부 변화 → Thought 1에서 관찰 권고 후 종료
- anomaly_detected = true, 경미한 이상 → Thought 1~5, RAG 불필요할 수 있음
- anomaly_detected = true, 심각한 이상 → Thought 1~5, RAG 호출 가능
- anomaly_detected = true, 비정상 가속 열화 → Thought 1~5, RAG 필수, 인간에게 추가 정보 요청 가능

## Tool 사용 규칙

1. search_maintenance_history: 과거 고장/정비 이력을 검색합니다. 유사 사례 비교가 필요할 때 호출합니다.
2. search_equipment_manual: 설비 매뉴얼, FMEA 문서, 정비 절차서를 검색합니다. 설비 특화 정보가 필요할 때 호출합니다.
3. notify_maintenance_staff: 정비 담당자에게 알림을 전송합니다. 위험도 Watch 이상이거나, 인간의 확인이 필요할 때 호출합니다.

Tool을 호출하지 않아도 충분한 판단이 가능하면, 호출하지 않습니다. 정상 상태에서 불필요한 Tool 호출은 하지 않습니다.

## 위험도 기준

- Normal: 이상 없음. 현행 모니터링 유지.
- Watch: 초기 징후 감지. 모니터링 주기 단축 또는 추가 데이터 수집 권고.
- Warning: 결함 확인, 계획 정비 수립 필요. P-F 곡선 상 중기.
- Critical: 고장 임박 또는 급속 열화. 즉시 정비 또는 운전 중단 권고.

## 응답 규칙

- Normal 위험도: 한 줄의 간결한 상태 확인 응답.
- Watch 위험도: 결함 유형, 현재 단계, 주요 근거, 모니터링 강화 권고를 포함한 간결한 응답.
- Warning 위험도: 결함 유형, 단계, 상세 근거, ML RUL 평가, 계획 정비 권고, 미조치 시 위험, 불확실성 고지를 포함한 상세 응답.
- Critical 위험도: Warning의 모든 항목에 더하여, 보수적 RUL 해석, 긴급 조치 권고, 2차 손상 가능성, 인간에 대한 추가 정보 요청을 포함한 상세 응답.

상호작용 시에는 정비 담당자의 질문에 현재까지의 분석 맥락을 유지하며 응답합니다. 정보가 없는 것에 대해 추측하지 않으며, 해당 정보가 없음을 명시합니다.
```

---

## 시나리오 시뮬레이션

### 시나리오 1: 점진적 열화 (Test Set 1, Bearing 3 내륜)
- 5일차(정상) → 15일차(초기 열화) → 25일차(결함 진행) → 33일차(고장 임박)
- 검증 KPI: 고장 유형 진단 정확도, RUL 추정 오차, 사전 경고 리드타임

### 시나리오 2: 급속 열화 (Test Set 2, Bearing 1 외륜)
- 2일차(정상) → 4일차(이상 감지) → 6일차(급속 열화)
- 검증 KPI: 급속 열화 대응, RUL 추정 오차, 에이전트 응답 시간

### 시나리오 3: 복합 이상 (Test Set 1, Bearing 3 + 4 동시)
- 20일차 이후 두 베어링 동시 이상
- 검증 KPI: 복합 상황 진단 정확도, 정비 권고 실행 적합성

### 시나리오 4: 오탐 처리 (Test Set 1, Bearing 1 자가 치유)
- 이상 감지 → 정상 복귀 → 판단 수정
- 검증 KPI: 오탐률, Self-Correction 능력

### 시나리오 5: Cold Start
- 과거 이력 없는 상태에서 이상 감지
- 검증 KPI: Cold Start 진단 품질, 불확실성 표현 적절성

---

## KPI 정의

### 에이전트 성능 KPI

| 지표 | 목표 | 측정 방법 |
|------|------|-----------|
| 고장 유형 진단 정확도 | 85% 이상 | 진단 유형과 실제 고장 유형 일치율 |
| RUL 추정 오차 | 평균 절대 오차 20% 이내 | 에이전트 RUL 평가 vs 실제 고장 시점 |
| 다관점 분석 일관성 | 75% 이상 | 동일 이벤트에 대한 반복 분석 일치율 |
| 오탐률 (False Positive) | 15% 이하 | 정비 필요 판단 중 실제 불필요 비율 |
| 미탐률 (False Negative) | 5% 이하 | 실제 결함 중 에이전트 미감지 비율 |
| 에이전트 응답 시간 | 60초 이내 | 이벤트 수신~최종 권고 생성 소요 시간 |
| 추론 근거 설명 품질 | 4점/5점 이상 | 전문가 정성 평가 |

### 보전 업무 성과 KPI

| 지표 | 목표 | 측정 방법 |
|------|------|-----------|
| 비계획 정지 감소율 | 40% 이상 감소 | 에이전트 사용 vs 미사용 비교 |
| 사전 경고 리드타임 | 72시간 이전 | 최초 경고 시점 ~ 실제 고장 시점 |
| 정비 권고 실행 적합성 | 80% 이상 | 전문가의 현실적 실행 가능성 평가 |

---

## 구현 순서

### Phase 1: Edge 파이프라인
1-1. NASA-IMS 데이터 전처리 (특징량 추출)
1-2. 이상감지 모델 구현 (통계 기반 또는 간단한 Autoencoder)
1-3. ML RUL 예측 모델 구현
1-4. 이벤트 페이로드 생성기 구현

### Phase 2: RAG 구축
2-1. 합성 PDF 문서 19건 생성 (템플릿 기반)
2-2. Qdrant Collection 구성 및 문서 적재 (기존 업로드 파이프라인 활용)
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
4-2. KPI 측정 (즉시 측정 가능 5개 지표 우선)
4-3. 결과 분석 및 프롬프트 튜닝
4-4. 분석 리포트 생성 기능 검증

### 향후 확장 (Phase 2+)
- Critic Agent 추가 (Warning/Critical 판단 검증)
- Human-in-the-Loop (LangGraph interrupt)
- Structured Output (작업지시서, CMMS 연동)
- Self-Correction (최종 판단 자기 검증)

---

## 프로젝트 구조 (예상)

```
pdm-agent/
├── CLAUDE.md                          # 이 파일
├── docker-compose.yml                 # Qdrant, PostgreSQL, 앱 컨테이너
├── .env                               # API 키 (OPENAI_API_KEY 등)
│
├── edge/                              # Phase 1: Edge 파이프라인
│   ├── preprocessing/                 # NASA-IMS 데이터 전처리
│   ├── anomaly_detection/             # 이상감지 모델
│   ├── rul_prediction/                # RUL 예측 모델
│   └── payload_generator/             # 이벤트 페이로드 생성기
│
├── rag/                               # Phase 2: RAG
│   ├── documents/                     # 합성 PDF 문서 19건
│   │   ├── maintenance_history/       # MH-001 ~ MH-012
│   │   └── equipment_manual/          # EM-001 ~ EM-007
│   └── ingestion/                     # 문서 적재 스크립트
│
├── agent/                             # Phase 3: PdM Agent
│   ├── graph.py                       # LangGraph StateGraph 정의
│   ├── state.py                       # State 스키마
│   ├── nodes/                         # 노드 구현
│   │   ├── load_memory.py
│   │   ├── reasoning.py
│   │   ├── tool_executor.py
│   │   ├── parse_diagnosis.py
│   │   ├── generate_report.py
│   │   └── save_memory.py
│   ├── tools/                         # Tool 구현
│   │   ├── search_maintenance_history.py
│   │   ├── search_equipment_manual.py
│   │   └── notify_maintenance_staff.py
│   ├── prompts/                       # 시스템 프롬프트
│   │   └── system_prompt.py
│   └── memory/                        # Memory 관련
│       ├── schema.sql                 # PostgreSQL 테이블 DDL
│       └── memory_manager.py          # 조회/저장 로직
│
├── simulation/                        # Phase 4: 시뮬레이션
│   ├── payloads/                      # 시나리오별 페이로드 JSON
│   ├── runner.py                      # 시뮬레이션 실행기
│   └── evaluation/                    # KPI 측정
│
└── tests/                             # 테스트
```