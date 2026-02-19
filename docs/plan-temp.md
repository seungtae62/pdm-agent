╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Plan to implement                                                                                                                                  │
│                                                                                                                                                    │
│ Edge 파이프라인 — Part 3: Anomaly Detection (3단계)                                                                                                │
│                                                                                                                                                    │
│ Context                                                                                                                                            │
│                                                                                                                                                    │
│ Feature extraction(Part 1)과 Health Index(Part 2) 위에 이상 감지 레이어를 추가한다.                                                                │
│ 3개의 독립적인 감지 모듈이 병행 실행되고, Combiner가 결과를 결합하여 최종 anomaly_score를 산출한다.                                                │
│ 구현은 A → B → C 3단계로 나누며, 각 단계마다 Combiner를 확장한다.                                                                                  │
│                                                                                                                                                    │
│ Edge는 단일 스냅샷 기준 무상태(stateless) 감지만 수행한다.                                                                                         │
│ Historical Trend는 edge 역량이 아님 — 플랫폼(Cloud) 레이어에서 처리.                                                                               │
│                                                                                                                                                    │
│ ---                                                                                                                                                │
│ Edge 전체 파이프라인 구조                                                                                                                          │
│                                                                                                                                                    │
│ [Raw Signal] (20,480 × N ch, per snapshot)                                                                                                         │
│     │                                                                                                                                              │
│     ▼                                                                                                                                              │
│ ┌────────────────────────────────┐                                                                                                                 │
│ │  Feature Extraction  [완료]    │                                                                                                                 │
│ │  → flat_features (21개)        │                                                                                                                 │
│ │  → payload: current_features   │                                                                                                                 │
│ └───────────────┬────────────────┘                                                                                                                 │
│                 │                                                                                                                                  │
│    ┌────────────┼────────────────────┐                                                                                                             │
│    │            │                    │                                                                                                             │
│    ▼            ▼                    ▼                                                                                                             │
│ [A] HI+Rule  [B] Statistical   [C] Autoencoder                                                                                                     │
│  (Part 3a)     (Part 3b)         (Part 3c)                                                                                                         │
│    │            │                    │                                                                                                             │
│    └──────┬─────┘────────────────────┘                                                                                                             │
│           ▼                                                                                                                                        │
│     ┌───────────┐                                                                                                                                  │
│     │ Combiner  │ → anomaly_detection_result                                                                                                       │
│     └───────────┘                                                                                                                                  │
│                                                                                                                                                    │
│ 모듈별 입출력:                                                                                                                                     │
│                                                                                                                                                    │
│ ┌────────────────────┬─────────────────────────────────┬───────────────────────────────────────────────┬──────────────────────────────────────┐    │
│ │        모듈        │              입력               │                     출력                      │             Payload 위치             │    │
│ ├────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────┤    │
│ │ Feature Extraction │ raw signal                      │ flat_features (21개)                          │ current_features                     │    │
│ ├────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────┤    │
│ │ [A] HI + Rule      │ flat_features + BaselineStats   │ rule_score, spiked_keys                       │ anomaly_detection_result.rule_based  │    │
│ ├────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────┤    │
│ │ [B] Statistical    │ flat_features + AnomalyBaseline │ stat_score, z_scores                          │ anomaly_detection_result.statistical │    │
│ ├────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────┤    │
│ │ [C] Autoencoder    │ flat_features + trained model   │ model_score, recon_error                      │ anomaly_detection_result.model_based │    │
│ ├────────────────────┼─────────────────────────────────┼───────────────────────────────────────────────┼──────────────────────────────────────┤    │
│ │ Combiner           │ A+B+C 결과                      │ anomaly_detected, anomaly_score, health_state │ anomaly_detection_result (top-level) │    │
│ └────────────────────┴─────────────────────────────────┴───────────────────────────────────────────────┴──────────────────────────────────────┘    │
│                                                                                                                                                    │
│ Payload 신규 섹션: health_index (Agent가 HI 값 직접 해석 가능)                                                                                     │
│                                                                                                                                                    │
│ ---                                                                                                                                                │
│ 구현 순서                                                                                                                                          │
│                                                                                                                                                    │
│ Part 3a: [A] HI + Rule Check ← 이번 구현 대상                                                                                                      │
│                                                                                                                                                    │
│ Feature에서 HI를 산출(기존 Part 2)하고, HI 값에 규칙 기반 임계값 검사를 적용.                                                                      │
│                                                                                                                                                    │
│ 파일:                                                                                                                                              │
│                                                                                                                                                    │
│ ┌──────┬─────────────────────────────────┐                                                                                                         │
│ │ 구분 │              경로               │                                                                                                         │
│ ├──────┼─────────────────────────────────┤                                                                                                         │
│ │ 수정 │ edge/config.py                  │                                                                                                         │
│ ├──────┼─────────────────────────────────┤                                                                                                         │
│ │ 생성 │ edge/anomaly_detection.py       │                                                                                                         │
│ ├──────┼─────────────────────────────────┤                                                                                                         │
│ │ 생성 │ tests/test_anomaly_detection.py │                                                                                                         │
│ └──────┴─────────────────────────────────┘                                                                                                         │
│                                                                                                                                                    │
│ 상세 설계:                                                                                                                                         │
│                                                                                                                                                    │
│ config.py 추가 상수:                                                                                                                               │
│ # HI → anomaly score 매핑 breakpoints (구간별 선형 보간)                                                                                           │
│ ANOMALY_HI_BREAKPOINTS: list[tuple[float, float]] = [                                                                                              │
│     (1.0, 0.0),    # 베이스라인 범위 내 → 정상                                                                                                     │
│     (2.0, 0.65),   # 2배 이탈 → 이상 판정 경계                                                                                                     │
│     (3.5, 0.90),   # 3.5배 이탈 → critical 경계                                                                                                    │
│     (5.0, 1.0),    # 5배 이탈 → 최대                                                                                                               │
│ ]                                                                                                                                                  │
│ ANOMALY_SPIKE_THRESHOLD: float = 2.0                                                                                                               │
│ ANOMALY_THRESHOLD: float = 0.65                                                                                                                    │
│ ANOMALY_HEALTH_STATE_TIERS: list[tuple[float, str]] = [                                                                                            │
│     (0.65, "normal"), (0.80, "watch"), (0.90, "warning"),                                                                                          │
│ ]                                                                                                                                                  │
│ ANOMALY_MODEL_ID: str = "rule_v1"                                                                                                                  │
│                                                                                                                                                    │
│ anomaly_detection.py 구조:                                                                                                                         │
│ @dataclass(frozen=True)                                                                                                                            │
│ class RuleCheckDetail:                                                                                                                             │
│     composite_hi_score: float                                                                                                                      │
│     spike_score: float                                                                                                                             │
│     spiked_keys: list[str]                                                                                                                         │
│                                                                                                                                                    │
│ @dataclass(frozen=True)                                                                                                                            │
│ class AnomalyResult:                                                                                                                               │
│     model_id: str                                                                                                                                  │
│     anomaly_detected: bool                                                                                                                         │
│     anomaly_score: float                                                                                                                           │
│     anomaly_threshold: float                                                                                                                       │
│     health_state: str          # normal | watch | warning | critical                                                                               │
│     confidence: float                                                                                                                              │
│     rule_detail: RuleCheckDetail                                                                                                                   │
│     # Part 3b에서 stat_detail 추가                                                                                                                 │
│     # Part 3c에서 model_detail 추가                                                                                                                │
│                                                                                                                                                    │
│ def _hi_to_score(hi_value: float) -> float                                                                                                         │
│     """HI → 0~1 score. 구간별 선형 보간."""                                                                                                        │
│                                                                                                                                                    │
│ def check_rules(hi_result: HealthIndexResult) -> RuleCheckDetail                                                                                   │
│     """Composite HI score + Individual spike score → max."""                                                                                       │
│                                                                                                                                                    │
│ def _classify_health_state(anomaly_score: float) -> str                                                                                            │
│                                                                                                                                                    │
│ def detect_anomaly(                                                                                                                                │
│     hi_result: HealthIndexResult,                                                                                                                  │
│     *,                                                                                                                                             │
│     threshold: float | None = None,                                                                                                                │
│ ) -> AnomalyResult                                                                                                                                 │
│     """Part 3a: Rule only → anomaly_score = rule_score."""                                                                                         │
│                                                                                                                                                    │
│ HI Rule 로직:                                                                                                                                      │
│ - _hi_to_score(): 공통 piecewise linear 매핑 (HI→score)                                                                                            │
│ - Composite HI → _hi_to_score(composite) → composite_score                                                                                         │
│ - Max individual HI (≥ 2.0인 것 중 최대) → _hi_to_score(max) → spike_score                                                                         │
│ - rule_score = max(composite_score, spike_score) — 어느 하나라도 충분                                                                              │
│ - anomaly_score = rule_score (Part 3a에서는 Rule만 사용)                                                                                           │
│                                                                                                                                                    │
│ Confidence (Part 3a):                                                                                                                              │
│ - 두 점수(composite, spike) 일치도 기반: 0.5 + 0.5 × (1 - |comp - spike|)                                                                          │
│ - Part 3b에서 Rule↔Stat 일치도로 확장                                                                                                              │
│                                                                                                                                                    │
│ 테스트:                                                                                                                                            │
│                                                                                                                                                    │
│ ┌─────────────────────────┬────────────────────────────────────────────────────────┐                                                               │
│ │         클래스          │                         케이스                         │                                                               │
│ ├─────────────────────────┼────────────────────────────────────────────────────────┤                                                               │
│ │ TestHiToScore           │ breakpoint 정확, 보간, 경계, ≤1.0→0, >5.0→1.0          │                                                               │
│ ├─────────────────────────┼────────────────────────────────────────────────────────┤                                                               │
│ │ TestCheckRules          │ 정상→0, 스파이크만→감지, composite만→감지, 둘 다→max   │                                                               │
│ ├─────────────────────────┼────────────────────────────────────────────────────────┤                                                               │
│ │ TestClassifyHealthState │ 4 states + 3 경계값                                    │                                                               │
│ ├─────────────────────────┼────────────────────────────────────────────────────────┤                                                               │
│ │ TestDetectAnomaly       │ 정상/열화, model_id, threshold, score 범위, confidence │                                                               │
│ ├─────────────────────────┼────────────────────────────────────────────────────────┤                                                               │
│ │ TestIntegration         │ 점진 열화 단조 증가, HI→detect 라운드트립              │                                                               │
│ └─────────────────────────┴────────────────────────────────────────────────────────┘                                                               │
│                                                                                                                                                    │
│ ---                                                                                                                                                │
│ Part 3b: [B] Statistical Anomaly (z-score) — 다음 단계                                                                                             │
│                                                                                                                                                    │
│ Feature의 원본값을 baseline mean/std와 비교하여 통계적 이탈도를 산출.                                                                              │
│                                                                                                                                                    │
│ 추가 사항:                                                                                                                                         │
│ - AnomalyBaseline dataclass (mean/std)                                                                                                             │
│ - compute_anomaly_baseline() 함수                                                                                                                  │
│ - StatCheckDetail dataclass (z_scores, max_z_score, max_z_feature)                                                                                 │
│ - check_statistical() 함수                                                                                                                         │
│ - AnomalyResult에 stat_detail 필드 추가                                                                                                            │
│ - detect_anomaly() 확장: anomaly_score = w_rule × rule + w_stat × stat                                                                             │
│ - Confidence: Rule↔Stat 일치도로 변경                                                                                                              │
│                                                                                                                                                    │
│ ---                                                                                                                                                │
│ Part 3c: [C] Autoencoder — 마지막 단계                                                                                                             │
│                                                                                                                                                    │
│ 간단한 MLP Autoencoder로 정상 패턴 학습 → reconstruction error 기반 감지.                                                                          │
│                                                                                                                                                    │
│ 구조:                                                                                                                                              │
│ Encoder: 21 → 8 (ReLU)                                                                                                                             │
│ Decoder: 8 → 21 (Linear)                                                                                                                           │
│ 학습: 베이스라인 20개 스냅샷, MSE loss                                                                                                             │
│ 이상 점수: reconstruction MSE → 정규화된 score                                                                                                     │
│ 프레임워크: PyTorch                                                                                                                                │
│                                                                                                                                                    │
│ 추가 사항:                                                                                                                                         │
│ - edge/autoencoder.py: 모델 정의, 학습, 추론                                                                                                       │
│ - ModelCheckDetail dataclass (recon_error, score)                                                                                                  │
│ - AnomalyResult에 model_detail 필드 추가                                                                                                           │
│ - detect_anomaly() 확장: anomaly_score = w_rule × rule + w_stat × stat + w_model × model                                                           │
│ - 학습 스크립트 또는 train_autoencoder() 함수                                                                                                      │
│                                                                                                                                                    │
│ ---                                                                                                                                                │
│ 최종 Event Payload 스키마 (3단계 완료 후)                                                                                                          │
│                                                                                                                                                    │
│ {                                                                                                                                                  │
│   "event_id": "EVT-YYYYMMDD-NNNN",                                                                                                                 │
│   "timestamp": "ISO 8601",                                                                                                                         │
│   "event_type": "periodic_monitoring | anomaly_alert",                                                                                             │
│   "edge_node_id": "EDGE-XXX",                                                                                                                      │
│                                                                                                                                                    │
│   "equipment_meta": { "..." },                                                                                                                     │
│                                                                                                                                                    │
│   "current_features": {                                                                                                                            │
│     "snapshot_timestamp": "",                                                                                                                      │
│     "time_domain": { "{ch}": { "...9개..." } },                                                                                                    │
│     "frequency_domain": { "{ch}": { "...12개..." } }                                                                                               │
│   },                                                                                                                                               │
│                                                                                                                                                    │
│   "health_index": {                                                                                                                                │
│     "baseline_snapshot_count": 20,                                                                                                                 │
│     "individual": {                                                                                                                                │
│       "hi_rms": 0.0,                                                                                                                               │
│       "hi_kurtosis": 0.0,                                                                                                                          │
│       "hi_crest_factor": 0.0,                                                                                                                      │
│       "hi_peak_frequency": 0.0,                                                                                                                    │
│       "hi_fft_energy": 0.0                                                                                                                         │
│     },                                                                                                                                             │
│     "composite": 0.0                                                                                                                               │
│   },                                                                                                                                               │
│                                                                                                                                                    │
│   "anomaly_detection_result": {                                                                                                                    │
│     "model_id": "rule_zscore_ae_v1",                                                                                                               │
│     "anomaly_detected": false,                                                                                                                     │
│     "anomaly_score": 0.0,                                                                                                                          │
│     "anomaly_threshold": 0.65,                                                                                                                     │
│     "health_state": "normal",                                                                                                                      │
│     "confidence": 0.0,                                                                                                                             │
│     "rule_based": {                                                                                                                                │
│       "score": 0.0,                                                                                                                                │
│       "composite_hi_score": 0.0,                                                                                                                   │
│       "spike_score": 0.0,                                                                                                                          │
│       "spiked_keys": []                                                                                                                            │
│     },                                                                                                                                             │
│     "statistical": {                                                                                                                               │
│       "score": 0.0,                                                                                                                                │
│       "z_scores": { "hi_rms": 0.0, "...": "..." },                                                                                                 │
│       "max_z_score": 0.0,                                                                                                                          │
│       "max_z_feature": ""                                                                                                                          │
│     },                                                                                                                                             │
│     "model_based": {                                                                                                                               │
│       "score": 0.0,                                                                                                                                │
│       "reconstruction_error": 0.0                                                                                                                  │
│     }                                                                                                                                              │
│   },                                                                                                                                               │
│                                                                                                                                                    │
│   "historical_trend": { "...플랫폼에서 제공..." },                                                                                                 │
│   "ml_rul_prediction": { "...Phase 1-3에서 구현..." }                                                                                              │
│ }                                                                                                                                                  │
│                                                                                                                                                    │
│ ---                                                                                                                                                │
│ Part 3a 검증 방법                                                                                                                                  │
│                                                                                                                                                    │
│ 1. pytest tests/test_anomaly_detection.py -v — 전체 통과                                                                                           │
│ 2. pytest tests/ -v — 기존 49개 + 신규 전체 통과                                                                                                   │
│ 3. 라운드트립:                                                                                                                                     │
│ from edge.health_index import compute_baseline, compute_health_indices                                                                             │
│ from edge.anomaly_detection import detect_anomaly                                                                                                  │
│                                                                                                                                                    │
│ baseline = compute_baseline(feature_dicts[:20])                                                                                                    │
│ hi = compute_health_indices(feature_dicts[25], baseline)                                                                                           │
│ result = detect_anomaly(hi)                                                                                                                        │
│ # result.anomaly_detected, result.health_state 확인                                                                                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
