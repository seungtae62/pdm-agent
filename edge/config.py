"""Edge 파이프라인 설정값."""

from __future__ import annotations

# 데이터 수집 파라미터
SAMPLING_RATE_HZ = 20_000
SAMPLES_PER_SNAPSHOT = 20_480

# 회전체 정보
SHAFT_RPM = 2000
SHAFT_FREQ_HZ = SHAFT_RPM / 60  # 33.33 Hz

# IMS 베어링 결함 주파수 (Hz)
DEFECT_FREQUENCIES_HZ: dict[str, float] = {
    "BPFO": 236.4,   # 외륜 (Ball Pass Frequency Outer)
    "BPFI": 296.9,   # 내륜 (Ball Pass Frequency Inner)
    "BSF": 141.1,    # 전동체 (Ball Spin Frequency)
    "FTF": 14.8,     # 보지기 (Fundamental Train Frequency)
}

# 주파수 분석 파라미터
FREQ_TOLERANCE_HZ = 2.0
HIGH_FREQ_BAND: tuple[float, float] = (5000.0, 10000.0)

# 사이드밴드 감지 파라미터
SIDEBAND_SEARCH_COUNT = 10
SIDEBAND_AMPLITUDE_RATIO = 0.3

# Health Index 설정
HI_BASELINE_SNAPSHOT_COUNT: int = 20

# 개별 HI 상한 클램핑 (베이스라인 대비 N배 이상 이탈은 동일하게 최대 취급)
HI_CLAMP_MAX: float = 10.0

HI_FEATURE_KEYS: dict[str, str] = {
    "hi_rms": "rms",
    "hi_kurtosis": "kurtosis",
    "hi_crest_factor": "crest_factor",
    "hi_fft_energy": "spectral_energy_total",
}

# 가중치 근거 (P-F curve 기반):
#   RMS(0.35)          — 전 P-F 구간에서 단조 증가. 가장 범용적인 열화 지표
#   Kurtosis(0.30)     — 초기~중기 감지 탁월. 4단계에서 감소 가능하나 RMS/FFT Energy가 보상
#   FFT Energy(0.20)   — 광대역 에너지 증가 포착. RMS 보완
#   Crest Factor(0.15) — 초중기 전이 지표. 비단조적이므로 낮은 가중치
#
# NOTE: hi_peak_frequency(dominant_frequency_hz) 제외.
#   지배 주파수는 본질적으로 이산적(특정 주파수에 고정)이라
#   Min-Max 정규화 기반 HI 산출에 부적합. 베이스라인 spread가 0에 가까워
#   정상 상태에서도 극단적 HI 값을 생성함.
#   주파수 영역 결함 정보는 Edge → Agent 페이로드의 결함 주파수 진폭으로 전달.
HI_WEIGHTS: dict[str, float] = {
    "hi_rms": 0.35,
    "hi_kurtosis": 0.30,
    "hi_crest_factor": 0.15,
    "hi_fft_energy": 0.20,
}

# ---------------------------------------------------------------------------
# Anomaly Detection 설정
# ---------------------------------------------------------------------------

# HI → anomaly score 매핑 breakpoints (구간별 선형 보간)
ANOMALY_HI_BREAKPOINTS: list[tuple[float, float]] = [
    (1.0, 0.0),    # 베이스라인 범위 내 → 정상
    (2.0, 0.65),   # 2배 이탈 → 이상 판정 경계
    (3.5, 0.90),   # 3.5배 이탈 → critical 경계
    (5.0, 1.0),    # 5배 이탈 → 최대
]

# 개별 HI 스파이크 감지 임계값
ANOMALY_SPIKE_THRESHOLD: float = 2.0

# 이상 판정 임계값
ANOMALY_THRESHOLD: float = 0.65

# anomaly_score → health_state 분류 기준 (score 미만이면 해당 상태)
ANOMALY_HEALTH_STATE_TIERS: list[tuple[float, str]] = [
    (0.65, "normal"),
    (0.80, "watch"),
    (0.90, "warning"),
]

# 모델 식별자
ANOMALY_MODEL_ID: str = "rule_v1"

# ---------------------------------------------------------------------------
# Statistical Anomaly (Part 3b)
# ---------------------------------------------------------------------------

# z-score → anomaly score 매핑 breakpoints (구간별 선형 보간)
STAT_Z_SCORE_BREAKPOINTS: list[tuple[float, float]] = [
    (2.0, 0.0),     # z≤2 → 정상 (95% 신뢰구간 이내)
    (3.0, 0.65),    # z=3 → 이상 판정 경계 (99.7%)
    (4.0, 0.90),    # z=4 → critical 경계
    (5.0, 1.0),     # z≥5 → 최대
]

# z-score 산출 대상 피처 (수치형만, bool/int 제외)
STAT_FEATURE_KEYS: list[str] = [
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "standard_deviation",
    "shape_factor",
    "bpfo_amplitude",
    "bpfi_amplitude",
    "bsf_amplitude",
    "ftf_amplitude",
    "spectral_energy_total",
    "spectral_energy_high_freq_band",
    "dominant_frequency_hz",
]

# Combiner 가중치 (Rule : Statistical)
# Part 3c에서 model_based 추가 시 재배분
COMBINER_WEIGHTS: dict[str, float] = {
    "rule_based": 0.5,
    "statistical": 0.5,
}

# 최소 std (0 나누기 방지)
STAT_MIN_STD: float = 1e-10
