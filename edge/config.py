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

HI_FEATURE_KEYS: dict[str, str] = {
    "hi_rms": "rms",
    "hi_kurtosis": "kurtosis",
    "hi_crest_factor": "crest_factor",
    "hi_peak_frequency": "dominant_frequency_hz",
    "hi_fft_energy": "spectral_energy_total",
}

# 가중치 근거 (P-F curve 기반):
#   RMS(0.30)          — 전 P-F 구간에서 단조 증가. 가장 범용적인 열화 지표
#   Kurtosis(0.25)     — 초기~중기 감지 탁월. 4단계에서 감소 가능하나 RMS/FFT Energy가 보상
#   FFT Energy(0.20)   — 광대역 에너지 증가 포착. RMS 보완
#   Crest Factor(0.15) — 초중기 전이 지표. 비단조적이므로 낮은 가중치
#   Peak Freq(0.10)    — 결함 주파수로의 이동 감지. 가변성이 크므로 최저 가중치
HI_WEIGHTS: dict[str, float] = {
    "hi_rms": 0.30,
    "hi_kurtosis": 0.25,
    "hi_crest_factor": 0.15,
    "hi_peak_frequency": 0.10,
    "hi_fft_energy": 0.20,
}
