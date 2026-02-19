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
