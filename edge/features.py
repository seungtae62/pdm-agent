"""시간/주파수 영역 특징량 추출."""

from __future__ import annotations

import numpy as np
from scipy import stats

from edge.config import (
    DEFECT_FREQUENCIES_HZ,
    FREQ_TOLERANCE_HZ,
    HIGH_FREQ_BAND,
    SAMPLING_RATE_HZ,
    SHAFT_FREQ_HZ,
    SIDEBAND_AMPLITUDE_RATIO,
    SIDEBAND_SEARCH_COUNT,
)

# ---------------------------------------------------------------------------
# 시간 영역 특징량 (9개)
# ---------------------------------------------------------------------------


def extract_time_domain(signal: np.ndarray) -> dict[str, float]:
    """1D 진동 신호에서 시간 영역 특징량 9개를 추출.

    Args:
        signal: 1D ndarray (단일 채널 진동 신호).

    Returns:
        rms, peak, peak_to_peak, crest_factor, kurtosis, skewness,
        standard_deviation, mean, shape_factor 포함 dict.
    """
    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    peak_to_peak = float(np.ptp(signal))
    crest_factor = peak / rms if rms > 0 else 0.0
    kurtosis_val = float(stats.kurtosis(signal, fisher=False))
    skewness_val = float(stats.skew(signal))
    std = float(np.std(signal, ddof=1))
    mean_val = float(np.mean(signal))
    mean_abs = float(np.mean(np.abs(signal)))
    shape_factor = rms / mean_abs if mean_abs > 0 else 0.0

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis_val,
        "skewness": skewness_val,
        "standard_deviation": std,
        "mean": mean_val,
        "shape_factor": shape_factor,
    }


# ---------------------------------------------------------------------------
# 주파수 영역 유틸리티
# ---------------------------------------------------------------------------


def compute_fft(
    signal: np.ndarray, sampling_rate: float = SAMPLING_RATE_HZ
) -> tuple[np.ndarray, np.ndarray]:
    """단측 FFT 수행.

    Args:
        signal: 1D 진동 신호.
        sampling_rate: 샘플링 주파수 (Hz).

    Returns:
        (frequencies, amplitudes) — 양의 주파수 영역만.
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    amplitudes = (2.0 / n) * np.abs(fft_vals)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return frequencies, amplitudes


def find_peak_amplitude(
    freqs: np.ndarray,
    amps: np.ndarray,
    target_freq: float,
    tolerance: float = FREQ_TOLERANCE_HZ,
) -> float:
    """target_freq +/- tolerance 범위 내 최대 진폭.

    Args:
        freqs: 주파수 배열.
        amps: 진폭 배열.
        target_freq: 검색 중심 주파수 (Hz).
        tolerance: 허용 범위 (Hz).

    Returns:
        범위 내 최대 진폭. 해당 범위에 데이터가 없으면 0.0.
    """
    mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
    if not np.any(mask):
        return 0.0
    return float(np.max(amps[mask]))


def detect_sidebands(
    freqs: np.ndarray,
    amps: np.ndarray,
    center_freq: float,
    spacing: float = SHAFT_FREQ_HZ,
    search_count: int = SIDEBAND_SEARCH_COUNT,
    threshold_ratio: float = SIDEBAND_AMPLITUDE_RATIO,
) -> tuple[bool, float, int]:
    """중심 주파수 주변 사이드밴드 감지.

    center_freq +/- n*spacing (n=1..search_count) 위치에서
    메인 피크의 threshold_ratio 이상인 피크 개수를 센다.

    Args:
        freqs: 주파수 배열.
        amps: 진폭 배열.
        center_freq: 사이드밴드 중심 주파수 (Hz).
        spacing: 사이드밴드 간격 (Hz). 기본값 SHAFT_FREQ_HZ.
        search_count: 탐색할 사이드밴드 개수 (양쪽 각각).
        threshold_ratio: 메인 피크 대비 최소 비율.

    Returns:
        (presence, spacing_hz, count).
        presence: 감지된 사이드밴드 >= 2이면 True.
    """
    main_amp = find_peak_amplitude(freqs, amps, center_freq)
    if main_amp <= 0:
        return False, 0.0, 0

    threshold = main_amp * threshold_ratio
    count = 0
    for n in range(1, search_count + 1):
        for sign in (+1, -1):
            sb_freq = center_freq + sign * n * spacing
            if sb_freq < 0:
                continue
            sb_amp = find_peak_amplitude(freqs, amps, sb_freq)
            if sb_amp >= threshold:
                count += 1

    return count >= 2, spacing, count


# ---------------------------------------------------------------------------
# 주파수 영역 특징량 (12개)
# ---------------------------------------------------------------------------


def extract_frequency_domain(
    signal: np.ndarray, sampling_rate: float = SAMPLING_RATE_HZ
) -> dict[str, float | bool | int]:
    """1D 진동 신호에서 주파수 영역 특징량 12개를 추출.

    Args:
        signal: 1D ndarray (단일 채널 진동 신호).
        sampling_rate: 샘플링 주파수 (Hz).

    Returns:
        결함 주파수 진폭, 고조파, 스펙트럼 에너지, 사이드밴드 정보 포함 dict.
    """
    freqs, amps = compute_fft(signal, sampling_rate)

    # 결함 주파수 진폭
    bpfo_amp = find_peak_amplitude(freqs, amps, DEFECT_FREQUENCIES_HZ["BPFO"])
    bpfi_amp = find_peak_amplitude(freqs, amps, DEFECT_FREQUENCIES_HZ["BPFI"])
    bsf_amp = find_peak_amplitude(freqs, amps, DEFECT_FREQUENCIES_HZ["BSF"])
    ftf_amp = find_peak_amplitude(freqs, amps, DEFECT_FREQUENCIES_HZ["FTF"])

    # 2차 고조파
    bpfo_2x = find_peak_amplitude(freqs, amps, 2 * DEFECT_FREQUENCIES_HZ["BPFO"])
    bpfi_2x = find_peak_amplitude(freqs, amps, 2 * DEFECT_FREQUENCIES_HZ["BPFI"])

    # 스펙트럼 에너지
    spectral_energy_total = float(np.sum(amps ** 2))
    hi_lo, hi_hi = HIGH_FREQ_BAND
    hf_mask = (freqs >= hi_lo) & (freqs <= hi_hi)
    spectral_energy_high = float(np.sum(amps[hf_mask] ** 2))

    # 지배 주파수
    dominant_idx = int(np.argmax(amps[1:])) + 1  # DC 제외
    dominant_freq = float(freqs[dominant_idx])

    # 사이드밴드 — BPFO/BPFI/BSF 중 최대 진폭 주파수 기준
    defect_amps = {"BPFO": bpfo_amp, "BPFI": bpfi_amp, "BSF": bsf_amp}
    max_defect_key = max(defect_amps, key=defect_amps.get)  # type: ignore[arg-type]
    center_freq = DEFECT_FREQUENCIES_HZ[max_defect_key]
    sb_presence, sb_spacing, sb_count = detect_sidebands(freqs, amps, center_freq)

    return {
        "bpfo_amplitude": bpfo_amp,
        "bpfi_amplitude": bpfi_amp,
        "bsf_amplitude": bsf_amp,
        "ftf_amplitude": ftf_amp,
        "bpfo_harmonics_2x": bpfo_2x,
        "bpfi_harmonics_2x": bpfi_2x,
        "spectral_energy_total": spectral_energy_total,
        "spectral_energy_high_freq_band": spectral_energy_high,
        "dominant_frequency_hz": dominant_freq,
        "sideband_presence": sb_presence,
        "sideband_spacing_hz": sb_spacing,
        "sideband_count": sb_count,
    }
