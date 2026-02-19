"""스냅샷 → 특징량 추출 오케스트레이션."""

from __future__ import annotations

import numpy as np

from edge.features import extract_frequency_domain, extract_time_domain


def extract_snapshot_features(
    snapshot: np.ndarray,
    bearing_channels: list[int],
    sampling_rate: float = 20_000,
) -> dict[str, dict[str, dict]]:
    """스냅샷 ndarray에서 특정 베어링의 채널별 특징량 추출.

    Args:
        snapshot: (samples, channels) ndarray.
        bearing_channels: 추출할 채널 인덱스 리스트.
        sampling_rate: 샘플링 주파수 (Hz).

    Returns:
        {
            "time_domain": {"ch0": {9개}, "ch1": {9개}, ...},
            "frequency_domain": {"ch0": {12개}, "ch1": {12개}, ...},
        }
    """
    time_domain: dict[str, dict] = {}
    frequency_domain: dict[str, dict] = {}

    for i, ch_idx in enumerate(bearing_channels):
        ch_key = f"ch{i}"
        signal = snapshot[:, ch_idx].astype(np.float64)
        time_domain[ch_key] = extract_time_domain(signal)
        frequency_domain[ch_key] = extract_frequency_domain(signal, sampling_rate)

    return {
        "time_domain": time_domain,
        "frequency_domain": frequency_domain,
    }


# 집계 시 max를 적용할 키 목록 (진폭, 에너지, 주파수 계열)
_MAX_KEYS = {
    "rms", "peak", "peak_to_peak", "crest_factor", "kurtosis",
    "standard_deviation", "shape_factor",
    "bpfo_amplitude", "bpfi_amplitude", "bsf_amplitude", "ftf_amplitude",
    "bpfo_harmonics_2x", "bpfi_harmonics_2x",
    "spectral_energy_total", "spectral_energy_high_freq_band",
    "dominant_frequency_hz",
    "sideband_spacing_hz", "sideband_count",
}

# any 집계 키 (bool)
_ANY_KEYS = {"sideband_presence"}

# 평균 집계 키
_MEAN_KEYS = {"skewness", "mean"}


def flatten_features(features: dict[str, dict[str, dict]]) -> dict[str, float | bool | int]:
    """다채널 특징량을 단일 dict로 집계.

    단일 채널이면 그대로 반환. 다채널이면:
    - 진폭/에너지/수치 계열 → max
    - bool (sideband_presence) → any
    - skewness, mean → 평균

    Args:
        features: extract_snapshot_features 반환값.

    Returns:
        평탄화된 특징량 dict.
    """
    # 모든 도메인의 채널 데이터를 합친다
    all_channel_dicts: list[dict] = []
    for domain in ("time_domain", "frequency_domain"):
        channel_data = features[domain]
        for ch_key in channel_data:
            all_channel_dicts.append(channel_data[ch_key])

    # 채널 수 파악 (time_domain 기준)
    channels = list(features["time_domain"].keys())
    if len(channels) == 1:
        result: dict[str, float | bool | int] = {}
        result.update(features["time_domain"][channels[0]])
        result.update(features["frequency_domain"][channels[0]])
        return result

    # 다채널 집계
    # 도메인별로 처리
    result = {}
    for domain in ("time_domain", "frequency_domain"):
        channel_data = features[domain]
        ch_keys = list(channel_data.keys())
        all_keys = channel_data[ch_keys[0]].keys()

        for key in all_keys:
            values = [channel_data[ch][key] for ch in ch_keys]

            if key in _ANY_KEYS:
                result[key] = any(values)
            elif key in _MEAN_KEYS:
                result[key] = float(np.mean(values))
            else:
                # _MAX_KEYS 및 기타 수치 → max
                result[key] = max(values)

    return result
