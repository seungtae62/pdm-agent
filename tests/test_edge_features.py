"""Edge 특징량 추출 단위 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from edge.config import DEFECT_FREQUENCIES_HZ, SAMPLING_RATE_HZ, SHAFT_FREQ_HZ
from edge.features import (
    compute_fft,
    detect_sidebands,
    extract_frequency_domain,
    extract_time_domain,
    find_peak_amplitude,
)
from edge.feature_pipeline import extract_snapshot_features, flatten_features


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_sine(freq: float, amp: float = 1.0, n: int = 20_480, sr: float = 20_000) -> np.ndarray:
    """지정 주파수의 사인파 생성."""
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# 시간 영역
# ---------------------------------------------------------------------------

class TestTimeDomain:
    def test_rms_of_sine(self):
        """사인파 RMS = amplitude / sqrt(2)."""
        amp = 3.0
        signal = _make_sine(100.0, amp=amp)
        features = extract_time_domain(signal)
        expected_rms = amp / np.sqrt(2)
        assert features["rms"] == pytest.approx(expected_rms, rel=1e-3)

    def test_peak(self):
        signal = _make_sine(100.0, amp=5.0)
        features = extract_time_domain(signal)
        assert features["peak"] == pytest.approx(5.0, rel=1e-3)

    def test_peak_to_peak(self):
        signal = _make_sine(100.0, amp=2.0)
        features = extract_time_domain(signal)
        assert features["peak_to_peak"] == pytest.approx(4.0, rel=1e-2)

    def test_crest_factor_sine(self):
        """사인파 crest factor = sqrt(2)."""
        signal = _make_sine(100.0, amp=1.0)
        features = extract_time_domain(signal)
        assert features["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-2)

    def test_kurtosis_gaussian(self):
        """정규분포 kurtosis ≈ 3 (excess kurtosis=0, fisher=False)."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 100_000)
        features = extract_time_domain(signal)
        assert features["kurtosis"] == pytest.approx(3.0, abs=0.1)

    def test_feature_count(self):
        signal = _make_sine(100.0)
        features = extract_time_domain(signal)
        assert len(features) == 9

    def test_shape_factor_sine(self):
        """사인파 shape factor = rms / mean(|x|)."""
        signal = _make_sine(100.0, amp=1.0)
        features = extract_time_domain(signal)
        # 사인파: rms = 1/sqrt(2), mean(|sin|) = 2/pi
        expected = (1.0 / np.sqrt(2)) / (2.0 / np.pi)
        assert features["shape_factor"] == pytest.approx(expected, rel=1e-2)


# ---------------------------------------------------------------------------
# 주파수 영역
# ---------------------------------------------------------------------------

class TestFrequencyDomain:
    def test_bpfo_sine_peak(self):
        """BPFO 주파수 사인파 → bpfo_amplitude가 최대."""
        signal = _make_sine(DEFECT_FREQUENCIES_HZ["BPFO"], amp=1.0)
        features = extract_frequency_domain(signal)
        assert features["bpfo_amplitude"] > features["bpfi_amplitude"]
        assert features["bpfo_amplitude"] > features["bsf_amplitude"]
        assert features["bpfo_amplitude"] > features["ftf_amplitude"]

    def test_bpfo_amplitude_value(self):
        """단일 사인파 FFT 진폭 ≈ 원래 진폭."""
        amp = 2.0
        signal = _make_sine(DEFECT_FREQUENCIES_HZ["BPFO"], amp=amp)
        features = extract_frequency_domain(signal)
        assert features["bpfo_amplitude"] == pytest.approx(amp, rel=0.05)

    def test_dominant_frequency(self):
        """지배 주파수가 입력 사인파 주파수와 일치."""
        freq = 500.0
        signal = _make_sine(freq, amp=3.0)
        features = extract_frequency_domain(signal)
        assert features["dominant_frequency_hz"] == pytest.approx(freq, abs=2.0)

    def test_spectral_energy_positive(self):
        signal = _make_sine(100.0)
        features = extract_frequency_domain(signal)
        assert features["spectral_energy_total"] > 0

    def test_feature_count(self):
        signal = _make_sine(100.0)
        features = extract_frequency_domain(signal)
        assert len(features) == 12

    def test_high_freq_energy(self):
        """고주파 대역 신호 → spectral_energy_high_freq_band 증가."""
        low = _make_sine(100.0, amp=1.0)
        high = _make_sine(7000.0, amp=1.0)
        feat_low = extract_frequency_domain(low)
        feat_high = extract_frequency_domain(high)
        assert feat_high["spectral_energy_high_freq_band"] > feat_low["spectral_energy_high_freq_band"]


# ---------------------------------------------------------------------------
# 사이드밴드
# ---------------------------------------------------------------------------

class TestSideband:
    def test_sideband_detection(self):
        """BPFO + 양측 사이드밴드 합성 → sideband_presence=True."""
        center = DEFECT_FREQUENCIES_HZ["BPFO"]
        signal = _make_sine(center, amp=1.0)
        # shaft frequency 간격으로 사이드밴드 추가
        for n in range(1, 4):
            signal += _make_sine(center + n * SHAFT_FREQ_HZ, amp=0.5)
            signal += _make_sine(center - n * SHAFT_FREQ_HZ, amp=0.5)

        features = extract_frequency_domain(signal)
        assert features["sideband_presence"] is True
        assert features["sideband_count"] >= 2

    def test_no_sideband_pure_sine(self):
        """단일 사인파 → sideband_presence=False."""
        signal = _make_sine(DEFECT_FREQUENCIES_HZ["BPFO"], amp=1.0)
        features = extract_frequency_domain(signal)
        assert features["sideband_presence"] is False
        assert features["sideband_count"] == 0


# ---------------------------------------------------------------------------
# FFT 유틸리티
# ---------------------------------------------------------------------------

class TestFFTUtils:
    def test_compute_fft_shape(self):
        signal = _make_sine(100.0, n=1024)
        freqs, amps = compute_fft(signal, 20_000)
        assert len(freqs) == len(amps)
        assert len(freqs) == 1024 // 2 + 1

    def test_find_peak_amplitude(self):
        signal = _make_sine(236.4, amp=2.0)
        freqs, amps = compute_fft(signal)
        peak = find_peak_amplitude(freqs, amps, 236.4, tolerance=2.0)
        assert peak == pytest.approx(2.0, rel=0.05)

    def test_find_peak_no_match(self):
        signal = _make_sine(100.0, amp=1.0)
        freqs, amps = compute_fft(signal)
        peak = find_peak_amplitude(freqs, amps, 9999.0, tolerance=0.5)
        assert peak == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

class TestPipeline:
    def _make_snapshot(self, n_samples: int = 20_480, n_channels: int = 4) -> np.ndarray:
        """다채널 테스트 스냅샷 생성."""
        rng = np.random.default_rng(42)
        snapshot = rng.normal(0, 0.1, (n_samples, n_channels))
        # 채널 0에 BPFO 신호 추가
        t = np.arange(n_samples) / SAMPLING_RATE_HZ
        snapshot[:, 0] += np.sin(2 * np.pi * DEFECT_FREQUENCIES_HZ["BPFO"] * t)
        return snapshot

    def test_extract_snapshot_features_structure(self):
        """출력 구조: time_domain, frequency_domain 각각 채널별 dict."""
        snapshot = self._make_snapshot()
        features = extract_snapshot_features(snapshot, bearing_channels=[0])
        assert "time_domain" in features
        assert "frequency_domain" in features
        assert "ch0" in features["time_domain"]
        assert len(features["time_domain"]["ch0"]) == 9
        assert len(features["frequency_domain"]["ch0"]) == 12

    def test_extract_multi_channel(self):
        """다채널 추출 시 채널 수만큼 키 생성."""
        snapshot = self._make_snapshot()
        features = extract_snapshot_features(snapshot, bearing_channels=[0, 1])
        assert "ch0" in features["time_domain"]
        assert "ch1" in features["time_domain"]

    def test_flatten_single_channel(self):
        """단일 채널 flatten → 그대로 반환."""
        snapshot = self._make_snapshot()
        features = extract_snapshot_features(snapshot, bearing_channels=[0])
        flat = flatten_features(features)
        assert "rms" in flat
        assert "bpfo_amplitude" in flat
        assert len(flat) == 21  # 9 + 12

    def test_flatten_multi_channel_max(self):
        """다채널 flatten → rms는 max 집계."""
        snapshot = self._make_snapshot()
        # 채널 0에 큰 신호, 채널 1에 작은 신호
        features = extract_snapshot_features(snapshot, bearing_channels=[0, 1])
        flat = flatten_features(features)

        ch0_rms = features["time_domain"]["ch0"]["rms"]
        ch1_rms = features["time_domain"]["ch1"]["rms"]
        assert flat["rms"] == max(ch0_rms, ch1_rms)

    def test_flatten_multi_channel_bool_any(self):
        """다채널 flatten → sideband_presence는 any 집계."""
        snapshot = self._make_snapshot()
        features = extract_snapshot_features(snapshot, bearing_channels=[0, 1])
        flat = flatten_features(features)
        assert isinstance(flat["sideband_presence"], bool)

    def test_flatten_multi_channel_mean(self):
        """다채널 flatten → skewness는 평균 집계."""
        snapshot = self._make_snapshot()
        features = extract_snapshot_features(snapshot, bearing_channels=[0, 1])
        flat = flatten_features(features)

        ch0_skew = features["time_domain"]["ch0"]["skewness"]
        ch1_skew = features["time_domain"]["ch1"]["skewness"]
        expected = (ch0_skew + ch1_skew) / 2
        assert flat["skewness"] == pytest.approx(expected, rel=1e-6)
