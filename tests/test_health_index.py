"""Health Index 산출 단위 테스트."""

from __future__ import annotations

import pytest

from edge.config import HI_FEATURE_KEYS, HI_WEIGHTS
from edge.health_index import (
    BaselineStats,
    HealthIndexResult,
    compute_baseline,
    compute_health_indices,
    normalize_hi,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

_FLAT_DEFAULTS: dict[str, float | bool | int] = {
    # 시간 영역 (9개)
    "rms": 0.10,
    "peak": 0.20,
    "peak_to_peak": 0.40,
    "crest_factor": 2.0,
    "kurtosis": 3.0,
    "skewness": 0.0,
    "standard_deviation": 0.10,
    "mean": 0.0,
    "shape_factor": 1.25,
    # 주파수 영역 (12개)
    "bpfo_amplitude": 0.01,
    "bpfi_amplitude": 0.01,
    "bsf_amplitude": 0.01,
    "ftf_amplitude": 0.01,
    "bpfo_harmonics_2x": 0.005,
    "bpfi_harmonics_2x": 0.005,
    "spectral_energy_total": 1.0,
    "spectral_energy_high_freq_band": 0.1,
    "dominant_frequency_hz": 33.0,
    "sideband_presence": False,
    "sideband_spacing_hz": 0.0,
    "sideband_count": 0,
}


def _make_flat_features(**overrides: float | bool | int) -> dict[str, float | bool | int]:
    """21개 피처의 기본값 + 선택적 오버라이드."""
    result = dict(_FLAT_DEFAULTS)
    result.update(overrides)
    return result


def _make_baseline(
    min_overrides: dict[str, float] | None = None,
    max_overrides: dict[str, float] | None = None,
    snapshot_count: int = 20,
) -> BaselineStats:
    """테스트용 BaselineStats 생성."""
    min_vals = {
        "hi_rms": 0.08,
        "hi_kurtosis": 2.8,
        "hi_crest_factor": 1.8,
        "hi_fft_energy": 0.8,
    }
    max_vals = {
        "hi_rms": 0.12,
        "hi_kurtosis": 3.2,
        "hi_crest_factor": 2.2,
        "hi_fft_energy": 1.2,
    }
    if min_overrides:
        min_vals.update(min_overrides)
    if max_overrides:
        max_vals.update(max_overrides)
    return BaselineStats(
        min_values=min_vals, max_values=max_vals, snapshot_count=snapshot_count
    )


# ---------------------------------------------------------------------------
# normalize_hi
# ---------------------------------------------------------------------------


class TestNormalizeHI:
    def test_min_returns_zero(self):
        assert normalize_hi(10.0, 10.0, 20.0) == 0.0

    def test_max_returns_one(self):
        assert normalize_hi(20.0, 10.0, 20.0) == 1.0

    def test_midpoint(self):
        assert normalize_hi(15.0, 10.0, 20.0) == pytest.approx(0.5)

    def test_below_min_clamped_to_zero(self):
        assert normalize_hi(5.0, 10.0, 20.0) == 0.0

    def test_above_max_exceeds_one(self):
        result = normalize_hi(30.0, 10.0, 20.0)
        assert result == pytest.approx(2.0)
        assert result > 1.0

    def test_clamp_max_default(self):
        """극단값은 HI_CLAMP_MAX(10.0)로 클램핑."""
        result = normalize_hi(1000.0, 10.0, 20.0)
        assert result == pytest.approx(10.0)

    def test_clamp_max_custom(self):
        """커스텀 clamp_max 적용."""
        result = normalize_hi(1000.0, 10.0, 20.0, clamp_max=5.0)
        assert result == pytest.approx(5.0)

    def test_clamp_max_not_applied_within_range(self):
        """클램핑 범위 내 값은 영향 없음."""
        result = normalize_hi(15.0, 10.0, 20.0)
        assert result == pytest.approx(0.5)

    def test_degenerate_clamp_max(self):
        """min==max 퇴화 케이스에서도 클램핑 적용."""
        result = normalize_hi(100.0, 5.0, 5.0)
        assert result == pytest.approx(10.0)  # (100-5)/5 = 19 → clamped to 10

    def test_degenerate_zero_baseline_clamp(self):
        """min==max==0에서 극단값 클램핑."""
        result = normalize_hi(100.0, 0.0, 0.0)
        assert result == pytest.approx(10.0)

    def test_degenerate_equal_to_baseline(self):
        """min==max이고 값이 같으면 0.0."""
        assert normalize_hi(5.0, 5.0, 5.0) == 0.0

    def test_degenerate_above_baseline(self):
        """min==max이고 값이 초과하면 상대 편차."""
        result = normalize_hi(10.0, 5.0, 5.0)
        assert result == pytest.approx(1.0)  # (10-5)/5

    def test_degenerate_below_baseline(self):
        """min==max이고 값이 미만이면 0 클램핑."""
        assert normalize_hi(3.0, 5.0, 5.0) == 0.0

    def test_degenerate_zero_baseline(self):
        """min==max==0이고 값이 양수이면 값 그대로."""
        assert normalize_hi(2.5, 0.0, 0.0) == 2.5

    def test_degenerate_zero_baseline_zero_value(self):
        """min==max==0이고 값도 0이면 0."""
        assert normalize_hi(0.0, 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_baseline
# ---------------------------------------------------------------------------


class TestComputeBaseline:
    def test_min_max_accuracy(self):
        """min/max가 정확하게 산출되는지."""
        dicts = []
        for i in range(20):
            dicts.append(_make_flat_features(rms=0.08 + i * 0.002))

        baseline = compute_baseline(dicts)
        assert baseline.min_values["hi_rms"] == pytest.approx(0.08)
        assert baseline.max_values["hi_rms"] == pytest.approx(0.08 + 19 * 0.002)

    def test_uses_only_first_n(self):
        """N개를 초과하는 데이터가 있어도 첫 N개만 사용."""
        dicts = [_make_flat_features(rms=0.10) for _ in range(20)]
        # 21번째에 극단값 추가
        dicts.append(_make_flat_features(rms=999.0))

        baseline = compute_baseline(dicts, n_snapshots=20)
        assert baseline.max_values["hi_rms"] == pytest.approx(0.10)

    def test_insufficient_snapshots_raises(self):
        """스냅샷 부족 시 ValueError."""
        dicts = [_make_flat_features() for _ in range(5)]
        with pytest.raises(ValueError, match="20개 스냅샷이 필요"):
            compute_baseline(dicts)

    def test_all_hi_keys_present(self):
        """모든 HI 키가 baseline에 존재."""
        dicts = [_make_flat_features() for _ in range(20)]
        baseline = compute_baseline(dicts)
        for hi_key in HI_FEATURE_KEYS:
            assert hi_key in baseline.min_values
            assert hi_key in baseline.max_values

    def test_snapshot_count_stored(self):
        dicts = [_make_flat_features() for _ in range(25)]
        baseline = compute_baseline(dicts, n_snapshots=25)
        assert baseline.snapshot_count == 25

    def test_custom_n_snapshots(self):
        """커스텀 n_snapshots 동작."""
        dicts = [_make_flat_features(rms=0.05 + i * 0.01) for i in range(10)]
        baseline = compute_baseline(dicts, n_snapshots=5)
        assert baseline.min_values["hi_rms"] == pytest.approx(0.05)
        assert baseline.max_values["hi_rms"] == pytest.approx(0.05 + 4 * 0.01)


# ---------------------------------------------------------------------------
# compute_health_indices
# ---------------------------------------------------------------------------


class TestComputeHealthIndices:
    def test_all_at_min_gives_zero(self):
        """모든 피처가 baseline min일 때 individual=0, composite=0."""
        baseline = _make_baseline()
        flat = _make_flat_features(
            rms=baseline.min_values["hi_rms"],
            kurtosis=baseline.min_values["hi_kurtosis"],
            crest_factor=baseline.min_values["hi_crest_factor"],
            spectral_energy_total=baseline.min_values["hi_fft_energy"],
        )
        result = compute_health_indices(flat, baseline)
        for v in result.individual.values():
            assert v == pytest.approx(0.0)
        assert result.composite == pytest.approx(0.0)

    def test_all_at_max_gives_one(self):
        """모든 피처가 baseline max일 때 individual=1, composite=1."""
        baseline = _make_baseline()
        flat = _make_flat_features(
            rms=baseline.max_values["hi_rms"],
            kurtosis=baseline.max_values["hi_kurtosis"],
            crest_factor=baseline.max_values["hi_crest_factor"],
            spectral_energy_total=baseline.max_values["hi_fft_energy"],
        )
        result = compute_health_indices(flat, baseline)
        for v in result.individual.values():
            assert v == pytest.approx(1.0)
        assert result.composite == pytest.approx(1.0)

    def test_rms_only_exceeds(self):
        """RMS만 초과 시 composite = rms weight × hi_rms + 나머지 min 기여."""
        baseline = _make_baseline()
        # RMS를 max의 2배로 설정 → hi_rms > 1
        rms_exceed = baseline.min_values["hi_rms"] + 2 * (
            baseline.max_values["hi_rms"] - baseline.min_values["hi_rms"]
        )
        flat = _make_flat_features(
            rms=rms_exceed,
            kurtosis=baseline.min_values["hi_kurtosis"],
            crest_factor=baseline.min_values["hi_crest_factor"],
            spectral_energy_total=baseline.min_values["hi_fft_energy"],
        )
        result = compute_health_indices(flat, baseline)
        assert result.individual["hi_rms"] == pytest.approx(2.0)
        assert result.composite == pytest.approx(HI_WEIGHTS["hi_rms"] * 2.0)

    def test_composite_is_weighted_sum(self):
        """composite = Σ(weight_i × hi_i)."""
        baseline = _make_baseline()
        flat = _make_flat_features(
            rms=baseline.max_values["hi_rms"],
            kurtosis=baseline.max_values["hi_kurtosis"],
            crest_factor=baseline.min_values["hi_crest_factor"],
            spectral_energy_total=baseline.max_values["hi_fft_energy"],
        )
        result = compute_health_indices(flat, baseline)
        # hi_rms=1, hi_kurtosis=1, hi_crest_factor=0, hi_fft_energy=1
        expected = (
            HI_WEIGHTS["hi_rms"] * 1.0
            + HI_WEIGHTS["hi_kurtosis"] * 1.0
            + HI_WEIGHTS["hi_crest_factor"] * 0.0
            + HI_WEIGHTS["hi_fft_energy"] * 1.0
        )
        assert result.composite == pytest.approx(expected)

    def test_custom_weights(self):
        """커스텀 가중치 적용."""
        baseline = _make_baseline()
        flat = _make_flat_features(
            rms=baseline.max_values["hi_rms"],
            kurtosis=baseline.max_values["hi_kurtosis"],
            crest_factor=baseline.max_values["hi_crest_factor"],
            spectral_energy_total=baseline.max_values["hi_fft_energy"],
        )
        custom_weights = {k: 0.25 for k in HI_WEIGHTS}
        result = compute_health_indices(flat, baseline, weights=custom_weights)
        # all HI = 1.0, equal weights of 0.20 each → composite = 1.0
        assert result.composite == pytest.approx(1.0)

    def test_result_type(self):
        """반환 타입 확인."""
        baseline = _make_baseline()
        flat = _make_flat_features()
        result = compute_health_indices(flat, baseline)
        assert isinstance(result, HealthIndexResult)
        assert isinstance(result.individual, dict)
        assert isinstance(result.composite, float)


# ---------------------------------------------------------------------------
# 통합 테스트
# ---------------------------------------------------------------------------


class TestHealthIndexIntegration:
    def test_baseline_to_hi_roundtrip(self):
        """baseline 산출 → HI 계산 라운드트립."""
        dicts = [_make_flat_features(rms=0.10, kurtosis=3.0) for _ in range(20)]
        baseline = compute_baseline(dicts)

        # 베이스라인과 동일한 피처 → HI ≈ 정상 범위
        flat = _make_flat_features(rms=0.10, kurtosis=3.0)
        result = compute_health_indices(flat, baseline)
        # 모든 값이 동일하면 min==max → 퇴화 케이스, 결과 0
        for v in result.individual.values():
            assert v >= 0.0

    def test_progressive_degradation_monotonic(self):
        """점진적 열화 시 composite HI 단조 증가."""
        # 베이스라인: rms 0.08~0.12 범위
        baseline_dicts = []
        for i in range(20):
            baseline_dicts.append(
                _make_flat_features(
                    rms=0.08 + i * 0.002,
                    kurtosis=2.8 + i * 0.02,
                    crest_factor=1.8 + i * 0.02,
                    dominant_frequency_hz=30.0 + i * 0.3,
                    spectral_energy_total=0.8 + i * 0.02,
                )
            )
        baseline = compute_baseline(baseline_dicts)

        # 점진적 열화 시뮬레이션
        composites = []
        for stage in range(5):
            factor = 1.0 + stage * 0.5  # 1.0, 1.5, 2.0, 2.5, 3.0
            flat = _make_flat_features(
                rms=0.12 * factor,
                kurtosis=3.2 * factor,
                crest_factor=2.2 * factor,
                dominant_frequency_hz=36.0 * factor,
                spectral_energy_total=1.2 * factor,
            )
            result = compute_health_indices(flat, baseline)
            composites.append(result.composite)

        # 단조 증가 확인
        for i in range(len(composites) - 1):
            assert composites[i + 1] > composites[i], (
                f"composite[{i+1}]={composites[i+1]:.4f} <= "
                f"composite[{i}]={composites[i]:.4f}"
            )

    def test_healthy_vs_degraded(self):
        """정상 스냅샷 vs 열화 스냅샷 비교."""
        baseline_dicts = [
            _make_flat_features(
                rms=0.09 + i * 0.001,
                kurtosis=2.9 + i * 0.01,
                spectral_energy_total=0.9 + i * 0.01,
            )
            for i in range(20)
        ]
        baseline = compute_baseline(baseline_dicts)

        healthy = _make_flat_features(
            rms=0.10, kurtosis=3.0, spectral_energy_total=1.0
        )
        degraded = _make_flat_features(
            rms=0.50, kurtosis=8.0, spectral_energy_total=5.0
        )

        hi_healthy = compute_health_indices(healthy, baseline)
        hi_degraded = compute_health_indices(degraded, baseline)

        assert hi_degraded.composite > hi_healthy.composite
        assert hi_degraded.individual["hi_rms"] > hi_healthy.individual["hi_rms"]
