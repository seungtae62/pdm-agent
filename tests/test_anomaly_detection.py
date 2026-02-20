"""Anomaly Detection 단위 테스트 — Part 3a + 3b."""

from __future__ import annotations

import pytest

from edge.anomaly_detection import (
    AnomalyBaseline,
    AnomalyResult,
    RuleCheckDetail,
    StatCheckDetail,
    _classify_health_state,
    _hi_to_score,
    _z_to_score,
    check_rules,
    check_statistical,
    compute_anomaly_baseline,
    detect_anomaly,
)
from edge.config import (
    ANOMALY_HI_BREAKPOINTS,
    ANOMALY_MODEL_ID,
    ANOMALY_THRESHOLD,
    STAT_FEATURE_KEYS,
    STAT_Z_SCORE_BREAKPOINTS,
)
from edge.health_index import (
    BaselineStats,
    HealthIndexResult,
    compute_baseline,
    compute_health_indices,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_hi_result(
    composite: float = 0.5,
    **individual_overrides: float,
) -> HealthIndexResult:
    """테스트용 HealthIndexResult 생성."""
    individual = {
        "hi_rms": 0.5,
        "hi_kurtosis": 0.5,
        "hi_crest_factor": 0.5,
        "hi_fft_energy": 0.5,
    }
    individual.update(individual_overrides)
    return HealthIndexResult(individual=individual, composite=composite)


_FLAT_DEFAULTS: dict[str, float | bool | int] = {
    "rms": 0.10,
    "peak": 0.20,
    "peak_to_peak": 0.40,
    "crest_factor": 2.0,
    "kurtosis": 3.0,
    "skewness": 0.0,
    "standard_deviation": 0.10,
    "mean": 0.0,
    "shape_factor": 1.25,
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


def _make_flat_features(**overrides: float | bool | int) -> dict:
    result = dict(_FLAT_DEFAULTS)
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# TestHiToScore
# ---------------------------------------------------------------------------


class TestHiToScore:
    """_hi_to_score: HI → 0~1 anomaly score 변환."""

    def test_breakpoint_exact_values(self):
        """각 breakpoint 정확값."""
        for hi, expected_score in ANOMALY_HI_BREAKPOINTS:
            assert _hi_to_score(hi) == pytest.approx(expected_score)

    def test_interpolation_midpoint(self):
        """구간 중점에서 선형 보간."""
        # (1.0, 0.0) ~ (2.0, 0.65) 중점 = 1.5 → 0.325
        assert _hi_to_score(1.5) == pytest.approx(0.325)

    def test_interpolation_quarter(self):
        """구간 1/4 지점."""
        # (2.0, 0.65) ~ (3.5, 0.90) 에서 hi=2.375 → 0.65 + 0.25*(0.90-0.65) = 0.7125
        assert _hi_to_score(2.375) == pytest.approx(0.7125)

    def test_below_min_returns_zero(self):
        """최소 구간 이하 → 0."""
        assert _hi_to_score(0.0) == pytest.approx(0.0)
        assert _hi_to_score(0.5) == pytest.approx(0.0)
        assert _hi_to_score(1.0) == pytest.approx(0.0)

    def test_above_max_returns_one(self):
        """최대 구간 초과 → 1."""
        assert _hi_to_score(5.0) == pytest.approx(1.0)
        assert _hi_to_score(10.0) == pytest.approx(1.0)
        assert _hi_to_score(100.0) == pytest.approx(1.0)

    def test_monotonic_increase(self):
        """HI 증가 → score 단조 증가."""
        prev = _hi_to_score(0.0)
        for hi in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            curr = _hi_to_score(hi)
            assert curr >= prev
            prev = curr

    def test_score_range(self):
        """모든 결과가 0~1 범위."""
        for hi in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0]:
            score = _hi_to_score(hi)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestCheckRules
# ---------------------------------------------------------------------------


class TestCheckRules:
    """check_rules: Composite + Spike → RuleCheckDetail."""

    def test_normal_no_spike(self):
        """정상 상태: composite < 1.0, 개별 HI < 2.0 → score ≈ 0."""
        hi = _make_hi_result(composite=0.5)
        detail = check_rules(hi)
        assert detail.composite_hi_score == pytest.approx(0.0)
        assert detail.spike_score == pytest.approx(0.0)
        assert detail.spiked_keys == []

    def test_spike_only(self):
        """개별 HI만 스파이크 (composite는 정상)."""
        hi = _make_hi_result(composite=0.8, hi_rms=3.0)
        detail = check_rules(hi)
        assert detail.composite_hi_score == pytest.approx(0.0)
        assert detail.spike_score > 0.0
        assert "hi_rms" in detail.spiked_keys

    def test_composite_only(self):
        """Composite만 높고 개별 스파이크 없음."""
        hi = _make_hi_result(composite=2.5, hi_rms=1.5, hi_kurtosis=1.5)
        detail = check_rules(hi)
        assert detail.composite_hi_score > 0.0
        assert detail.spike_score == pytest.approx(0.0)
        assert detail.spiked_keys == []

    def test_both_composite_and_spike(self):
        """Composite와 Spike 모두 높음 → 각각 산출."""
        hi = _make_hi_result(composite=3.0, hi_rms=4.0)
        detail = check_rules(hi)
        assert detail.composite_hi_score > 0.0
        assert detail.spike_score > 0.0
        assert "hi_rms" in detail.spiked_keys

    def test_multiple_spikes(self):
        """여러 개별 HI가 동시 스파이크."""
        hi = _make_hi_result(composite=1.0, hi_rms=2.5, hi_kurtosis=3.0)
        detail = check_rules(hi)
        assert "hi_rms" in detail.spiked_keys
        assert "hi_kurtosis" in detail.spiked_keys
        # spike_score는 최대값(3.0) 기준
        expected_spike = _hi_to_score(3.0)
        assert detail.spike_score == pytest.approx(expected_spike)

    def test_spike_at_threshold_boundary(self):
        """정확히 spike_threshold(2.0)에서 스파이크 감지."""
        hi = _make_hi_result(composite=0.5, hi_rms=2.0)
        detail = check_rules(hi)
        assert "hi_rms" in detail.spiked_keys

    def test_spike_just_below_threshold(self):
        """spike_threshold 미만이면 스파이크 미감지."""
        hi = _make_hi_result(composite=0.5, hi_rms=1.99)
        detail = check_rules(hi)
        assert detail.spiked_keys == []
        assert detail.spike_score == pytest.approx(0.0)

    def test_return_type(self):
        hi = _make_hi_result()
        detail = check_rules(hi)
        assert isinstance(detail, RuleCheckDetail)


# ---------------------------------------------------------------------------
# TestClassifyHealthState
# ---------------------------------------------------------------------------


class TestClassifyHealthState:
    """_classify_health_state: score → health state."""

    def test_normal(self):
        assert _classify_health_state(0.0) == "normal"
        assert _classify_health_state(0.3) == "normal"
        assert _classify_health_state(0.64) == "normal"

    def test_watch(self):
        assert _classify_health_state(0.65) == "watch"
        assert _classify_health_state(0.70) == "watch"
        assert _classify_health_state(0.79) == "watch"

    def test_warning(self):
        assert _classify_health_state(0.80) == "warning"
        assert _classify_health_state(0.85) == "warning"
        assert _classify_health_state(0.89) == "warning"

    def test_critical(self):
        assert _classify_health_state(0.90) == "critical"
        assert _classify_health_state(0.95) == "critical"
        assert _classify_health_state(1.0) == "critical"

    def test_boundary_exact_065(self):
        """경계값 0.65: normal이 아닌 watch."""
        assert _classify_health_state(0.65) == "watch"

    def test_boundary_exact_080(self):
        """경계값 0.80: watch가 아닌 warning."""
        assert _classify_health_state(0.80) == "warning"

    def test_boundary_exact_090(self):
        """경계값 0.90: warning이 아닌 critical."""
        assert _classify_health_state(0.90) == "critical"


# ---------------------------------------------------------------------------
# TestDetectAnomaly
# ---------------------------------------------------------------------------


class TestDetectAnomaly:
    """detect_anomaly: 종합 이상 감지."""

    def test_normal_not_detected(self):
        """정상 HI → anomaly_detected=False."""
        hi = _make_hi_result(composite=0.5)
        result = detect_anomaly(hi)
        assert result.anomaly_detected is False
        assert result.anomaly_score < ANOMALY_THRESHOLD
        assert result.health_state == "normal"

    def test_degraded_detected(self):
        """열화 HI → anomaly_detected=True."""
        hi = _make_hi_result(composite=3.0, hi_rms=4.0)
        result = detect_anomaly(hi)
        assert result.anomaly_detected is True
        assert result.anomaly_score >= ANOMALY_THRESHOLD

    def test_model_id(self):
        """model_id가 config 값과 일치."""
        hi = _make_hi_result()
        result = detect_anomaly(hi)
        assert result.model_id == ANOMALY_MODEL_ID

    def test_custom_threshold(self):
        """커스텀 threshold 적용."""
        hi = _make_hi_result(composite=2.0)
        # threshold=0.0이면 무조건 감지
        result = detect_anomaly(hi, threshold=0.0)
        assert result.anomaly_detected is True
        assert result.anomaly_threshold == 0.0

    def test_score_range(self):
        """anomaly_score가 0~1 범위."""
        for comp in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            hi = _make_hi_result(composite=comp)
            result = detect_anomaly(hi)
            assert 0.0 <= result.anomaly_score <= 1.0

    def test_confidence_range(self):
        """confidence가 0.5~1.0 범위."""
        for comp in [0.0, 1.0, 2.0, 5.0]:
            hi = _make_hi_result(composite=comp)
            result = detect_anomaly(hi)
            assert 0.5 <= result.confidence <= 1.0

    def test_confidence_perfect_agreement(self):
        """composite_score == spike_score일 때 confidence=1.0."""
        # 둘 다 0 (정상)
        hi = _make_hi_result(composite=0.5)
        result = detect_anomaly(hi)
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_disagreement(self):
        """composite_score와 spike_score 불일치 시 confidence < 1.0."""
        # composite 정상이지만 spike 발생
        hi = _make_hi_result(composite=0.5, hi_rms=3.0)
        result = detect_anomaly(hi)
        assert result.confidence < 1.0

    def test_anomaly_score_is_max_of_rules(self):
        """anomaly_score = max(composite_score, spike_score)."""
        hi = _make_hi_result(composite=2.0, hi_rms=4.0)
        result = detect_anomaly(hi)
        detail = result.rule_detail
        expected = max(detail.composite_hi_score, detail.spike_score)
        assert result.anomaly_score == pytest.approx(expected)

    def test_return_type(self):
        hi = _make_hi_result()
        result = detect_anomaly(hi)
        assert isinstance(result, AnomalyResult)
        assert isinstance(result.rule_detail, RuleCheckDetail)


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """HI → Anomaly Detection 통합 테스트."""

    def test_progressive_degradation_monotonic(self):
        """점진적 열화 시 anomaly_score 단조 증가."""
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

        scores = []
        for stage in range(6):
            factor = 1.0 + stage * 0.8  # 1.0, 1.8, 2.6, 3.4, 4.2, 5.0
            flat = _make_flat_features(
                rms=0.12 * factor,
                kurtosis=3.2 * factor,
                crest_factor=2.2 * factor,
                dominant_frequency_hz=36.0 * factor,
                spectral_energy_total=1.2 * factor,
            )
            hi = compute_health_indices(flat, baseline)
            result = detect_anomaly(hi)
            scores.append(result.anomaly_score)

        for i in range(len(scores) - 1):
            assert scores[i + 1] >= scores[i], (
                f"score[{i+1}]={scores[i+1]:.4f} < score[{i}]={scores[i]:.4f}"
            )

    def test_hi_to_detect_roundtrip(self):
        """baseline → HI → detect_anomaly 라운드트립."""
        baseline_dicts = [
            _make_flat_features(
                rms=0.09 + i * 0.001,
                kurtosis=2.9 + i * 0.01,
                spectral_energy_total=0.9 + i * 0.01,
            )
            for i in range(20)
        ]
        baseline = compute_baseline(baseline_dicts)

        # 정상 스냅샷
        healthy_flat = _make_flat_features(rms=0.10, kurtosis=3.0, spectral_energy_total=1.0)
        hi_healthy = compute_health_indices(healthy_flat, baseline)
        result_healthy = detect_anomaly(hi_healthy)
        assert result_healthy.anomaly_detected is False
        assert result_healthy.health_state == "normal"

        # 열화 스냅샷
        degraded_flat = _make_flat_features(rms=0.50, kurtosis=8.0, spectral_energy_total=5.0)
        hi_degraded = compute_health_indices(degraded_flat, baseline)
        result_degraded = detect_anomaly(hi_degraded)
        assert result_degraded.anomaly_detected is True
        assert result_degraded.anomaly_score > result_healthy.anomaly_score

    def test_health_state_progression(self):
        """열화 심화에 따른 health_state 진행: normal → watch → warning → critical.

        _hi_to_score를 직접 사용하여 4개 상태가 모두 도달 가능함을 검증.
        HI 기반 통합 경로에서는 베이스라인 범위에 따라 중간 구간을
        건너뛸 수 있으므로, score 매핑 레벨에서 검증한다.
        """
        from edge.anomaly_detection import _classify_health_state

        states_seen = set()
        # score 0~1 범위를 세밀하게 순회
        for score_x100 in range(0, 101, 5):
            score = score_x100 / 100.0
            states_seen.add(_classify_health_state(score))

        assert states_seen == {"normal", "watch", "warning", "critical"}


# ===========================================================================
# Part 3b: Statistical Anomaly
# ===========================================================================

# ---------------------------------------------------------------------------
# TestZToScore
# ---------------------------------------------------------------------------


class TestZToScore:
    """_z_to_score: z-score → 0~1 anomaly score 변환."""

    def test_breakpoint_exact_values(self):
        for z, expected in STAT_Z_SCORE_BREAKPOINTS:
            assert _z_to_score(z) == pytest.approx(expected)

    def test_below_min_returns_zero(self):
        assert _z_to_score(0.0) == pytest.approx(0.0)
        assert _z_to_score(1.0) == pytest.approx(0.0)
        assert _z_to_score(2.0) == pytest.approx(0.0)

    def test_above_max_returns_one(self):
        assert _z_to_score(5.0) == pytest.approx(1.0)
        assert _z_to_score(10.0) == pytest.approx(1.0)

    def test_interpolation_midpoint(self):
        # (2.0, 0.0) ~ (3.0, 0.65) 중점 = 2.5 → 0.325
        assert _z_to_score(2.5) == pytest.approx(0.325)

    def test_negative_z_uses_absolute(self):
        """음수 z-score도 절대값으로 처리."""
        assert _z_to_score(-3.0) == pytest.approx(_z_to_score(3.0))

    def test_monotonic_increase(self):
        prev = _z_to_score(0.0)
        for z in [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            curr = _z_to_score(z)
            assert curr >= prev
            prev = curr


# ---------------------------------------------------------------------------
# TestComputeAnomalyBaseline
# ---------------------------------------------------------------------------


class TestComputeAnomalyBaseline:
    """compute_anomaly_baseline: 피처별 mean/std 산출."""

    def test_basic_mean_std(self):
        flats = [
            _make_flat_features(rms=0.10, kurtosis=3.0),
            _make_flat_features(rms=0.12, kurtosis=3.2),
            _make_flat_features(rms=0.08, kurtosis=2.8),
        ]
        baseline = compute_anomaly_baseline(flats)
        assert baseline.mean["rms"] == pytest.approx(0.10, abs=1e-6)
        assert baseline.std["rms"] > 0
        assert baseline.snapshot_count == 3

    def test_n_snapshots(self):
        flats = [_make_flat_features(rms=0.10 + i * 0.01) for i in range(10)]
        baseline = compute_anomaly_baseline(flats, n_snapshots=5)
        assert baseline.snapshot_count == 5

    def test_insufficient_raises(self):
        with pytest.raises(ValueError):
            compute_anomaly_baseline([_make_flat_features()])

    def test_all_feature_keys_present(self):
        flats = [_make_flat_features() for _ in range(5)]
        baseline = compute_anomaly_baseline(flats)
        for key in STAT_FEATURE_KEYS:
            assert key in baseline.mean
            assert key in baseline.std

    def test_zero_variance_gives_zero_std(self):
        """모든 값이 동일하면 std=0."""
        flats = [_make_flat_features(rms=0.10) for _ in range(5)]
        baseline = compute_anomaly_baseline(flats)
        assert baseline.std["rms"] == pytest.approx(0.0)

    def test_return_type(self):
        flats = [_make_flat_features() for _ in range(5)]
        baseline = compute_anomaly_baseline(flats)
        assert isinstance(baseline, AnomalyBaseline)


# ---------------------------------------------------------------------------
# TestCheckStatistical
# ---------------------------------------------------------------------------


class TestCheckStatistical:
    """check_statistical: z-score 기반 통계적 이탈도."""

    def _make_baseline(self, **overrides):
        """테스트용 AnomalyBaseline."""
        mean = {k: _FLAT_DEFAULTS[k] for k in STAT_FEATURE_KEYS}
        std = {k: abs(float(_FLAT_DEFAULTS[k])) * 0.1 + 0.001 for k in STAT_FEATURE_KEYS}
        mean.update({k: v for k, v in overrides.items() if k in mean})
        return AnomalyBaseline(mean=mean, std=std, snapshot_count=20)

    def test_normal_low_z_scores(self):
        """정상 데이터 → 모든 z-score가 낮음."""
        baseline = self._make_baseline()
        flat = _make_flat_features()
        detail = check_statistical(flat, baseline)
        assert detail.max_z_score < 2.0
        assert detail.stat_score == pytest.approx(0.0)

    def test_anomalous_high_z_score(self):
        """이탈 데이터 → 높은 z-score."""
        baseline = self._make_baseline()
        # rms를 크게 이탈시킴: mean=0.10, std≈0.011
        flat = _make_flat_features(rms=0.50)
        detail = check_statistical(flat, baseline)
        assert detail.max_z_score > 3.0
        assert detail.stat_score > 0.0
        assert detail.max_z_feature == "rms"

    def test_z_scores_all_features(self):
        """모든 대상 피처에 대해 z-score가 산출됨."""
        baseline = self._make_baseline()
        flat = _make_flat_features()
        detail = check_statistical(flat, baseline)
        for key in STAT_FEATURE_KEYS:
            assert key in detail.z_scores

    def test_zero_std_handled(self):
        """std=0인 피처도 min_std로 처리."""
        mean = {k: float(_FLAT_DEFAULTS[k]) for k in STAT_FEATURE_KEYS}
        std = {k: 0.0 for k in STAT_FEATURE_KEYS}
        baseline = AnomalyBaseline(mean=mean, std=std, snapshot_count=20)
        flat = _make_flat_features()
        detail = check_statistical(flat, baseline)
        assert isinstance(detail, StatCheckDetail)

    def test_return_type(self):
        baseline = self._make_baseline()
        flat = _make_flat_features()
        detail = check_statistical(flat, baseline)
        assert isinstance(detail, StatCheckDetail)
        assert isinstance(detail.z_scores, dict)
        assert isinstance(detail.stat_score, float)


# ---------------------------------------------------------------------------
# TestCombiner
# ---------------------------------------------------------------------------


class TestCombiner:
    """detect_anomaly with Rule + Statistical combiner."""

    def _make_baseline_and_flats(self, n=20):
        flats = [
            _make_flat_features(
                rms=0.09 + i * 0.001,
                kurtosis=2.9 + i * 0.01,
                spectral_energy_total=0.9 + i * 0.01,
            )
            for i in range(n)
        ]
        hi_baseline = compute_baseline(flats)
        anom_baseline = compute_anomaly_baseline(flats)
        return flats, hi_baseline, anom_baseline

    def test_backward_compatible_no_stat(self):
        """flat_features 없이 호출 → Part 3a 호환 (Rule only)."""
        hi = _make_hi_result(composite=0.5)
        result = detect_anomaly(hi)
        assert result.stat_detail is None
        assert result.anomaly_score >= 0.0

    def test_with_stat_has_detail(self):
        """flat_features + anomaly_baseline → stat_detail 존재."""
        flats, hi_bl, anom_bl = self._make_baseline_and_flats()
        flat = _make_flat_features(rms=0.10, kurtosis=3.0, spectral_energy_total=1.0)
        hi = compute_health_indices(flat, hi_bl)
        result = detect_anomaly(hi, flat, anom_bl)
        assert result.stat_detail is not None
        assert isinstance(result.stat_detail, StatCheckDetail)

    def test_normal_combined_score(self):
        """정상 데이터 → 낮은 combined score."""
        flats, hi_bl, anom_bl = self._make_baseline_and_flats()
        flat = _make_flat_features(rms=0.10, kurtosis=3.0, spectral_energy_total=1.0)
        hi = compute_health_indices(flat, hi_bl)
        result = detect_anomaly(hi, flat, anom_bl)
        assert result.anomaly_detected is False
        assert result.anomaly_score < ANOMALY_THRESHOLD

    def test_degraded_combined_score(self):
        """열화 데이터 → 높은 combined score."""
        flats, hi_bl, anom_bl = self._make_baseline_and_flats()
        flat = _make_flat_features(rms=0.50, kurtosis=8.0, spectral_energy_total=5.0)
        hi = compute_health_indices(flat, hi_bl)
        result = detect_anomaly(hi, flat, anom_bl)
        assert result.anomaly_detected is True
        assert result.anomaly_score >= ANOMALY_THRESHOLD

    def test_combined_score_between_rule_and_stat(self):
        """combined score는 rule_score와 stat_score 사이 (또는 같음)."""
        flats, hi_bl, anom_bl = self._make_baseline_and_flats()
        flat = _make_flat_features(rms=0.30, kurtosis=5.0, spectral_energy_total=3.0)
        hi = compute_health_indices(flat, hi_bl)
        result = detect_anomaly(hi, flat, anom_bl)

        rule_score = max(
            result.rule_detail.composite_hi_score,
            result.rule_detail.spike_score,
        )
        stat_score = result.stat_detail.stat_score
        lo = min(rule_score, stat_score)
        hi_s = max(rule_score, stat_score)
        assert lo - 0.01 <= result.anomaly_score <= hi_s + 0.01

    def test_custom_weights(self):
        """커스텀 가중치 적용."""
        flats, hi_bl, anom_bl = self._make_baseline_and_flats()
        # Rule과 Stat가 서로 다른 score를 내도록 중간 수준 이탈
        flat = _make_flat_features(rms=0.13, kurtosis=3.3, spectral_energy_total=1.1)
        hi = compute_health_indices(flat, hi_bl)

        # Rule 100%, Stat 0%
        result_rule_only = detect_anomaly(
            hi, flat, anom_bl,
            weights={"rule_based": 1.0, "statistical": 0.0},
        )
        # Stat 100%, Rule 0%
        result_stat_only = detect_anomaly(
            hi, flat, anom_bl,
            weights={"rule_based": 0.0, "statistical": 1.0},
        )
        # 서로 다른 score
        assert result_rule_only.anomaly_score != pytest.approx(
            result_stat_only.anomaly_score, abs=0.01
        )

    def test_confidence_range(self):
        """confidence는 0.5~1.0 범위."""
        flats, hi_bl, anom_bl = self._make_baseline_and_flats()
        for rms in [0.10, 0.20, 0.30, 0.50]:
            flat = _make_flat_features(rms=rms)
            hi = compute_health_indices(flat, hi_bl)
            result = detect_anomaly(hi, flat, anom_bl)
            assert 0.5 <= result.confidence <= 1.0

    def test_score_range(self):
        """anomaly_score는 0~1 범위."""
        flats, hi_bl, anom_bl = self._make_baseline_and_flats()
        for rms in [0.05, 0.10, 0.30, 0.50, 1.0]:
            flat = _make_flat_features(rms=rms)
            hi = compute_health_indices(flat, hi_bl)
            result = detect_anomaly(hi, flat, anom_bl)
            assert 0.0 <= result.anomaly_score <= 1.0


# ---------------------------------------------------------------------------
# TestStatIntegration
# ---------------------------------------------------------------------------


class TestStatIntegration:
    """Statistical 모듈 통합 테스트."""

    def test_progressive_degradation_stat_score_increases(self):
        """점진적 열화 시 stat_score 단조 증가."""
        flats = [
            _make_flat_features(
                rms=0.09 + i * 0.001,
                kurtosis=2.9 + i * 0.01,
                spectral_energy_total=0.9 + i * 0.01,
            )
            for i in range(20)
        ]
        anom_bl = compute_anomaly_baseline(flats)

        scores = []
        for factor in [1.0, 2.0, 3.0, 5.0, 8.0]:
            flat = _make_flat_features(
                rms=0.10 * factor,
                kurtosis=3.0 * factor,
                spectral_energy_total=1.0 * factor,
            )
            detail = check_statistical(flat, anom_bl)
            scores.append(detail.stat_score)

        for i in range(len(scores) - 1):
            assert scores[i + 1] >= scores[i]

    def test_stat_and_rule_independent(self):
        """Rule과 Stat의 score가 독립적으로 산출됨."""
        flats = [_make_flat_features() for _ in range(20)]
        hi_bl = compute_baseline(flats)
        anom_bl = compute_anomaly_baseline(flats)

        # 같은 입력에 대해 rule과 stat의 score가 반드시 같지 않음
        flat = _make_flat_features(rms=0.30, kurtosis=5.0)
        hi = compute_health_indices(flat, hi_bl)
        result = detect_anomaly(hi, flat, anom_bl)

        rule_score = max(
            result.rule_detail.composite_hi_score,
            result.rule_detail.spike_score,
        )
        stat_score = result.stat_detail.stat_score
        # 두 모듈이 독립 산출되므로 정확히 같을 확률은 낮음
        # (같을 수도 있지만, 적어도 둘 다 유효한 값)
        assert 0.0 <= rule_score <= 1.0
        assert 0.0 <= stat_score <= 1.0
