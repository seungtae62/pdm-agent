"""Autoencoder 모듈 테스트 — Part 3c."""

from __future__ import annotations

import math
import random

import pytest
import torch

from edge.autoencoder import (
    AutoencoderBaseline,
    BearingAutoencoder,
    compute_reconstruction_error,
    train_autoencoder,
    _normalize_features,
)
from edge.anomaly_detection import (
    ModelCheckDetail,
    _recon_to_score,
    detect_anomaly,
)
from edge.config import MODEL_FEATURE_KEYS, MODEL_RECON_BREAKPOINTS
from edge.health_index import HealthIndexResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURE_KEYS = ["feat_a", "feat_b", "feat_c"]


def _make_normal_sample(seed: int = 0) -> dict:
    """정상 범위 피처 생성."""
    rng = random.Random(seed)
    return {k: rng.gauss(10.0, 1.0) for k in FEATURE_KEYS}


def _make_normal_samples(n: int = 20) -> list[dict]:
    return [_make_normal_sample(i) for i in range(n)]


def _make_anomalous_sample() -> dict:
    """비정상 피처 (정상 범위에서 크게 벗어남)."""
    return {k: 100.0 for k in FEATURE_KEYS}


def _make_hi_result(composite: float = 1.0) -> HealthIndexResult:
    return HealthIndexResult(
        individual={"hi_rms": composite, "hi_kurtosis": composite},
        composite=composite,
        
    )


# ---------------------------------------------------------------------------
# BearingAutoencoder 모델 구조
# ---------------------------------------------------------------------------


class TestBearingAutoencoder:
    """모델 구조 테스트."""

    def test_forward_shape(self):
        model = BearingAutoencoder(input_dim=5, latent_dim=3)
        x = torch.randn(4, 5)
        out = model(x)
        assert out.shape == (4, 5)

    def test_encoder_decoder_dims(self):
        model = BearingAutoencoder(input_dim=10, latent_dim=4)
        # encoder: 10 → 4
        assert model.encoder[0].in_features == 10
        assert model.encoder[0].out_features == 4
        # decoder: 4 → 10
        assert model.decoder[0].in_features == 4
        assert model.decoder[0].out_features == 10

    def test_default_latent_dim(self):
        from edge.config import MODEL_LATENT_DIM

        model = BearingAutoencoder(input_dim=15)
        assert model.encoder[0].out_features == MODEL_LATENT_DIM


# ---------------------------------------------------------------------------
# _normalize_features
# ---------------------------------------------------------------------------


class TestNormalizeFeatures:
    """피처 정규화 테스트."""

    def test_basic_normalization(self):
        feat = {"a": 10.0, "b": 20.0}
        means = {"a": 5.0, "b": 15.0}
        stds = {"a": 2.5, "b": 5.0}
        result = _normalize_features(feat, ["a", "b"], means, stds)
        assert result == pytest.approx([2.0, 1.0])

    def test_zero_std_uses_min_std(self):
        feat = {"a": 5.0}
        means = {"a": 5.0}
        stds = {"a": 0.0}
        result = _normalize_features(feat, ["a"], means, stds)
        assert result == pytest.approx([0.0])


# ---------------------------------------------------------------------------
# train_autoencoder
# ---------------------------------------------------------------------------


class TestTrainAutoencoder:
    """학습 테스트."""

    def test_basic_training(self):
        samples = _make_normal_samples(10)
        baseline = train_autoencoder(
            samples,
            feature_keys=FEATURE_KEYS,
            epochs=50,
        )
        assert isinstance(baseline, AutoencoderBaseline)
        assert baseline.snapshot_count == 10
        assert set(baseline.feature_keys) == set(FEATURE_KEYS)
        assert baseline.normal_recon_mean >= 0
        assert baseline.normal_recon_std >= 0

    def test_n_snapshots_limit(self):
        samples = _make_normal_samples(20)
        baseline = train_autoencoder(
            samples,
            n_snapshots=5,
            feature_keys=FEATURE_KEYS,
            epochs=10,
        )
        assert baseline.snapshot_count == 5

    def test_too_few_snapshots_raises(self):
        with pytest.raises(ValueError, match="최소 2개"):
            train_autoencoder(
                [_make_normal_sample()],
                feature_keys=FEATURE_KEYS,
            )

    def test_means_stds_computed(self):
        samples = _make_normal_samples(10)
        baseline = train_autoencoder(
            samples,
            feature_keys=FEATURE_KEYS,
            epochs=10,
        )
        for key in FEATURE_KEYS:
            assert key in baseline.feature_means
            assert key in baseline.feature_stds

    def test_reconstruction_low_for_normal(self):
        """정상 데이터로 학습 후 정상 데이터의 recon error가 낮아야 함."""
        samples = _make_normal_samples(20)
        baseline = train_autoencoder(
            samples,
            feature_keys=FEATURE_KEYS,
            epochs=200,
        )
        # 학습 데이터에 대한 reconstruction error는 낮아야 한다
        assert baseline.normal_recon_mean < 1.0


# ---------------------------------------------------------------------------
# compute_reconstruction_error
# ---------------------------------------------------------------------------


class TestComputeReconstructionError:
    """추론 테스트."""

    @pytest.fixture()
    def trained_baseline(self):
        samples = _make_normal_samples(20)
        return train_autoencoder(
            samples,
            feature_keys=FEATURE_KEYS,
            epochs=200,
        )

    def test_normal_sample_low_error(self, trained_baseline):
        normal = _make_normal_sample(seed=99)
        error = compute_reconstruction_error(normal, trained_baseline)
        assert error >= 0
        # 정상 범위 내 데이터는 상대적으로 낮은 error
        threshold = (
            trained_baseline.normal_recon_mean
            + 5.0 * trained_baseline.normal_recon_std
        )
        assert error < threshold * 5  # 넉넉한 범위

    def test_anomalous_sample_high_error(self, trained_baseline):
        anomalous = _make_anomalous_sample()
        error_anom = compute_reconstruction_error(anomalous, trained_baseline)

        normal = _make_normal_sample(seed=99)
        error_norm = compute_reconstruction_error(normal, trained_baseline)

        # 비정상 데이터는 정상보다 확실히 높은 error
        assert error_anom > error_norm


# ---------------------------------------------------------------------------
# _recon_to_score
# ---------------------------------------------------------------------------


class TestReconToScore:
    """Reconstruction error → score 매핑."""

    def test_below_threshold_low_score(self):
        # ratio = 0.3 (threshold의 30%) → breakpoints[0]=(0.5, 0.0) 이하 → 0.0
        score = _recon_to_score(0.3, normal_mean=0.5, normal_std=0.1)
        # threshold = 0.5 + 3*0.1 = 0.8, ratio = 0.3/0.8 = 0.375
        assert score == 0.0

    def test_at_threshold_boundary(self):
        # threshold = mean + 3*std = 0.1 + 3*0.03 = 0.19
        # ratio = 0.19/0.19 = 1.0 → score = 0.65
        score = _recon_to_score(0.19, normal_mean=0.1, normal_std=0.03)
        assert score == pytest.approx(0.65, abs=0.01)

    def test_high_error_approaches_max(self):
        # threshold = 0.1 + 0.03*3 = 0.19
        # ratio = 0.57/0.19 = 3.0 → score = 1.0
        score = _recon_to_score(0.57, normal_mean=0.1, normal_std=0.03)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_std_handled(self):
        # std=0 → threshold = mean + 3*1e-10
        score = _recon_to_score(0.0, normal_mean=0.0, normal_std=0.0)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ModelCheckDetail
# ---------------------------------------------------------------------------


class TestModelCheckDetail:
    """ModelCheckDetail dataclass."""

    def test_creation(self):
        detail = ModelCheckDetail(reconstruction_error=0.5, score=0.8)
        assert detail.reconstruction_error == 0.5
        assert detail.score == 0.8

    def test_frozen(self):
        detail = ModelCheckDetail(reconstruction_error=0.5, score=0.8)
        with pytest.raises(AttributeError):
            detail.score = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# detect_anomaly 통합 (3모듈 Combiner)
# ---------------------------------------------------------------------------


class TestDetectAnomalyWithModel:
    """3모듈 통합 Combiner 테스트."""

    @pytest.fixture()
    def setup(self):
        """정상 데이터로 모든 baseline 구성."""
        from edge.anomaly_detection import compute_anomaly_baseline

        # 15개 실제 피처키 사용
        keys = MODEL_FEATURE_KEYS
        rng = random.Random(42)

        samples = []
        for i in range(20):
            sample = {k: rng.gauss(10.0, 1.0) for k in keys}
            samples.append(sample)

        anomaly_bl = compute_anomaly_baseline(
            samples, feature_keys=keys
        )
        ae_bl = train_autoencoder(
            samples,
            feature_keys=keys,
            epochs=100,
        )
        return samples, anomaly_bl, ae_bl

    def test_normal_three_modules(self, setup):
        samples, anomaly_bl, ae_bl = setup
        hi = _make_hi_result(0.5)  # 정상 HI
        result = detect_anomaly(
            hi,
            flat_features=samples[0],
            anomaly_baseline=anomaly_bl,
            autoencoder_baseline=ae_bl,
        )
        assert result.stat_detail is not None
        assert result.model_detail is not None
        assert result.health_state == "normal"
        assert result.anomaly_detected is False

    def test_model_detail_populated(self, setup):
        samples, anomaly_bl, ae_bl = setup
        hi = _make_hi_result(0.5)
        result = detect_anomaly(
            hi,
            flat_features=samples[0],
            anomaly_baseline=anomaly_bl,
            autoencoder_baseline=ae_bl,
        )
        assert result.model_detail is not None
        assert result.model_detail.reconstruction_error >= 0
        assert 0.0 <= result.model_detail.score <= 1.0

    def test_without_autoencoder_baseline(self, setup):
        """autoencoder_baseline=None이면 model_detail 없이 동작."""
        samples, anomaly_bl, _ = setup
        hi = _make_hi_result(0.5)
        result = detect_anomaly(
            hi,
            flat_features=samples[0],
            anomaly_baseline=anomaly_bl,
            autoencoder_baseline=None,
        )
        assert result.model_detail is None
        assert result.stat_detail is not None

    def test_rule_only_backward_compat(self):
        """flat_features 없이 Rule만 사용하는 기존 동작 유지."""
        hi = _make_hi_result(0.5)
        result = detect_anomaly(hi)
        assert result.stat_detail is None
        assert result.model_detail is None
        assert result.health_state == "normal"

    def test_anomalous_detected(self, setup):
        """비정상 데이터가 높은 score를 산출."""
        _, anomaly_bl, ae_bl = setup
        hi = _make_hi_result(4.0)  # 높은 HI
        anomalous = {k: 100.0 for k in MODEL_FEATURE_KEYS}
        result = detect_anomaly(
            hi,
            flat_features=anomalous,
            anomaly_baseline=anomaly_bl,
            autoencoder_baseline=ae_bl,
        )
        assert result.anomaly_detected is True
        assert result.anomaly_score > 0.65

    def test_confidence_with_three_modules(self, setup):
        samples, anomaly_bl, ae_bl = setup
        hi = _make_hi_result(0.5)
        result = detect_anomaly(
            hi,
            flat_features=samples[0],
            anomaly_baseline=anomaly_bl,
            autoencoder_baseline=ae_bl,
        )
        assert 0.5 <= result.confidence <= 1.0
