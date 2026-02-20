"""Anomaly Detection 모듈 — Part 3a + 3b.

3개의 독립적인 감지 모듈이 flat_features를 입력으로 병렬 실행되며,
Combiner가 결과를 결합하여 최종 anomaly_score를 산출한다.

[A] HI + Rule Check (Part 3a): HI 값에 규칙 기반 임계값 검사
[B] Statistical (Part 3b): flat_features의 z-score 기반 통계적 이탈도
[C] Autoencoder (Part 3c): 미구현

Part 3b에서는 anomaly_score = w_rule × rule + w_stat × stat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from edge.config import (
    ANOMALY_HI_BREAKPOINTS,
    ANOMALY_HEALTH_STATE_TIERS,
    ANOMALY_MODEL_ID,
    ANOMALY_SPIKE_THRESHOLD,
    ANOMALY_THRESHOLD,
    COMBINER_WEIGHTS,
    STAT_FEATURE_KEYS,
    STAT_MIN_STD,
    STAT_Z_SCORE_BREAKPOINTS,
)
from edge.health_index import HealthIndexResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleCheckDetail:
    """Rule Check 세부 결과."""

    composite_hi_score: float
    spike_score: float
    spiked_keys: list[str]


@dataclass(frozen=True)
class AnomalyBaseline:
    """Statistical 모듈용 베이스라인 (피처별 mean/std).

    compute_anomaly_baseline()으로 생성한다.
    """

    mean: dict[str, float]
    std: dict[str, float]
    snapshot_count: int


@dataclass(frozen=True)
class StatCheckDetail:
    """Statistical Check 세부 결과."""

    z_scores: dict[str, float]
    max_z_score: float
    max_z_feature: str
    stat_score: float


@dataclass(frozen=True)
class AnomalyResult:
    """이상 감지 최종 결과."""

    model_id: str
    anomaly_detected: bool
    anomaly_score: float
    anomaly_threshold: float
    health_state: str  # normal | watch | warning | critical
    confidence: float
    rule_detail: RuleCheckDetail
    stat_detail: StatCheckDetail | None = None
    # Part 3c에서 model_detail 추가 예정


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------


def _piecewise_linear(
    value: float,
    breakpoints: list[tuple[float, float]],
) -> float:
    """구간별 선형 보간으로 value → 0~1 score 변환.

    Args:
        value: 변환할 값.
        breakpoints: (input, score) 쌍 리스트. 오름차순 정렬.

    Returns:
        0.0~1.0 범위의 score.
    """
    if value <= breakpoints[0][0]:
        return breakpoints[0][1]

    if value >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    for i in range(len(breakpoints) - 1):
        lo_val, lo_score = breakpoints[i]
        hi_val, hi_score = breakpoints[i + 1]
        if lo_val <= value <= hi_val:
            ratio = (value - lo_val) / (hi_val - lo_val)
            return lo_score + ratio * (hi_score - lo_score)

    return breakpoints[-1][1]


def _hi_to_score(
    hi_value: float,
    breakpoints: list[tuple[float, float]] | None = None,
) -> float:
    """HI 값을 0~1 anomaly score로 변환."""
    if breakpoints is None:
        breakpoints = ANOMALY_HI_BREAKPOINTS
    return _piecewise_linear(hi_value, breakpoints)


def _z_to_score(
    z_value: float,
    breakpoints: list[tuple[float, float]] | None = None,
) -> float:
    """z-score 절대값을 0~1 anomaly score로 변환."""
    if breakpoints is None:
        breakpoints = STAT_Z_SCORE_BREAKPOINTS
    return _piecewise_linear(abs(z_value), breakpoints)


# ---------------------------------------------------------------------------
# [A] Rule Check (Part 3a)
# ---------------------------------------------------------------------------


def check_rules(
    hi_result: HealthIndexResult,
    *,
    spike_threshold: float | None = None,
) -> RuleCheckDetail:
    """Composite HI score + Individual spike score 산출."""
    if spike_threshold is None:
        spike_threshold = ANOMALY_SPIKE_THRESHOLD

    composite_hi_score = _hi_to_score(hi_result.composite)

    spiked_keys: list[str] = []
    max_spiked_hi = 0.0

    for key, value in hi_result.individual.items():
        if value >= spike_threshold:
            spiked_keys.append(key)
            if value > max_spiked_hi:
                max_spiked_hi = value

    spike_score = _hi_to_score(max_spiked_hi) if spiked_keys else 0.0

    return RuleCheckDetail(
        composite_hi_score=composite_hi_score,
        spike_score=spike_score,
        spiked_keys=spiked_keys,
    )


# ---------------------------------------------------------------------------
# [B] Statistical (Part 3b)
# ---------------------------------------------------------------------------


def compute_anomaly_baseline(
    flat_features_list: list[dict],
    n_snapshots: int | None = None,
    *,
    feature_keys: list[str] | None = None,
) -> AnomalyBaseline:
    """베이스라인 스냅샷들로부터 피처별 mean/std를 산출.

    Args:
        flat_features_list: flatten_features() 결과 리스트.
        n_snapshots: 사용할 스냅샷 수. None이면 전체 사용.
        feature_keys: 대상 피처 키 목록. None이면 config 기본값.

    Returns:
        AnomalyBaseline with mean, std per feature.

    Raises:
        ValueError: 스냅샷이 2개 미만일 때.
    """
    if feature_keys is None:
        feature_keys = STAT_FEATURE_KEYS

    if n_snapshots is not None:
        flat_features_list = flat_features_list[:n_snapshots]

    n = len(flat_features_list)
    if n < 2:
        raise ValueError(f"최소 2개 스냅샷 필요, {n}개 제공됨")

    mean: dict[str, float] = {}
    std: dict[str, float] = {}

    for key in feature_keys:
        values = [f[key] for f in flat_features_list]
        m = sum(values) / n
        variance = sum((v - m) ** 2 for v in values) / (n - 1)  # sample std
        mean[key] = m
        std[key] = math.sqrt(variance)

    return AnomalyBaseline(mean=mean, std=std, snapshot_count=n)


def check_statistical(
    flat_features: dict,
    anomaly_baseline: AnomalyBaseline,
    *,
    feature_keys: list[str] | None = None,
    min_std: float | None = None,
) -> StatCheckDetail:
    """각 피처의 z-score를 산출하고 최대 z-score를 score로 변환.

    Args:
        flat_features: 현재 스냅샷의 flatten_features() 결과.
        anomaly_baseline: compute_anomaly_baseline() 결과.
        feature_keys: 대상 피처 키 목록. None이면 config 기본값.
        min_std: 최소 std (0 나누기 방지). None이면 config 기본값.

    Returns:
        StatCheckDetail with z_scores, max_z_score, max_z_feature, stat_score.
    """
    if feature_keys is None:
        feature_keys = STAT_FEATURE_KEYS
    if min_std is None:
        min_std = STAT_MIN_STD

    z_scores: dict[str, float] = {}
    max_z_score = 0.0
    max_z_feature = feature_keys[0]

    for key in feature_keys:
        value = flat_features[key]
        mean = anomaly_baseline.mean[key]
        s = max(anomaly_baseline.std[key], min_std)
        z = abs(value - mean) / s
        z_scores[key] = z

        if z > max_z_score:
            max_z_score = z
            max_z_feature = key

    stat_score = _z_to_score(max_z_score)

    return StatCheckDetail(
        z_scores=z_scores,
        max_z_score=max_z_score,
        max_z_feature=max_z_feature,
        stat_score=stat_score,
    )


# ---------------------------------------------------------------------------
# Health State 분류
# ---------------------------------------------------------------------------


def _classify_health_state(
    anomaly_score: float,
    tiers: list[tuple[float, str]] | None = None,
) -> str:
    """anomaly_score → health_state 분류."""
    if tiers is None:
        tiers = ANOMALY_HEALTH_STATE_TIERS

    for threshold, state in tiers:
        if anomaly_score < threshold:
            return state

    return "critical"


# ---------------------------------------------------------------------------
# Combiner + 최종 감지
# ---------------------------------------------------------------------------


def detect_anomaly(
    hi_result: HealthIndexResult,
    flat_features: dict | None = None,
    anomaly_baseline: AnomalyBaseline | None = None,
    *,
    threshold: float | None = None,
    weights: dict[str, float] | None = None,
) -> AnomalyResult:
    """이상 감지 메인 진입점.

    Part 3b: Rule + Statistical 두 모듈을 가중합산하여
    최종 anomaly_score를 산출한다.

    flat_features와 anomaly_baseline이 모두 제공되면 Statistical 모듈 실행.
    하나라도 None이면 Rule만 사용 (Part 3a 호환).

    Args:
        hi_result: compute_health_indices() 결과.
        flat_features: 현재 스냅샷의 flatten_features() 결과.
        anomaly_baseline: compute_anomaly_baseline() 결과.
        threshold: 이상 판정 임계값. None이면 config 기본값.
        weights: Combiner 가중치. None이면 config 기본값.

    Returns:
        AnomalyResult with rule_detail and optional stat_detail.
    """
    if threshold is None:
        threshold = ANOMALY_THRESHOLD
    if weights is None:
        weights = COMBINER_WEIGHTS

    # [A] Rule Check
    rule_detail = check_rules(hi_result)
    rule_score = max(rule_detail.composite_hi_score, rule_detail.spike_score)

    # [B] Statistical (optional)
    stat_detail: StatCheckDetail | None = None
    use_stat = flat_features is not None and anomaly_baseline is not None

    if use_stat:
        stat_detail = check_statistical(flat_features, anomaly_baseline)
        stat_score = stat_detail.stat_score

        # Combiner: 가중합산
        w_rule = weights.get("rule_based", 0.5)
        w_stat = weights.get("statistical", 0.5)
        total_w = w_rule + w_stat
        anomaly_score = (w_rule * rule_score + w_stat * stat_score) / total_w

        # Confidence: Rule↔Stat 일치도
        score_diff = abs(rule_score - stat_score)
        confidence = 0.5 + 0.5 * (1.0 - min(score_diff, 1.0))
    else:
        # Part 3a 호환: Rule만 사용
        anomaly_score = rule_score
        score_diff = abs(rule_detail.composite_hi_score - rule_detail.spike_score)
        confidence = 0.5 + 0.5 * (1.0 - min(score_diff, 1.0))

    # anomaly_score 범위 보장
    anomaly_score = max(0.0, min(1.0, anomaly_score))

    return AnomalyResult(
        model_id=ANOMALY_MODEL_ID,
        anomaly_detected=anomaly_score >= threshold,
        anomaly_score=anomaly_score,
        anomaly_threshold=threshold,
        health_state=_classify_health_state(anomaly_score),
        confidence=confidence,
        rule_detail=rule_detail,
        stat_detail=stat_detail,
    )
