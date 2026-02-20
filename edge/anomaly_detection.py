"""Anomaly Detection 모듈 — Part 3a: HI + Rule Check.

Health Index 결과에 규칙 기반 임계값 검사를 적용하여
anomaly_score (0~1)와 health_state를 산출한다.

3개의 독립적인 감지 모듈(Rule, Statistical, Autoencoder)이
동일한 flat_features를 입력으로 병렬 실행되며,
Combiner가 결과를 결합하여 최종 anomaly_score를 산출한다.

Part 3a에서는 [A] Rule Check만 구현하며,
anomaly_score = rule_score로 동작한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from edge.config import (
    ANOMALY_HI_BREAKPOINTS,
    ANOMALY_HEALTH_STATE_TIERS,
    ANOMALY_MODEL_ID,
    ANOMALY_SPIKE_THRESHOLD,
    ANOMALY_THRESHOLD,
)
from edge.health_index import HealthIndexResult


@dataclass(frozen=True)
class RuleCheckDetail:
    """Rule Check 세부 결과."""

    composite_hi_score: float
    spike_score: float
    spiked_keys: list[str]


@dataclass(frozen=True)
class AnomalyResult:
    """이상 감지 최종 결과.

    Part 3b에서 stat_detail, Part 3c에서 model_detail 추가 예정.
    """

    model_id: str
    anomaly_detected: bool
    anomaly_score: float
    anomaly_threshold: float
    health_state: str  # normal | watch | warning | critical
    confidence: float
    rule_detail: RuleCheckDetail


def _hi_to_score(
    hi_value: float,
    breakpoints: list[tuple[float, float]] | None = None,
) -> float:
    """HI 값을 0~1 anomaly score로 변환 (구간별 선형 보간).

    Args:
        hi_value: Health Index 값 (0 이상, 상한 없음).
        breakpoints: (hi, score) 쌍 리스트. 오름차순 정렬.
            None이면 config 기본값 사용.

    Returns:
        0.0~1.0 범위의 anomaly score. 최소 구간 이하 0, 최대 구간 초과 1.
    """
    if breakpoints is None:
        breakpoints = ANOMALY_HI_BREAKPOINTS

    # 최소 구간 이하
    if hi_value <= breakpoints[0][0]:
        return breakpoints[0][1]

    # 최대 구간 초과
    if hi_value >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    # 구간별 선형 보간
    for i in range(len(breakpoints) - 1):
        hi_lo, score_lo = breakpoints[i]
        hi_hi, score_hi = breakpoints[i + 1]
        if hi_lo <= hi_value <= hi_hi:
            ratio = (hi_value - hi_lo) / (hi_hi - hi_lo)
            return score_lo + ratio * (score_hi - score_lo)

    # 도달 불가능하지만 안전장치
    return breakpoints[-1][1]


def check_rules(
    hi_result: HealthIndexResult,
    *,
    spike_threshold: float | None = None,
) -> RuleCheckDetail:
    """Composite HI score + Individual spike score 산출.

    Args:
        hi_result: compute_health_indices() 결과.
        spike_threshold: 개별 HI 스파이크 감지 임계값.
            None이면 config 기본값 사용.

    Returns:
        RuleCheckDetail with composite_hi_score, spike_score, spiked_keys.
    """
    if spike_threshold is None:
        spike_threshold = ANOMALY_SPIKE_THRESHOLD

    # Composite HI → score
    composite_hi_score = _hi_to_score(hi_result.composite)

    # 개별 HI 중 spike_threshold 이상인 것 추출
    spiked_keys: list[str] = []
    max_spiked_hi = 0.0

    for key, value in hi_result.individual.items():
        if value >= spike_threshold:
            spiked_keys.append(key)
            if value > max_spiked_hi:
                max_spiked_hi = value

    # 스파이크가 있으면 최대값을 score로 변환, 없으면 0
    spike_score = _hi_to_score(max_spiked_hi) if spiked_keys else 0.0

    return RuleCheckDetail(
        composite_hi_score=composite_hi_score,
        spike_score=spike_score,
        spiked_keys=spiked_keys,
    )


def _classify_health_state(
    anomaly_score: float,
    tiers: list[tuple[float, str]] | None = None,
) -> str:
    """anomaly_score → health_state 분류.

    Args:
        anomaly_score: 0~1 범위의 이상 점수.
        tiers: (threshold, state) 쌍 리스트.
            score < threshold이면 해당 state.
            None이면 config 기본값 사용.

    Returns:
        "normal" | "watch" | "warning" | "critical"
    """
    if tiers is None:
        tiers = ANOMALY_HEALTH_STATE_TIERS

    for threshold, state in tiers:
        if anomaly_score < threshold:
            return state

    return "critical"


def detect_anomaly(
    hi_result: HealthIndexResult,
    *,
    threshold: float | None = None,
) -> AnomalyResult:
    """Part 3a: Rule-based anomaly detection.

    HealthIndexResult를 받아 Rule Check를 수행하고
    최종 AnomalyResult를 반환한다.

    Part 3a에서는 anomaly_score = rule_score (Rule만 사용).
    Part 3b에서 Statistical, Part 3c에서 Autoencoder 추가 시
    Combiner 가중합산으로 확장된다.

    Args:
        hi_result: compute_health_indices() 결과.
        threshold: 이상 판정 임계값. None이면 config 기본값 사용.

    Returns:
        AnomalyResult with all detection details.
    """
    if threshold is None:
        threshold = ANOMALY_THRESHOLD

    rule_detail = check_rules(hi_result)

    # Part 3a: anomaly_score = max(composite_score, spike_score)
    rule_score = max(rule_detail.composite_hi_score, rule_detail.spike_score)
    anomaly_score = rule_score

    # Confidence: 두 점수 일치도 기반
    # 0.5 + 0.5 × (1 - |composite - spike|)
    score_diff = abs(rule_detail.composite_hi_score - rule_detail.spike_score)
    confidence = 0.5 + 0.5 * (1.0 - min(score_diff, 1.0))

    return AnomalyResult(
        model_id=ANOMALY_MODEL_ID,
        anomaly_detected=anomaly_score >= threshold,
        anomaly_score=anomaly_score,
        anomaly_threshold=threshold,
        health_state=_classify_health_state(anomaly_score),
        confidence=confidence,
        rule_detail=rule_detail,
    )
