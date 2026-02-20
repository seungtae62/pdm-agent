"""Health Index 산출 모듈.

초기 N개 스냅샷의 특징량으로 베이스라인을 구성하고,
이후 스냅샷의 특징량을 정규화하여 5종 개별 HI 및 Composite HI를 산출한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from edge.config import HI_BASELINE_SNAPSHOT_COUNT, HI_CLAMP_MAX, HI_FEATURE_KEYS, HI_WEIGHTS


@dataclass(frozen=True)
class BaselineStats:
    """HI 피처별 베이스라인 min/max 통계."""

    min_values: dict[str, float]
    max_values: dict[str, float]
    snapshot_count: int


@dataclass(frozen=True)
class HealthIndexResult:
    """5종 개별 HI + Composite HI."""

    individual: dict[str, float]
    composite: float


def compute_baseline(
    feature_dicts: list[dict],
    n_snapshots: int = HI_BASELINE_SNAPSHOT_COUNT,
) -> BaselineStats:
    """초기 N개 스냅샷의 flat features에서 HI 피처별 min/max 산출.

    Args:
        feature_dicts: flatten_features() 결과 리스트.
        n_snapshots: 베이스라인에 사용할 스냅샷 수.

    Returns:
        BaselineStats with min/max per HI feature key.

    Raises:
        ValueError: feature_dicts 길이가 n_snapshots 미만일 때.
    """
    if len(feature_dicts) < n_snapshots:
        raise ValueError(
            f"베이스라인 산출에 {n_snapshots}개 스냅샷이 필요하지만 "
            f"{len(feature_dicts)}개만 제공됨"
        )

    baseline_dicts = feature_dicts[:n_snapshots]
    min_values: dict[str, float] = {}
    max_values: dict[str, float] = {}

    for hi_key, feat_key in HI_FEATURE_KEYS.items():
        values = [d[feat_key] for d in baseline_dicts]
        min_values[hi_key] = min(values)
        max_values[hi_key] = max(values)

    return BaselineStats(
        min_values=min_values,
        max_values=max_values,
        snapshot_count=n_snapshots,
    )


def normalize_hi(
    value: float,
    baseline_min: float,
    baseline_max: float,
    clamp_max: float | None = None,
) -> float:
    """Min-Max 정규화.

    0=baseline_min, 1=baseline_max, >1=열화 진행.
    하한 0 클램핑, 상한 clamp_max 클램핑.

    min==max 퇴화 케이스:
        baseline 값과 같으면 0.0, 초과하면 상대 편차.

    Args:
        value: 정규화할 값.
        baseline_min: 베이스라인 최소값.
        baseline_max: 베이스라인 최대값.
        clamp_max: 상한 클램핑 값. None이면 config 기본값 사용.
    """
    if clamp_max is None:
        clamp_max = HI_CLAMP_MAX

    if baseline_min == baseline_max:
        if baseline_max == 0:
            raw = 0.0 if value <= 0 else float(value)
        else:
            raw = max(0.0, (value - baseline_max) / baseline_max)
    else:
        raw = max(0.0, (value - baseline_min) / (baseline_max - baseline_min))

    return min(raw, clamp_max)


def compute_health_indices(
    flat_features: dict,
    baseline: BaselineStats,
    weights: dict[str, float] | None = None,
) -> HealthIndexResult:
    """5종 개별 HI + Composite HI 산출.

    Args:
        flat_features: flatten_features() 결과 (단일 스냅샷).
        baseline: compute_baseline()으로 산출한 베이스라인.
        weights: HI별 가중치. None이면 config 기본값 사용.

    Returns:
        HealthIndexResult with individual HIs and composite.
    """
    if weights is None:
        weights = HI_WEIGHTS

    individual: dict[str, float] = {}
    for hi_key, feat_key in HI_FEATURE_KEYS.items():
        value = flat_features[feat_key]
        individual[hi_key] = normalize_hi(
            value, baseline.min_values[hi_key], baseline.max_values[hi_key]
        )

    composite = sum(weights[k] * individual[k] for k in individual)

    return HealthIndexResult(individual=individual, composite=composite)
