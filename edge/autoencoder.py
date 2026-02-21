"""Autoencoder 기반 이상 감지 모듈 — Part 3c.

정상 패턴을 학습한 MLP Autoencoder의 reconstruction error를
기반으로 이상을 감지한다.

구조: Input(N) → Encoder(N→latent, ReLU) → Decoder(latent→N, Tanh)
- Tanh: 출력 범위를 [-1, 1]로 제한하여 수치 안정성 확보
- 입력은 z-score 정규화 후 ±CLAMP 범위로 클램핑

학습: 베이스라인 스냅샷의 MSE loss 최소화 + weight decay + early stopping
추론: reconstruction MSE → piecewise linear score 매핑
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from edge.config import (
    MODEL_BASELINE_SNAPSHOT_COUNT,
    MODEL_EARLY_STOPPING_MIN_DELTA,
    MODEL_EARLY_STOPPING_PATIENCE,
    MODEL_FEATURE_KEYS,
    MODEL_INPUT_CLAMP,
    MODEL_LATENT_DIM,
    MODEL_LEARNING_RATE,
    MODEL_TRAIN_EPOCHS,
    MODEL_WEIGHT_DECAY,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BearingAutoencoder(nn.Module):
    """MLP Autoencoder with bounded output.

    Args:
        input_dim: 입력 피처 수.
        latent_dim: 잠재 공간 차원.
    """

    def __init__(self, input_dim: int, latent_dim: int = MODEL_LATENT_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Tanh(),  # 출력 범위 [-1, 1] 제한
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode → decode."""
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Baseline (학습 결과)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoencoderBaseline:
    """학습된 Autoencoder + 정상 데이터 통계.

    Attributes:
        model: 학습된 BearingAutoencoder.
        feature_keys: 사용된 피처 키 목록.
        feature_means: 피처별 평균 (정규화용).
        feature_stds: 피처별 표준편차 (정규화용).
        input_clamp: 정규화 후 클램핑 범위.
        normal_recon_mean: 정상 데이터 reconstruction MSE 평균.
        normal_recon_std: 정상 데이터 reconstruction MSE 표준편차.
        snapshot_count: 학습에 사용된 스냅샷 수.
        final_epoch: 실제 학습 에폭 수 (early stopping 적용).
    """

    model: BearingAutoencoder
    feature_keys: list[str]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    input_clamp: float
    normal_recon_mean: float
    normal_recon_std: float
    snapshot_count: int
    final_epoch: int


# ---------------------------------------------------------------------------
# 정규화 + 클램핑
# ---------------------------------------------------------------------------


def _normalize_features(
    flat_features: dict,
    feature_keys: list[str],
    means: dict[str, float],
    stds: dict[str, float],
    clamp: float = MODEL_INPUT_CLAMP,
    min_std: float = 1e-10,
) -> list[float]:
    """피처를 z-score 정규화 후 클램핑하여 리스트로 반환.

    Args:
        flat_features: 피처 dict.
        feature_keys: 정규화할 키 목록.
        means: 피처별 평균.
        stds: 피처별 표준편차.
        clamp: 정규화 후 ±clamp 범위로 클램핑.
        min_std: 최소 std (0 나누기 방지).

    Returns:
        정규화+클램핑된 값 리스트.
    """
    result: list[float] = []
    for key in feature_keys:
        val = float(flat_features[key])
        std = max(stds[key], min_std)
        z = (val - means[key]) / std
        z = max(-clamp, min(clamp, z))  # 클램핑
        result.append(z)
    return result


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------


def train_autoencoder(
    flat_features_list: list[dict],
    n_snapshots: int | None = None,
    *,
    feature_keys: list[str] | None = None,
    latent_dim: int = MODEL_LATENT_DIM,
    epochs: int = MODEL_TRAIN_EPOCHS,
    learning_rate: float = MODEL_LEARNING_RATE,
    weight_decay: float = MODEL_WEIGHT_DECAY,
    input_clamp: float = MODEL_INPUT_CLAMP,
    patience: int = MODEL_EARLY_STOPPING_PATIENCE,
    min_delta: float = MODEL_EARLY_STOPPING_MIN_DELTA,
) -> AutoencoderBaseline:
    """베이스라인 스냅샷으로 Autoencoder를 학습.

    Args:
        flat_features_list: flatten_features() 결과 리스트.
        n_snapshots: 사용할 스냅샷 수. None이면 config 기본값.
        feature_keys: 대상 피처 키 목록. None이면 config 기본값.
        latent_dim: 잠재 공간 차원.
        epochs: 최대 학습 에폭 수.
        learning_rate: 학습률.
        weight_decay: L2 regularization 강도.
        input_clamp: 정규화 후 클램핑 범위.
        patience: Early stopping patience (에폭).
        min_delta: Early stopping 최소 개선량.

    Returns:
        AutoencoderBaseline with trained model and normalization stats.

    Raises:
        ValueError: 스냅샷이 2개 미만일 때.
    """
    if feature_keys is None:
        feature_keys = MODEL_FEATURE_KEYS

    if n_snapshots is None:
        n_snapshots = MODEL_BASELINE_SNAPSHOT_COUNT

    flat_features_list = flat_features_list[:n_snapshots]

    n = len(flat_features_list)
    if n < 2:
        raise ValueError(f"최소 2개 스냅샷 필요, {n}개 제공됨")

    # 피처별 mean/std 산출
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in feature_keys:
        values = [float(f[key]) for f in flat_features_list]
        m = sum(values) / n
        var = sum((v - m) ** 2 for v in values) / (n - 1)
        means[key] = m
        stds[key] = math.sqrt(var)

    # 정규화 + 클램핑된 텐서 생성
    rows: list[list[float]] = []
    for feat_dict in flat_features_list:
        rows.append(
            _normalize_features(feat_dict, feature_keys, means, stds, clamp=input_clamp)
        )

    data_tensor = torch.tensor(rows, dtype=torch.float32)

    # 모델 생성 및 학습
    input_dim = len(feature_keys)
    model = BearingAutoencoder(input_dim, latent_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()

    # Early stopping
    best_loss = float("inf")
    patience_counter = 0
    final_epoch = 0

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data_tensor)
        loss = loss_fn(output, data_tensor)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        final_epoch = epoch + 1

        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # 정상 데이터의 reconstruction error 통계
    model.eval()
    with torch.no_grad():
        recon = model(data_tensor)
        per_sample_mse = ((recon - data_tensor) ** 2).mean(dim=1)
        recon_mean = per_sample_mse.mean().item()
        recon_std = per_sample_mse.std().item() if n > 1 else 0.0

    return AutoencoderBaseline(
        model=model,
        feature_keys=feature_keys,
        feature_means=means,
        feature_stds=stds,
        input_clamp=input_clamp,
        normal_recon_mean=recon_mean,
        normal_recon_std=recon_std,
        snapshot_count=n,
        final_epoch=final_epoch,
    )


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------


def compute_reconstruction_error(
    flat_features: dict,
    baseline: AutoencoderBaseline,
) -> float:
    """단일 스냅샷의 reconstruction MSE를 산출.

    입력은 학습 시와 동일한 정규화+클램핑을 적용한다.
    클램핑 덕분에 극단적 입력에도 출력이 bounded.

    Args:
        flat_features: flatten_features() 결과.
        baseline: train_autoencoder() 결과.

    Returns:
        Reconstruction MSE (정규화+클램핑된 피처 공간).
    """
    normed = _normalize_features(
        flat_features,
        baseline.feature_keys,
        baseline.feature_means,
        baseline.feature_stds,
        clamp=baseline.input_clamp,
    )
    input_tensor = torch.tensor([normed], dtype=torch.float32)

    baseline.model.eval()
    with torch.no_grad():
        output = baseline.model(input_tensor)
        mse = ((output - input_tensor) ** 2).mean().item()

    return mse
