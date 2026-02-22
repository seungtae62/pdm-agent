"""시나리오별 이벤트 페이로드 생성 스크립트.

IMS 베어링 데이터셋에서 시나리오 4건의 이벤트 페이로드를 생성한다.

Usage:
    python scripts/generate_payloads.py
    python scripts/generate_payloads.py --data-dir /path/to/ims
    python scripts/generate_payloads.py --output-dir /path/to/output

시나리오:
    SC-001: Test1-Brg3, 5일차 (정상)
    SC-002: Test1-Brg3, 15일차 (초기 열화)
    SC-003: Test1-Brg3, 25일차 (결함 진행)
    SC-004: Test2-Brg1, 4일차 (급속 열화)
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.anomaly_detection import compute_anomaly_baseline, detect_anomaly
from edge.autoencoder import train_autoencoder
from edge.config import HI_BASELINE_SNAPSHOT_COUNT, MODEL_BASELINE_SNAPSHOT_COUNT
from edge.feature_pipeline import extract_snapshot_features, flatten_features
from edge.health_index import compute_baseline, compute_health_indices
from edge.loader import get_test_set_info, list_snapshots, load_snapshot
from edge.metadata import get_channel_indices
from edge.payload_generator import build_event_payload, save_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ims"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "payloads"


# ---------------------------------------------------------------------------
# 시나리오 정의
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """시나리오 설정."""

    scenario_id: str
    test_set_id: str
    bearing_id: str
    target_day: int
    description: str
    output_filename: str


SCENARIOS: list[ScenarioConfig] = [
    ScenarioConfig(
        scenario_id="SC-001",
        test_set_id="1st_test",
        bearing_id="BRG-003",
        target_day=5,
        description="정상 구간 (5일차)",
        output_filename="scenario1_day05.json",
    ),
    ScenarioConfig(
        scenario_id="SC-002",
        test_set_id="1st_test",
        bearing_id="BRG-003",
        target_day=15,
        description="초기 열화 (15일차)",
        output_filename="scenario1_day15.json",
    ),
    ScenarioConfig(
        scenario_id="SC-003",
        test_set_id="1st_test",
        bearing_id="BRG-003",
        target_day=25,
        description="결함 진행 (25일차)",
        output_filename="scenario1_day25.json",
    ),
    ScenarioConfig(
        scenario_id="SC-004",
        test_set_id="2nd_test",
        bearing_id="BRG-001",
        target_day=4,
        description="급속 열화 (4일차)",
        output_filename="scenario2_day04.json",
    ),
]


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------


def find_snapshot_at_day(
    snapshots: list[tuple[datetime, Path]],
    target_day: int,
) -> int:
    """target_day에 해당하는 스냅샷 인덱스를 반환.

    시작 시점으로부터 target_day일이 경과한 시점에서
    가장 가까운 스냅샷의 인덱스를 반환한다.
    """
    start_ts = snapshots[0][0]
    target_ts = start_ts + timedelta(days=target_day)

    best_idx = 0
    best_diff = abs((snapshots[0][0] - target_ts).total_seconds())

    for i, (ts, _) in enumerate(snapshots):
        diff = abs((ts - target_ts).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    return best_idx


def extract_features_batch(
    snapshots: list[tuple[datetime, Path]],
    channel_indices: list[int],
    channels_per_file: int,
    indices: list[int],
) -> list[tuple[datetime, dict[str, dict[str, dict]]]]:
    """지정된 인덱스의 스냅샷에서 특징량을 일괄 추출.

    Returns:
        (timestamp, features) 리스트.
    """
    results = []
    for idx in indices:
        ts, path = snapshots[idx]
        snapshot_data = load_snapshot(path, channels_per_file)
        features = extract_snapshot_features(
            snapshot_data, channel_indices
        )
        results.append((ts, features))
    return results


def generate_scenario_payload(
    scenario: ScenarioConfig,
    data_dir: Path,
) -> dict:
    """단일 시나리오의 이벤트 페이로드를 생성.

    파이프라인:
    1. 스냅샷 목록 로드
    2. 베이스라인 스냅샷 특징량 추출 (HI/Statistical/AE 학습용)
    3. 대상 스냅샷 특징량 추출
    4. 베이스라인 학습 (HI, Statistical, Autoencoder)
    5. 이상감지 실행
    6. 페이로드 조립
    """
    test_dir = data_dir / scenario.test_set_id
    if not test_dir.exists():
        raise FileNotFoundError(f"데이터 디렉토리 없음: {test_dir}")

    test_info = get_test_set_info(scenario.test_set_id)
    channels_per_file = test_info["channels_per_file"]
    channel_indices = get_channel_indices(scenario.test_set_id, scenario.bearing_id)

    # 1. 스냅샷 목록
    snapshots = list_snapshots(test_dir)
    logger.info(
        f"  [{scenario.scenario_id}] 스냅샷 {len(snapshots)}개 로드 "
        f"({snapshots[0][0].date()} ~ {snapshots[-1][0].date()})"
    )

    # 대상 스냅샷 인덱스
    target_idx = find_snapshot_at_day(snapshots, scenario.target_day)
    target_ts = snapshots[target_idx][0]
    logger.info(
        f"  [{scenario.scenario_id}] 대상: idx={target_idx}, "
        f"timestamp={target_ts}"
    )

    # 2. 베이스라인 스냅샷 특징량 추출
    # AE가 가장 많이 필요 (100개), 나머지는 20개
    n_baseline = min(MODEL_BASELINE_SNAPSHOT_COUNT, target_idx)
    if n_baseline < HI_BASELINE_SNAPSHOT_COUNT:
        logger.warning(
            f"  [{scenario.scenario_id}] 베이스라인 스냅샷 부족: "
            f"{n_baseline} < {HI_BASELINE_SNAPSHOT_COUNT}"
        )

    baseline_indices = list(range(n_baseline))
    logger.info(f"  [{scenario.scenario_id}] 베이스라인 {n_baseline}개 추출 중...")

    baseline_batch = extract_features_batch(
        snapshots, channel_indices, channels_per_file, baseline_indices
    )
    baseline_flat = [flatten_features(feat) for _, feat in baseline_batch]

    # 3. 대상 스냅샷 특징량
    target_data = load_snapshot(snapshots[target_idx][1], channels_per_file)
    target_features = extract_snapshot_features(target_data, channel_indices)
    target_flat = flatten_features(target_features)

    # 4. 베이스라인 학습
    logger.info(f"  [{scenario.scenario_id}] 베이스라인 학습 중...")

    # HI baseline
    hi_baseline = compute_baseline(
        baseline_flat,
        n_snapshots=min(HI_BASELINE_SNAPSHOT_COUNT, n_baseline),
    )

    # Statistical baseline
    stat_baseline = compute_anomaly_baseline(
        baseline_flat,
        n_snapshots=min(HI_BASELINE_SNAPSHOT_COUNT, n_baseline),
    )

    # Autoencoder baseline
    ae_baseline = None
    if n_baseline >= 20:  # AE는 최소 20개 이상에서만 학습
        ae_baseline = train_autoencoder(baseline_flat, n_snapshots=n_baseline)
        logger.info(
            f"  [{scenario.scenario_id}] AE 학습 완료 "
            f"(epoch={ae_baseline.final_epoch}, "
            f"recon_mean={ae_baseline.normal_recon_mean:.6f})"
        )

    # 5. 이상감지
    hi_result = compute_health_indices(target_flat, hi_baseline)
    anomaly_result = detect_anomaly(
        hi_result,
        flat_features=target_flat,
        anomaly_baseline=stat_baseline,
        autoencoder_baseline=ae_baseline,
    )
    logger.info(
        f"  [{scenario.scenario_id}] 이상감지 결과: "
        f"score={anomaly_result.anomaly_score:.4f}, "
        f"detected={anomaly_result.anomaly_detected}, "
        f"state={anomaly_result.health_state}"
    )

    # 6. 페이로드 조립
    operation_start = snapshots[0][0]
    payload = build_event_payload(
        test_set_id=scenario.test_set_id,
        bearing_id=scenario.bearing_id,
        snapshot_timestamp=target_ts,
        operation_start_date=operation_start,
        features=target_features,
        anomaly_result=anomaly_result,
        event_seq=1,
    )

    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="시나리오별 이벤트 페이로드 생성"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"IMS 데이터 디렉토리 (기본: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"출력 디렉토리 (기본: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="특정 시나리오만 실행 (SC-001, SC-002, SC-003, SC-004)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(
            f"IMS 데이터 디렉토리 없음: {args.data_dir}\n"
            "  먼저 실행: python scripts/download_ims_dataset.py"
        )
        sys.exit(1)

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s.scenario_id == args.scenario]
        if not scenarios:
            logger.error(f"알 수 없는 시나리오: {args.scenario}")
            sys.exit(1)

    logger.info(f"=== 이벤트 페이로드 생성 시작 ({len(scenarios)}건) ===")

    for scenario in scenarios:
        logger.info(
            f"\n[{scenario.scenario_id}] {scenario.description} "
            f"({scenario.test_set_id}/{scenario.bearing_id}, "
            f"day {scenario.target_day})"
        )
        try:
            payload = generate_scenario_payload(scenario, args.data_dir)
            output_path = save_payload(
                payload, args.output_dir / scenario.output_filename
            )
            logger.info(f"  [{scenario.scenario_id}] 저장 완료: {output_path}")
        except Exception:
            logger.exception(f"  [{scenario.scenario_id}] 생성 실패")

    logger.info("\n=== 완료 ===")


if __name__ == "__main__":
    main()
