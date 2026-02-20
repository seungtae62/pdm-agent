"""Part 3a 검증 스크립트: 실제 IMS 데이터셋 기반 전체 파이프라인 검증.

flat_features → HI → anomaly detection 파이프라인을 3개 테스트셋에 대해
실행하고, breakpoints/threshold/health_state가 데이터 특성에 적합한지 분석한다.

Usage:
    python scripts/validate_part3a.py
    python scripts/validate_part3a.py --data-dir /path/to/ims
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.anomaly_detection import detect_anomaly, _hi_to_score
from edge.config import HI_BASELINE_SNAPSHOT_COUNT, HI_FEATURE_KEYS, ANOMALY_HI_BREAKPOINTS
from edge.feature_pipeline import extract_snapshot_features, flatten_features
from edge.health_index import compute_baseline, compute_health_indices
from edge.loader import (
    get_bearing_info,
    get_test_set_info,
    list_snapshots,
    load_snapshot,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ims"


def analyze_test_set(
    test_set_id: str,
    bearing_id: str,
    data_dir: Path,
    sample_interval: int = 1,
) -> dict:
    """단일 테스트셋/베어링에 대해 전체 파이프라인 실행 및 분석.

    Args:
        test_set_id: '1st_test', '2nd_test', '3rd_test'
        bearing_id: 'BRG-001', 'BRG-003', 'BRG-004'
        data_dir: IMS 데이터셋 경로
        sample_interval: 스냅샷 샘플링 간격 (1=전부, 10=10개마다 1개)

    Returns:
        분석 결과 dict
    """
    test_info = get_test_set_info(test_set_id)
    bearing_info = get_bearing_info(test_set_id, bearing_id)
    test_dir = data_dir / test_set_id

    print(f"\n{'='*70}")
    print(f"  {test_set_id} / {bearing_id} ({bearing_info['label']})")
    print(f"  고장 유형: {bearing_info['fault_type']}")
    print(f"  채널: {bearing_info['channels']}")
    print(f"{'='*70}")

    # 스냅샷 로드
    snapshots = list_snapshots(test_dir)
    total = len(snapshots)
    print(f"  총 스냅샷: {total}개")

    if total < HI_BASELINE_SNAPSHOT_COUNT + 10:
        print(f"  [ERROR] 스냅샷 부족 (최소 {HI_BASELINE_SNAPSHOT_COUNT + 10}개 필요)")
        return {}

    # 특징량 추출
    print(f"  특징량 추출 중 (interval={sample_interval})...", end="", flush=True)
    flat_features_list = []
    timestamps = []

    for i, (ts, path) in enumerate(snapshots):
        if i % sample_interval != 0:
            continue
        try:
            snapshot = load_snapshot(path, test_info["channels_per_file"])
            features = extract_snapshot_features(
                snapshot, bearing_info["channels"]
            )
            flat = flatten_features(features)
            flat_features_list.append(flat)
            timestamps.append(ts)
        except Exception as e:
            print(f"\n  [WARN] 스냅샷 {path.name} 로드 실패: {e}")
            continue

    print(f" {len(flat_features_list)}개 완료")

    # 베이스라인 산출
    baseline = compute_baseline(flat_features_list, HI_BASELINE_SNAPSHOT_COUNT)
    print(f"  베이스라인: {baseline.snapshot_count}개 스냅샷 사용")

    # HI + Anomaly Detection
    results = []
    for i, flat in enumerate(flat_features_list):
        hi = compute_health_indices(flat, baseline)
        anomaly = detect_anomaly(hi)
        results.append({
            "idx": i,
            "timestamp": timestamps[i],
            "composite_hi": hi.composite,
            "individual_hi": dict(hi.individual),
            "anomaly_score": anomaly.anomaly_score,
            "anomaly_detected": anomaly.anomaly_detected,
            "health_state": anomaly.health_state,
            "confidence": anomaly.confidence,
            "rule_detail": anomaly.rule_detail,
        })

    # 분석
    composites = [r["composite_hi"] for r in results]
    scores = [r["anomaly_score"] for r in results]
    states = [r["health_state"] for r in results]
    detected = [r["anomaly_detected"] for r in results]

    # 기본 통계
    print(f"\n  --- Composite HI 통계 ---")
    print(f"  min: {min(composites):.4f}")
    print(f"  max: {max(composites):.4f}")
    print(f"  mean: {np.mean(composites):.4f}")
    print(f"  median: {np.median(composites):.4f}")
    print(f"  std: {np.std(composites):.4f}")

    print(f"\n  --- Anomaly Score 통계 ---")
    print(f"  min: {min(scores):.4f}")
    print(f"  max: {max(scores):.4f}")
    print(f"  mean: {np.mean(scores):.4f}")
    print(f"  median: {np.median(scores):.4f}")

    print(f"\n  --- Health State 분포 ---")
    for state in ["normal", "watch", "warning", "critical"]:
        count = states.count(state)
        pct = count / len(states) * 100
        print(f"  {state:10s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  --- Anomaly Detection ---")
    detected_count = sum(detected)
    print(f"  detected: {detected_count}/{len(detected)} ({detected_count/len(detected)*100:.1f}%)")

    # 첫 감지 시점
    first_detected_idx = None
    for i, d in enumerate(detected):
        if d:
            first_detected_idx = i
            break

    if first_detected_idx is not None:
        total_snapshots = len(results)
        remaining = total_snapshots - first_detected_idx
        pct_remaining = remaining / total_snapshots * 100
        print(f"  첫 감지: idx={first_detected_idx}/{total_snapshots} ({pct_remaining:.1f}% 남음)")
        print(f"  첫 감지 시점: {timestamps[first_detected_idx]}")
        print(f"  마지막 시점: {timestamps[-1]}")
    else:
        print(f"  [WARN] 이상 미감지!")

    # 개별 HI 스파이크 분석
    print(f"\n  --- 개별 HI 최대값 ---")
    hi_keys = list(HI_FEATURE_KEYS.keys())
    for key in hi_keys:
        vals = [r["individual_hi"][key] for r in results]
        print(f"  {key:20s}: min={min(vals):.3f}  max={max(vals):.3f}  "
              f"mean={np.mean(vals):.3f}  >2.0: {sum(1 for v in vals if v >= 2.0)}/{len(vals)}")

    # 시간별 추이 (처음/중간/마지막 10개)
    print(f"\n  --- 시간별 추이 (처음 5 / 중간 5 / 마지막 5) ---")
    print(f"  {'idx':>5s}  {'timestamp':>20s}  {'comp_hi':>8s}  {'score':>8s}  {'state':>10s}  {'detected':>8s}")
    show_indices = (
        list(range(min(5, len(results))))
        + list(range(len(results)//2 - 2, len(results)//2 + 3))
        + list(range(max(0, len(results)-5), len(results)))
    )
    show_indices = sorted(set(i for i in show_indices if 0 <= i < len(results)))
    prev_i = -1
    for i in show_indices:
        if prev_i >= 0 and i - prev_i > 1:
            print(f"  {'...':>5s}")
        r = results[i]
        print(f"  {r['idx']:5d}  {str(r['timestamp']):>20s}  "
              f"{r['composite_hi']:8.3f}  {r['anomaly_score']:8.4f}  "
              f"{r['health_state']:>10s}  {str(r['anomaly_detected']):>8s}")
        prev_i = i

    # Breakpoint 적합성 분석
    print(f"\n  --- Breakpoint 적합성 ---")
    normal_his = [c for c, s in zip(composites, states) if s == "normal"]
    abnormal_his = [c for c, d in zip(composites, detected) if d]
    if normal_his:
        print(f"  정상 구간 HI 범위: {min(normal_his):.3f} ~ {max(normal_his):.3f}")
    if abnormal_his:
        print(f"  이상 감지 HI 범위: {min(abnormal_his):.3f} ~ {max(abnormal_his):.3f}")

    bp_1 = ANOMALY_HI_BREAKPOINTS[0][0]  # 1.0
    bp_2 = ANOMALY_HI_BREAKPOINTS[1][0]  # 2.0
    below_bp1 = sum(1 for c in composites if c <= bp_1)
    between = sum(1 for c in composites if bp_1 < c <= bp_2)
    above_bp2 = sum(1 for c in composites if c > bp_2)
    print(f"  HI <= {bp_1}: {below_bp1} ({below_bp1/len(composites)*100:.1f}%)")
    print(f"  {bp_1} < HI <= {bp_2}: {between} ({between/len(composites)*100:.1f}%)")
    print(f"  HI > {bp_2}: {above_bp2} ({above_bp2/len(composites)*100:.1f}%)")

    return {
        "test_set": test_set_id,
        "bearing_id": bearing_id,
        "fault_type": bearing_info["fault_type"],
        "total_snapshots": len(results),
        "composites": composites,
        "scores": scores,
        "states": states,
        "detected": detected,
        "first_detected_idx": first_detected_idx,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Part 3a IMS 데이터셋 검증")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="IMS 데이터셋 경로"
    )
    parser.add_argument(
        "--interval", type=int, default=1,
        help="스냅샷 샘플링 간격 (기본: 1=전부)"
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"[ERROR] 데이터셋 경로가 없습니다: {args.data_dir}")
        print("  python scripts/download_ims_dataset.py 를 먼저 실행하세요.")
        return 1

    # 분석 대상 정의
    targets = [
        ("1st_test", "BRG-003"),  # 내륜 결함, 35일
        ("1st_test", "BRG-004"),  # 전동체 결함, 35일
        ("2nd_test", "BRG-001"),  # 외륜 결함, 7일 (급속)
        ("3rd_test", "BRG-003"),  # 외륜 결함, 30일
    ]

    all_results = []
    for test_set_id, bearing_id in targets:
        test_dir = args.data_dir / test_set_id
        if not test_dir.exists():
            print(f"\n[WARN] {test_set_id} 디렉토리 없음. 건너뜀.")
            continue
        result = analyze_test_set(test_set_id, bearing_id, args.data_dir, args.interval)
        if result:
            all_results.append(result)

    # 종합 요약
    print(f"\n{'='*70}")
    print(f"  종합 요약")
    print(f"{'='*70}")
    for r in all_results:
        fault = r["fault_type"]
        total = r["total_snapshots"]
        first = r["first_detected_idx"]
        states = r["states"]
        detected_pct = sum(r["detected"]) / total * 100

        if first is not None:
            lead_pct = (total - first) / total * 100
            print(f"  {r['test_set']}/{r['bearing_id']} ({fault:15s}): "
                  f"감지 {detected_pct:5.1f}%  |  첫 감지 리드타임 {lead_pct:5.1f}%  |  "
                  f"N={states.count('normal')} W={states.count('watch')} "
                  f"Wn={states.count('warning')} C={states.count('critical')}")
        else:
            print(f"  {r['test_set']}/{r['bearing_id']} ({fault:15s}): 이상 미감지!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
