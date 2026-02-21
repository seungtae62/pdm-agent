"""Part 3 통합 검증 스크립트: Rule + Statistical + Autoencoder.

IMS 데이터셋 기반 전체 파이프라인 검증.
flat_features → HI → 3모듈 Anomaly Detection → Combiner

Usage:
    python scripts/validate_part3.py
    python scripts/validate_part3.py --data-dir /path/to/ims
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.anomaly_detection import compute_anomaly_baseline, detect_anomaly
from edge.autoencoder import train_autoencoder
from edge.config import (
    ANOMALY_HI_BREAKPOINTS,
    HI_BASELINE_SNAPSHOT_COUNT,
    HI_FEATURE_KEYS,
    MODEL_BASELINE_SNAPSHOT_COUNT,
)
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
    """단일 테스트셋/베어링에 대해 3모듈 파이프라인 실행 및 분석."""
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
        print(f"  [ERROR] 스냅샷 부족")
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

    # 3종 베이스라인 산출
    n_bl = HI_BASELINE_SNAPSHOT_COUNT
    n_ae = MODEL_BASELINE_SNAPSHOT_COUNT
    baseline = compute_baseline(flat_features_list, n_bl)
    anom_baseline = compute_anomaly_baseline(flat_features_list, n_bl)

    print(f"  Autoencoder 학습 중 (베이스라인 {min(n_ae, len(flat_features_list))}개)...",
          end="", flush=True)
    ae_baseline = train_autoencoder(flat_features_list, n_ae)
    print(f" 완료 (epochs={ae_baseline.final_epoch}, "
          f"recon_mean={ae_baseline.normal_recon_mean:.6f}, "
          f"std={ae_baseline.normal_recon_std:.6f})")

    # 3모듈 Anomaly Detection
    results = []
    for i, flat in enumerate(flat_features_list):
        hi = compute_health_indices(flat, baseline)
        anomaly = detect_anomaly(
            hi, flat, anom_baseline, ae_baseline,
        )
        results.append({
            "idx": i,
            "timestamp": timestamps[i],
            "composite_hi": hi.composite,
            "individual_hi": dict(hi.individual),
            "anomaly_score": anomaly.anomaly_score,
            "anomaly_detected": anomaly.anomaly_detected,
            "health_state": anomaly.health_state,
            "confidence": anomaly.confidence,
            "rule_score": max(
                anomaly.rule_detail.composite_hi_score,
                anomaly.rule_detail.spike_score,
            ),
            "stat_score": anomaly.stat_detail.stat_score if anomaly.stat_detail else None,
            "model_score": anomaly.model_detail.score if anomaly.model_detail else None,
            "recon_error": anomaly.model_detail.reconstruction_error if anomaly.model_detail else None,
        })

    # 분석 출력
    composites = [r["composite_hi"] for r in results]
    scores = [r["anomaly_score"] for r in results]
    states = [r["health_state"] for r in results]
    detected = [r["anomaly_detected"] for r in results]

    print(f"\n  --- Composite HI 통계 ---")
    print(f"  min={min(composites):.4f}  max={max(composites):.4f}  "
          f"mean={np.mean(composites):.4f}  median={np.median(composites):.4f}")

    print(f"\n  --- Anomaly Score 통계 ---")
    print(f"  min={min(scores):.4f}  max={max(scores):.4f}  "
          f"mean={np.mean(scores):.4f}  median={np.median(scores):.4f}")

    # 개별 모듈 score 통계
    rule_scores = [r["rule_score"] for r in results]
    stat_scores = [r["stat_score"] for r in results if r["stat_score"] is not None]
    model_scores = [r["model_score"] for r in results if r["model_score"] is not None]
    recon_errors = [r["recon_error"] for r in results if r["recon_error"] is not None]

    print(f"\n  --- 모듈별 Score 통계 ---")
    print(f"  Rule:  min={min(rule_scores):.4f}  max={max(rule_scores):.4f}  mean={np.mean(rule_scores):.4f}")
    if stat_scores:
        print(f"  Stat:  min={min(stat_scores):.4f}  max={max(stat_scores):.4f}  mean={np.mean(stat_scores):.4f}")
    if model_scores:
        print(f"  Model: min={min(model_scores):.4f}  max={max(model_scores):.4f}  mean={np.mean(model_scores):.4f}")
    if recon_errors:
        print(f"  Recon: min={min(recon_errors):.6f}  max={max(recon_errors):.6f}  mean={np.mean(recon_errors):.6f}")

    print(f"\n  --- Health State 분포 ---")
    for state in ["normal", "watch", "warning", "critical"]:
        count = states.count(state)
        pct = count / len(states) * 100
        print(f"  {state:10s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  --- Anomaly Detection ---")
    detected_count = sum(detected)
    print(f"  detected: {detected_count}/{len(detected)} ({detected_count/len(detected)*100:.1f}%)")

    first_detected_idx = None
    for i, d in enumerate(detected):
        if d:
            first_detected_idx = i
            break

    if first_detected_idx is not None:
        total_snapshots = len(results)
        remaining = total_snapshots - first_detected_idx
        pct_remaining = remaining / total_snapshots * 100
        print(f"  첫 감지: idx={first_detected_idx}/{total_snapshots} "
              f"({pct_remaining:.1f}% lead)")
    else:
        print(f"  [WARN] 이상 미감지!")

    # 시간별 추이 (처음 5 / 중간 5 / 마지막 5)
    print(f"\n  --- 시간별 추이 ---")
    print(f"  {'idx':>5s}  {'comp_hi':>8s}  {'rule':>6s}  {'stat':>6s}  {'model':>6s}  "
          f"{'recon':>10s}  {'score':>8s}  {'state':>10s}")
    show_indices = sorted(set(
        list(range(min(5, len(results))))
        + list(range(len(results)//2 - 2, len(results)//2 + 3))
        + list(range(max(0, len(results)-5), len(results)))
    ))
    show_indices = [i for i in show_indices if 0 <= i < len(results)]
    prev_i = -1
    for i in show_indices:
        if prev_i >= 0 and i - prev_i > 1:
            print(f"  {'...':>5s}")
        r = results[i]
        stat_s = f"{r['stat_score']:.4f}" if r['stat_score'] is not None else "   N/A"
        model_s = f"{r['model_score']:.4f}" if r['model_score'] is not None else "   N/A"
        recon_s = f"{r['recon_error']:.6f}" if r['recon_error'] is not None else "       N/A"
        print(f"  {r['idx']:5d}  {r['composite_hi']:8.3f}  {r['rule_score']:6.4f}  "
              f"{stat_s}  {model_s}  {recon_s}  {r['anomaly_score']:8.4f}  "
              f"{r['health_state']:>10s}")
        prev_i = i

    return {
        "test_set": test_set_id,
        "bearing_id": bearing_id,
        "fault_type": bearing_info["fault_type"],
        "total_snapshots": len(results),
        "states": states,
        "detected": detected,
        "first_detected_idx": first_detected_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Part 3 통합 IMS 검증")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--interval", type=int, default=1)
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"[ERROR] 데이터셋 경로 없음: {args.data_dir}")
        return 1

    targets = [
        ("1st_test", "BRG-003"),
        ("1st_test", "BRG-004"),
        ("2nd_test", "BRG-001"),
        ("3rd_test", "BRG-003"),
    ]

    all_results = []
    for test_set_id, bearing_id in targets:
        test_dir = args.data_dir / test_set_id
        if not test_dir.exists():
            print(f"\n[WARN] {test_set_id} 없음. 건너뜀.")
            continue
        result = analyze_test_set(test_set_id, bearing_id, args.data_dir, args.interval)
        if result:
            all_results.append(result)

    # 종합 요약
    print(f"\n{'='*70}")
    print(f"  종합 요약 (Part 3: Rule + Statistical + Autoencoder)")
    print(f"{'='*70}")
    fmt = "  {test}/{brg} ({fault:15s}): {snaps:>4d}snaps  det={det:5.1f}%  lead={lead:5.1f}%  N={n} W={w} Wn={wn} C={c}"
    for r in all_results:
        total = r["total_snapshots"]
        det_pct = sum(r["detected"]) / total * 100
        first = r["first_detected_idx"]
        lead_pct = ((total - first) / total * 100) if first is not None else 0.0
        s = r["states"]
        print(fmt.format(
            test=r["test_set"], brg=r["bearing_id"], fault=r["fault_type"],
            snaps=total, det=det_pct, lead=lead_pct,
            n=s.count("normal"), w=s.count("watch"),
            wn=s.count("warning"), c=s.count("critical"),
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
