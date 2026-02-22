"""이벤트 페이로드 생성기 테스트."""

from __future__ import annotations

from datetime import datetime

import pytest

from edge.anomaly_detection import AnomalyResult, RuleCheckDetail
from edge.metadata import get_channel_indices, get_edge_node_id, get_equipment_meta
from edge.payload_generator import build_event_payload, _round_dict


# ---------------------------------------------------------------------------
# metadata 테스트
# ---------------------------------------------------------------------------


class TestMetadata:
    """메타데이터 모듈 테스트."""

    def test_get_equipment_meta_1st_test_brg003(self):
        meta = get_equipment_meta("1st_test", "BRG-003")
        assert meta["equipment_id"] == "IMS-TESTRIG-01"
        assert meta["bearing"]["bearing_id"] == "BRG-003"
        assert meta["bearing"]["model"] == "Rexnord ZA-2115"
        assert meta["sensor_config"]["sensor_count"] == 8
        assert meta["sensor_config"]["channels"] == ["ch0", "ch1"]

    def test_get_equipment_meta_2nd_test_brg001(self):
        meta = get_equipment_meta("2nd_test", "BRG-001")
        assert meta["equipment_id"] == "IMS-TESTRIG-02"
        assert meta["bearing"]["bearing_id"] == "BRG-001"
        assert meta["sensor_config"]["channels"] == ["ch0"]

    def test_get_equipment_meta_invalid(self):
        with pytest.raises(KeyError):
            get_equipment_meta("1st_test", "BRG-999")

    def test_get_edge_node_id(self):
        assert get_edge_node_id("1st_test") == "EDGE-001"
        assert get_edge_node_id("2nd_test") == "EDGE-002"

    def test_get_channel_indices(self):
        assert get_channel_indices("1st_test", "BRG-003") == [4, 5]
        assert get_channel_indices("2nd_test", "BRG-001") == [0]


# ---------------------------------------------------------------------------
# payload_generator 테스트
# ---------------------------------------------------------------------------


def _make_dummy_features() -> dict:
    """테스트용 더미 특징량."""
    td = {
        "rms": 0.1, "peak": 0.3, "peak_to_peak": 0.6,
        "crest_factor": 3.0, "kurtosis": 3.0, "skewness": 0.0,
        "standard_deviation": 0.1, "mean": 0.0, "shape_factor": 1.25,
    }
    fd = {
        "bpfo_amplitude": 0.01, "bpfi_amplitude": 0.02,
        "bsf_amplitude": 0.005, "ftf_amplitude": 0.001,
        "bpfo_harmonics_2x": 0.005, "bpfi_harmonics_2x": 0.01,
        "spectral_energy_total": 0.5, "spectral_energy_high_freq_band": 0.1,
        "dominant_frequency_hz": 296.9,
        "sideband_presence": False, "sideband_spacing_hz": 0.0,
        "sideband_count": 0,
    }
    return {
        "time_domain": {"ch0": td},
        "frequency_domain": {"ch0": fd},
    }


def _make_dummy_anomaly_result(detected: bool = False) -> AnomalyResult:
    """테스트용 더미 이상감지 결과."""
    return AnomalyResult(
        model_id="rule_stat_ae_v1",
        anomaly_detected=detected,
        anomaly_score=0.3 if not detected else 0.8,
        anomaly_threshold=0.65,
        health_state="normal" if not detected else "warning",
        confidence=0.85,
        rule_detail=RuleCheckDetail(
            composite_hi_score=0.2,
            spike_score=0.0,
            spiked_keys=[],
        ),
    )


class TestBuildEventPayload:
    """build_event_payload 테스트."""

    def test_basic_structure(self):
        payload = build_event_payload(
            test_set_id="1st_test",
            bearing_id="BRG-003",
            snapshot_timestamp=datetime(2003, 10, 27, 12, 0, 0),
            operation_start_date=datetime(2003, 10, 22, 0, 0, 0),
            features=_make_dummy_features(),
            anomaly_result=_make_dummy_anomaly_result(detected=False),
        )

        # 최상위 필드
        assert payload["event_id"].startswith("EVT-20031027-")
        assert payload["event_type"] == "periodic_monitoring"
        assert payload["edge_node_id"] == "EDGE-001"

        # equipment_meta
        meta = payload["equipment_meta"]
        assert meta["equipment_id"] == "IMS-TESTRIG-01"
        assert meta["operation_days_elapsed"] == 5
        assert meta["total_running_hours"] > 0
        assert meta["bearing"]["model"] == "Rexnord ZA-2115"

        # anomaly_detection_result
        anom = payload["anomaly_detection_result"]
        assert anom["anomaly_detected"] is False
        assert anom["health_state"] == "normal"

        # current_features
        feat = payload["current_features"]
        assert "ch0" in feat["time_domain"]
        assert "rms" in feat["time_domain"]["ch0"]
        assert "bpfo_amplitude" in feat["frequency_domain"]["ch0"]

        # ml_rul_prediction
        rul = payload["ml_rul_prediction"]
        assert rul["prediction_status"] == "not_applicable"

    def test_anomaly_alert_type(self):
        payload = build_event_payload(
            test_set_id="2nd_test",
            bearing_id="BRG-001",
            snapshot_timestamp=datetime(2004, 2, 16, 12, 0, 0),
            operation_start_date=datetime(2004, 2, 12, 0, 0, 0),
            features=_make_dummy_features(),
            anomaly_result=_make_dummy_anomaly_result(detected=True),
        )

        assert payload["event_type"] == "anomaly_alert"
        assert payload["anomaly_detection_result"]["anomaly_detected"] is True

    def test_operation_days_calculation(self):
        payload = build_event_payload(
            test_set_id="1st_test",
            bearing_id="BRG-003",
            snapshot_timestamp=datetime(2003, 11, 16, 12, 0, 0),
            operation_start_date=datetime(2003, 10, 22, 0, 0, 0),
            features=_make_dummy_features(),
            anomaly_result=_make_dummy_anomaly_result(),
        )

        assert payload["equipment_meta"]["operation_days_elapsed"] == 25


class TestRoundDict:
    """_round_dict 유틸 테스트."""

    def test_nested_rounding(self):
        d = {"a": {"b": 1.123456789}, "c": 2.999999}
        result = _round_dict(d, ndigits=4)
        assert result["a"]["b"] == 1.1235
        assert result["c"] == 3.0

    def test_non_float_preserved(self):
        d = {"flag": True, "count": 5, "name": "test"}
        result = _round_dict(d)
        assert result == d
