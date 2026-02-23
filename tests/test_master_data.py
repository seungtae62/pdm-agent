"""기준정보(Master Data) 테스트."""

from __future__ import annotations

import pytest

from master_data.equipment import get_equipment, EQUIPMENT
from master_data.bearing import get_bearing, list_bearings, BEARINGS
from master_data.sensor import get_sensor, list_sensors, SENSORS


class TestEquipment:
    def test_get_equipment(self):
        eq = get_equipment("IMS-TESTRIG-01")
        assert eq["equipment_name"] == "IMS Bearing Test Rig #1"
        assert eq["shaft_rpm"] == 2000

    def test_get_equipment_invalid(self):
        with pytest.raises(KeyError):
            get_equipment("INVALID")

    def test_all_equipment_have_required_fields(self):
        required = {"equipment_id", "equipment_name", "location", "shaft_rpm"}
        for eid, eq in EQUIPMENT.items():
            assert required.issubset(eq.keys()), f"{eid} missing fields"


class TestBearing:
    def test_get_bearing(self):
        brg = get_bearing("IMS-TESTRIG-01", "BRG-003")
        assert brg["model"] == "Rexnord ZA-2115"
        assert brg["known_fault_type"] == "inner_race"
        assert "BPFO" in brg["defect_frequencies_hz"]

    def test_get_bearing_invalid(self):
        with pytest.raises(KeyError):
            get_bearing("IMS-TESTRIG-01", "BRG-999")

    def test_list_bearings(self):
        brgs = list_bearings("IMS-TESTRIG-01")
        assert len(brgs) == 4
        ids = {b["bearing_id"] for b in brgs}
        assert "BRG-003" in ids
        assert "BRG-004" in ids

    def test_all_bearings_have_defect_frequencies(self):
        for key, brg in BEARINGS.items():
            assert "defect_frequencies_hz" in brg, f"{key} missing defect_frequencies_hz"
            freq = brg["defect_frequencies_hz"]
            assert {"BPFO", "BPFI", "BSF", "FTF"} == set(freq.keys())


class TestSensor:
    def test_get_sensor(self):
        s = get_sensor("IMS-TESTRIG-01", "BRG-003", "ch0")
        assert s["channel_index"] == 4
        assert s["sampling_rate_hz"] == 20000

    def test_get_sensor_invalid(self):
        with pytest.raises(KeyError):
            get_sensor("IMS-TESTRIG-01", "BRG-003", "ch9")

    def test_list_sensors(self):
        sensors = list_sensors("IMS-TESTRIG-01", "BRG-003")
        assert len(sensors) == 2
        channels = {s["channel"] for s in sensors}
        assert channels == {"ch0", "ch1"}

    def test_single_channel_bearing(self):
        sensors = list_sensors("IMS-TESTRIG-02", "BRG-001")
        assert len(sensors) == 1
        assert sensors[0]["channel"] == "ch0"
