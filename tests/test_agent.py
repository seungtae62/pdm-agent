"""PdM Agent 테스트 — State, 노드, 그래프 구조."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from agent.config import AgentConfig
from agent.state import PdMAgentState
from agent.nodes.load_memory import load_memory
from agent.nodes.parse_diagnosis import parse_diagnosis, _extract_json
from agent.nodes.save_memory import save_memory
from agent.prompts.system_prompt import load_system_prompt
from agent.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# 더미 데이터
# ---------------------------------------------------------------------------

DUMMY_PAYLOAD = {
    "event_id": "EVT-20260223-0001",
    "timestamp": "2026-02-23T12:00:00",
    "event_type": "anomaly_alert",
    "edge_node_id": "EDGE-001",
    "equipment_id": "IMS-TESTRIG-01",
    "bearing_id": "BRG-003",
    "sensor_channels": ["ch0", "ch1"],
    "anomaly_detection_result": {
        "model_id": "rule_v1",
        "anomaly_detected": True,
        "anomaly_score": 0.89,
        "anomaly_threshold": 0.65,
        "health_state": "warning",
        "confidence": 0.87,
    },
    "current_features": {
        "snapshot_timestamp": "2026-02-23T12:00:00",
        "time_domain": {"ch0": {"rms": 0.166, "kurtosis": 4.04}},
        "frequency_domain": {"ch0": {"bpfi_amplitude": 0.0014}},
    },
    "ml_rul_prediction": {
        "prediction_status": "not_applicable",
    },
}


def _make_state(**overrides) -> PdMAgentState:
    """테스트용 State 생성."""
    state: PdMAgentState = {
        "event_payload": DUMMY_PAYLOAD,
        "memory_context": {},
        "messages": [],
        "diagnosis_result": {},
        "tool_calls_count": 0,
        "deep_research_activated": False,
        "report": "",
        "work_order": "",
        "next_action": "",
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_from_env_defaults(self):
        config = AgentConfig()
        assert config.llm_model == "gpt-4o"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.max_tool_calls == 10

    def test_postgres_dsn(self):
        config = AgentConfig(
            postgres_user="user",
            postgres_password="pass",
            postgres_host="db",
            postgres_port=5432,
            postgres_db="testdb",
        )
        assert config.postgres_dsn == "postgresql://user:pass@db:5432/testdb"


# ---------------------------------------------------------------------------
# load_memory
# ---------------------------------------------------------------------------


class TestLoadMemory:
    def test_without_store(self):
        state = _make_state()
        result = load_memory(state, store=None)
        assert "memory_context" in result
        assert result["memory_context"]["equipment_id"] == "IMS-TESTRIG-01"
        assert result["memory_context"]["history_summary"] == ""

    def test_extracts_ids_from_payload(self):
        state = _make_state()
        result = load_memory(state, store=None)
        ctx = result["memory_context"]
        assert ctx["equipment_id"] == "IMS-TESTRIG-01"
        assert ctx["bearing_id"] == "BRG-003"


# ---------------------------------------------------------------------------
# parse_diagnosis
# ---------------------------------------------------------------------------


class TestParseDiagnosis:
    def test_extract_json_from_code_block(self):
        text = '''분석 결과:
```json
{"fault_type": "inner_race", "fault_stage": 2, "risk_level": "warning",
 "degradation_speed": "normal", "recommendation": "모니터링 강화",
 "uncertainty_notes": "", "reasoning_summary": "BPFI 상승",
 "rul_assessment": {"ml_rul_hours": null, "agent_assessment": "2~3주", "confidence_level": "medium"}}
```'''
        result = _extract_json(text)
        assert result is not None
        assert result["fault_type"] == "inner_race"
        assert result["risk_level"] == "warning"

    def test_extract_json_without_code_block(self):
        text = '결과: {"fault_type": "none", "fault_stage": 0, "risk_level": "normal"}'
        result = _extract_json(text)
        assert result is not None
        assert result["fault_type"] == "none"

    def test_extract_json_no_match(self):
        text = "JSON이 없는 텍스트입니다."
        result = _extract_json(text)
        assert result is None

    def test_parse_diagnosis_with_ai_message(self):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content='```json\n{"fault_type": "outer_race", "fault_stage": 3, "risk_level": "critical"}\n```')
        state = _make_state(messages=[msg])
        result = parse_diagnosis(state)
        assert result["diagnosis_result"]["fault_type"] == "outer_race"
        assert result["diagnosis_result"]["risk_level"] == "critical"

    def test_parse_diagnosis_fills_defaults(self):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content='```json\n{"fault_type": "none", "risk_level": "normal"}\n```')
        state = _make_state(messages=[msg])
        result = parse_diagnosis(state)
        d = result["diagnosis_result"]
        assert "fault_stage" in d
        assert "degradation_speed" in d
        assert "rul_assessment" in d

    def test_parse_diagnosis_no_messages(self):
        state = _make_state(messages=[])
        result = parse_diagnosis(state)
        assert result["diagnosis_result"]["fault_type"] == "unknown"


# ---------------------------------------------------------------------------
# save_memory
# ---------------------------------------------------------------------------


class TestSaveMemory:
    def test_without_store(self):
        state = _make_state(
            diagnosis_result={"fault_type": "inner_race", "risk_level": "warning"}
        )
        result = save_memory(state, store=None)
        assert result == {}


# ---------------------------------------------------------------------------
# MemoryStore.summarize_history
# ---------------------------------------------------------------------------


class TestSummarizeHistory:
    def test_empty_records(self):
        assert MemoryStore.summarize_history([]) == ""

    def test_with_records(self):
        records = [
            {
                "event_timestamp": datetime(2026, 2, 22, 10, 0),
                "fault_type": "inner_race",
                "fault_stage": 2,
                "risk_level": "watch",
                "reasoning_summary": "BPFI 상승 감지",
            },
            {
                "event_timestamp": datetime(2026, 2, 20, 10, 0),
                "fault_type": "none",
                "fault_stage": 0,
                "risk_level": "normal",
                "reasoning_summary": "정상 판정",
            },
        ]
        summary = MemoryStore.summarize_history(records)
        assert "2건" in summary
        assert "inner_race" in summary
        assert "normal" in summary


# ---------------------------------------------------------------------------
# system_prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_load_returns_nonempty(self):
        prompt = load_system_prompt()
        assert len(prompt) > 100
        assert "PdM Agent" in prompt or "예지보전" in prompt


# ---------------------------------------------------------------------------
# graph 구조 (빌드 테스트 — LLM 호출 없이)
# ---------------------------------------------------------------------------


class TestGraphStructure:
    def test_route_after_reasoning_call_tool(self):
        from agent.graph import _route_after_reasoning
        state = _make_state(next_action="call_tool", tool_calls_count=0)
        assert _route_after_reasoning(state) == "tool_executor"

    def test_route_after_reasoning_generate_report(self):
        from agent.graph import _route_after_reasoning
        state = _make_state(next_action="generate_report", tool_calls_count=0)
        assert _route_after_reasoning(state) == "parse_diagnosis"

    def test_route_after_reasoning_safety(self):
        from agent.graph import _route_after_reasoning
        state = _make_state(next_action="call_tool", tool_calls_count=11)
        assert _route_after_reasoning(state) == "parse_diagnosis"

    def test_route_after_report_warning(self):
        from agent.graph import _route_after_report
        state = _make_state(diagnosis_result={"risk_level": "warning"})
        assert _route_after_report(state) == "generate_work_order"

    def test_route_after_report_normal(self):
        from agent.graph import _route_after_report
        state = _make_state(diagnosis_result={"risk_level": "normal"})
        assert _route_after_report(state) == "save_memory"
