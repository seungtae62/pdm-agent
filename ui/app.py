"""PdM Agent Streamlit 데모 UI.

시나리오 선택 -> 에이전트 실행 모니터링 -> 진단 결과 대시보드 -> 리포트/작업지시서.
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

# ui/ 디렉토리를 path에 추가 (components, styles import용)
sys.path.insert(0, os.path.dirname(__file__))

from api_client import stream_events, submit_event
from styles import GLOBAL_CSS, SIDEBAR_CSS
from components import (
    render_diagnosis_cards,
    render_equipment_info,
    render_report,
    render_thoughts,
    render_thoughts_live,
    render_work_order,
)

# ──────────────────────────── 페이지 설정 ────────────────────────────

st.set_page_config(
    page_title="PdM Agent",
    page_icon="PdM",
    layout="wide",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

# ──────────────────────────── 샘플 시나리오 ────────────────────────────

SAMPLE_SCENARIOS: dict[str, dict] = {
    "SC-001: 정상 구간 (Normal)": {
        "event_id": "EVT-20260314-0001",
        "timestamp": "2026-03-14T10:00:00+09:00",
        "event_type": "periodic_monitoring",
        "edge_node_id": "EDGE-IMS-01",
        "equipment_meta": {
            "equipment_id": "IMS-TESTRIG-01",
            "equipment_name": "IMS Bearing Test Rig",
            "location": "Lab-A",
            "shaft_rpm": 2000,
            "radial_load_lbs": 6000,
            "operation_start_date": "2026-01-01",
            "bearing": {
                "bearing_id": "BRG-001",
                "position": "Bearing 1",
                "install_date": "2026-01-01",
                "model": "Rexnord ZA-2115",
                "type": "Double Row Bearing",
                "rolling_elements_count": 16,
                "ball_diameter_inch": 0.331,
                "pitch_diameter_inch": 2.815,
                "contact_angle_deg": 15.17,
                "defect_frequencies_hz": {
                    "BPFO": 236.4,
                    "BPFI": 296.9,
                    "BSF": 139.7,
                    "FTF": 14.8,
                },
            },
            "sensor_config": {
                "sensor_count": 2,
                "channels": ["ch1_x", "ch2_y"],
                "sensor_type": "accelerometer",
                "sampling_rate_hz": 20480,
                "samples_per_snapshot": 20480,
                "snapshot_interval_min": 10,
            },
            "operation_days_elapsed": 10,
            "total_running_hours": 240.0,
        },
        "anomaly_detection_result": {
            "model_id": "AE-v1",
            "anomaly_detected": False,
            "anomaly_score": 0.15,
            "anomaly_threshold": 0.5,
            "health_state": "normal",
            "confidence": 0.95,
        },
        "current_features": {
            "snapshot_timestamp": "2026-03-14T10:00:00+09:00",
            "time_domain": {
                "ch1_x": {
                    "rms": 0.08, "peak": 0.25, "peak_to_peak": 0.50,
                    "crest_factor": 3.1, "kurtosis": 3.0, "skewness": 0.01,
                    "standard_deviation": 0.08, "mean": 0.001, "shape_factor": 1.25,
                },
            },
            "frequency_domain": {
                "ch1_x": {
                    "bpfo_amplitude": 0.005, "bpfi_amplitude": 0.004,
                    "bsf_amplitude": 0.003, "ftf_amplitude": 0.002,
                    "bpfo_harmonics_2x": 0.002, "bpfi_harmonics_2x": 0.001,
                    "spectral_energy_total": 0.1, "spectral_energy_high_freq_band": 0.01,
                    "dominant_frequency_hz": 33.3, "sideband_presence": False,
                    "sideband_spacing_hz": 0.0, "sideband_count": 0,
                },
            },
        },
        "ml_rul_prediction": None,
    },
    "SC-002: 내륜 결함 초기 (Watch)": {
        "event_id": "EVT-20260314-0002",
        "timestamp": "2026-03-14T10:00:00+09:00",
        "event_type": "anomaly_alert",
        "edge_node_id": "EDGE-IMS-01",
        "equipment_meta": {
            "equipment_id": "IMS-TESTRIG-01",
            "equipment_name": "IMS Bearing Test Rig",
            "location": "Lab-A",
            "shaft_rpm": 2000,
            "radial_load_lbs": 6000,
            "operation_start_date": "2026-01-01",
            "bearing": {
                "bearing_id": "BRG-003",
                "position": "Bearing 3",
                "install_date": "2026-01-01",
                "model": "Rexnord ZA-2115",
                "type": "Double Row Bearing",
                "rolling_elements_count": 16,
                "ball_diameter_inch": 0.331,
                "pitch_diameter_inch": 2.815,
                "contact_angle_deg": 15.17,
                "defect_frequencies_hz": {
                    "BPFO": 236.4,
                    "BPFI": 296.9,
                    "BSF": 139.7,
                    "FTF": 14.8,
                },
            },
            "sensor_config": {
                "sensor_count": 2,
                "channels": ["ch1_x", "ch2_y"],
                "sensor_type": "accelerometer",
                "sampling_rate_hz": 20480,
                "samples_per_snapshot": 20480,
                "snapshot_interval_min": 10,
            },
            "operation_days_elapsed": 25,
            "total_running_hours": 600.0,
        },
        "anomaly_detection_result": {
            "model_id": "AE-v1",
            "anomaly_detected": True,
            "anomaly_score": 0.62,
            "anomaly_threshold": 0.5,
            "health_state": "watch",
            "confidence": 0.88,
        },
        "current_features": {
            "snapshot_timestamp": "2026-03-14T10:00:00+09:00",
            "time_domain": {
                "ch1_x": {
                    "rms": 0.15, "peak": 0.55, "peak_to_peak": 1.10,
                    "crest_factor": 3.7, "kurtosis": 4.2, "skewness": 0.3,
                    "standard_deviation": 0.15, "mean": 0.002, "shape_factor": 1.35,
                },
            },
            "frequency_domain": {
                "ch1_x": {
                    "bpfo_amplitude": 0.008, "bpfi_amplitude": 0.045,
                    "bsf_amplitude": 0.006, "ftf_amplitude": 0.003,
                    "bpfo_harmonics_2x": 0.003, "bpfi_harmonics_2x": 0.020,
                    "spectral_energy_total": 0.25, "spectral_energy_high_freq_band": 0.05,
                    "dominant_frequency_hz": 296.9, "sideband_presence": True,
                    "sideband_spacing_hz": 33.3, "sideband_count": 2,
                },
            },
        },
        "ml_rul_prediction": None,
    },
    "SC-003: 내륜 결함 진행 (Warning)": {
        "event_id": "EVT-20260314-0003",
        "timestamp": "2026-03-14T10:00:00+09:00",
        "event_type": "anomaly_alert",
        "edge_node_id": "EDGE-IMS-01",
        "equipment_meta": {
            "equipment_id": "IMS-TESTRIG-01",
            "equipment_name": "IMS Bearing Test Rig",
            "location": "Lab-A",
            "shaft_rpm": 2000,
            "radial_load_lbs": 6000,
            "operation_start_date": "2026-01-01",
            "bearing": {
                "bearing_id": "BRG-003",
                "position": "Bearing 3",
                "install_date": "2026-01-01",
                "model": "Rexnord ZA-2115",
                "type": "Double Row Bearing",
                "rolling_elements_count": 16,
                "ball_diameter_inch": 0.331,
                "pitch_diameter_inch": 2.815,
                "contact_angle_deg": 15.17,
                "defect_frequencies_hz": {
                    "BPFO": 236.4,
                    "BPFI": 296.9,
                    "BSF": 139.7,
                    "FTF": 14.8,
                },
            },
            "sensor_config": {
                "sensor_count": 2,
                "channels": ["ch1_x", "ch2_y"],
                "sensor_type": "accelerometer",
                "sampling_rate_hz": 20480,
                "samples_per_snapshot": 20480,
                "snapshot_interval_min": 10,
            },
            "operation_days_elapsed": 32,
            "total_running_hours": 768.0,
        },
        "anomaly_detection_result": {
            "model_id": "AE-v1",
            "anomaly_detected": True,
            "anomaly_score": 0.85,
            "anomaly_threshold": 0.5,
            "health_state": "warning",
            "confidence": 0.92,
        },
        "current_features": {
            "snapshot_timestamp": "2026-03-14T10:00:00+09:00",
            "time_domain": {
                "ch1_x": {
                    "rms": 0.35, "peak": 1.40, "peak_to_peak": 2.80,
                    "crest_factor": 4.0, "kurtosis": 6.5, "skewness": 0.8,
                    "standard_deviation": 0.35, "mean": 0.005, "shape_factor": 1.50,
                },
            },
            "frequency_domain": {
                "ch1_x": {
                    "bpfo_amplitude": 0.012, "bpfi_amplitude": 0.120,
                    "bsf_amplitude": 0.015, "ftf_amplitude": 0.005,
                    "bpfo_harmonics_2x": 0.005, "bpfi_harmonics_2x": 0.055,
                    "spectral_energy_total": 0.60, "spectral_energy_high_freq_band": 0.18,
                    "dominant_frequency_hz": 296.9, "sideband_presence": True,
                    "sideband_spacing_hz": 33.3, "sideband_count": 4,
                },
            },
        },
        "ml_rul_prediction": {
            "model_id": "RUL-LSTM-v1",
            "predicted_rul_hours": 210.0,
            "confidence_interval_hours": {"lower": 150.0, "upper": 280.0},
            "prediction_status": "predicted",
            "reason": None,
        },
    },
    "SC-004: 외륜 결함 심각 (Critical)": {
        "event_id": "EVT-20260314-0004",
        "timestamp": "2026-03-14T10:00:00+09:00",
        "event_type": "anomaly_alert",
        "edge_node_id": "EDGE-IMS-01",
        "equipment_meta": {
            "equipment_id": "IMS-TESTRIG-01",
            "equipment_name": "IMS Bearing Test Rig",
            "location": "Lab-A",
            "shaft_rpm": 2000,
            "radial_load_lbs": 6000,
            "operation_start_date": "2026-01-01",
            "bearing": {
                "bearing_id": "BRG-004",
                "position": "Bearing 4",
                "install_date": "2026-01-01",
                "model": "Rexnord ZA-2115",
                "type": "Double Row Bearing",
                "rolling_elements_count": 16,
                "ball_diameter_inch": 0.331,
                "pitch_diameter_inch": 2.815,
                "contact_angle_deg": 15.17,
                "defect_frequencies_hz": {
                    "BPFO": 236.4,
                    "BPFI": 296.9,
                    "BSF": 139.7,
                    "FTF": 14.8,
                },
            },
            "sensor_config": {
                "sensor_count": 2,
                "channels": ["ch1_x", "ch2_y"],
                "sensor_type": "accelerometer",
                "sampling_rate_hz": 20480,
                "samples_per_snapshot": 20480,
                "snapshot_interval_min": 10,
            },
            "operation_days_elapsed": 35,
            "total_running_hours": 840.0,
        },
        "anomaly_detection_result": {
            "model_id": "AE-v1",
            "anomaly_detected": True,
            "anomaly_score": 0.96,
            "anomaly_threshold": 0.5,
            "health_state": "critical",
            "confidence": 0.97,
        },
        "current_features": {
            "snapshot_timestamp": "2026-03-14T10:00:00+09:00",
            "time_domain": {
                "ch1_x": {
                    "rms": 0.80, "peak": 3.50, "peak_to_peak": 7.00,
                    "crest_factor": 4.4, "kurtosis": 12.0, "skewness": 1.5,
                    "standard_deviation": 0.80, "mean": 0.010, "shape_factor": 1.65,
                },
            },
            "frequency_domain": {
                "ch1_x": {
                    "bpfo_amplitude": 0.250, "bpfi_amplitude": 0.030,
                    "bsf_amplitude": 0.040, "ftf_amplitude": 0.010,
                    "bpfo_harmonics_2x": 0.120, "bpfi_harmonics_2x": 0.010,
                    "spectral_energy_total": 1.50, "spectral_energy_high_freq_band": 0.55,
                    "dominant_frequency_hz": 236.4, "sideband_presence": True,
                    "sideband_spacing_hz": 33.3, "sideband_count": 6,
                },
            },
        },
        "ml_rul_prediction": {
            "model_id": "RUL-LSTM-v1",
            "predicted_rul_hours": 48.0,
            "confidence_interval_hours": {"lower": 20.0, "upper": 80.0},
            "prediction_status": "predicted",
            "reason": None,
        },
    },
}


# ──────────────────────────── 세션 상태 초기화 ────────────────────────────

def _init_session_state() -> None:
    """세션 상태 기본값 초기화."""
    defaults = {
        "run_id": None,
        "status": "idle",  # idle / running / completed / failed
        "thoughts": [],  # list of {text, tool_calls, status}
        "diagnosis": {},
        "report": "",
        "work_order": {},
        "error_msg": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_session_state() -> None:
    """분석 결과 초기화."""
    st.session_state.run_id = None
    st.session_state.status = "idle"
    st.session_state.thoughts = []
    st.session_state.diagnosis = {}
    st.session_state.report = ""
    st.session_state.work_order = {}
    st.session_state.error_msg = ""


_init_session_state()


# ──────────────────────────── 사이드바 ────────────────────────────

with st.sidebar:
    st.markdown("## PdM Agent")
    st.markdown("---")

    # 시나리오 선택
    scenario_names = list(SAMPLE_SCENARIOS.keys())
    selected_scenario = st.selectbox(
        "시나리오",
        scenario_names,
        label_visibility="collapsed",
    )
    payload = SAMPLE_SCENARIOS[selected_scenario]

    # 설비 정보 + 이상감지
    render_equipment_info(payload)

    st.markdown("---")

    # 분석 시작 버튼
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button(
            "분석 시작",
            use_container_width=True,
            disabled=st.session_state.status == "running",
        )
    with col_btn2:
        reset_btn = st.button(
            "초기화",
            use_container_width=True,
        )

    st.markdown("---")

    # API 설정
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
    )

# ──────────────────────────── 초기화 처리 ────────────────────────────

if reset_btn:
    _reset_session_state()
    st.rerun()

# ──────────────────────────── 메인 영역 ────────────────────────────

st.markdown("## PdM Agent 진단 대시보드")
st.markdown("---")


# ──────────────────────────── 분석 실행 ────────────────────────────

def _run_analysis() -> None:
    """에이전트 분석 실행 (SSE 스트리밍)."""
    _reset_session_state()
    st.session_state.status = "running"

    # Placeholder 생성: 하나의 placeholder에서 모든 thought를 누적 렌더링
    thoughts_ph = st.empty()
    status_ph = st.empty()

    # 로컬 추적 변수
    thoughts: list[dict] = []
    current_text = ""
    in_reasoning = False

    def _render_live() -> None:
        """완료된 thoughts + 현재 streaming을 통합 렌더링."""
        render_thoughts_live(thoughts, current_text, thoughts_ph)

    def _start_new_thought() -> None:
        """이전 thought를 완료하고 새 thought 시작."""
        nonlocal current_text, in_reasoning
        if in_reasoning and current_text:
            thoughts.append({
                "text": current_text,
                "tool_calls": [],
                "status": "done",
            })
        current_text = ""
        in_reasoning = True
        _render_live()

    def _finish_current_thought() -> None:
        """현재 thought를 완료 처리."""
        nonlocal current_text, in_reasoning
        if in_reasoning and current_text:
            thoughts.append({
                "text": current_text,
                "tool_calls": [],
                "status": "done",
            })
        current_text = ""
        in_reasoning = False

    try:
        # 1. 이벤트 제출
        status_ph.info("이벤트 제출 중...")
        run_id = submit_event(api_url, payload)
        st.session_state.run_id = run_id
        status_ph.info(f"Run ID: {run_id} - 스트리밍 시작...")

        # 2. SSE 스트리밍
        pending_tool: dict | None = None

        for event in stream_events(api_url, run_id):
            event_type = event.get("event", "")

            if event_type == "run_started":
                status_ph.info("에이전트 실행 시작됨")

            elif event_type == "node_entered":
                node_name = event.get("node_name", "")
                if node_name == "reasoning":
                    _start_new_thought()
                elif in_reasoning:
                    # reasoning 밖으로 나감 → 현재 thought 완료
                    _finish_current_thought()

            elif event_type == "reasoning_token":
                token = event.get("token", "")
                current_text += token
                _render_live()

            elif event_type == "tool_call":
                pending_tool = {
                    "name": event.get("tool_name", ""),
                    "arguments": event.get("arguments", {}),
                    "result": None,
                }

            elif event_type == "tool_result":
                if pending_tool:
                    pending_tool["result"] = event.get("result", "")
                    # Tool 호출을 마지막 thought에 연결
                    if thoughts:
                        thoughts[-1]["tool_calls"].append(pending_tool)
                    _render_live()
                    pending_tool = None

            elif event_type == "diagnosis":
                st.session_state.diagnosis = event.get("diagnosis", {})

            elif event_type == "report_generated":
                st.session_state.report = event.get("report", "")

            elif event_type == "work_order_generated":
                st.session_state.work_order = event.get("work_order", {})

            elif event_type == "run_completed":
                _finish_current_thought()
                st.session_state.thoughts = thoughts
                st.session_state.status = "completed"
                return

            elif event_type == "error":
                _finish_current_thought()
                st.session_state.thoughts = thoughts
                st.session_state.status = "failed"
                st.session_state.error_msg = event.get("message", "")
                return

    except Exception as e:
        _finish_current_thought()
        st.session_state.thoughts = thoughts
        st.session_state.status = "failed"
        st.session_state.error_msg = str(e)


if start_btn:
    _run_analysis()
    st.rerun()


# ──────────────────────────── 결과 표시 (완료 후) ────────────────────────────

if st.session_state.status == "completed":
    # 추론 과정 (Thought 단위)
    if st.session_state.thoughts:
        st.caption(f"DEBUG: {len(st.session_state.thoughts)} thoughts")  # TODO: 확인 후 제거
        st.markdown("### 에이전트 추론 과정")
        render_thoughts(st.session_state.thoughts)

    # 진단 결과
    if st.session_state.diagnosis:
        st.markdown("### 진단 결과")
        render_diagnosis_cards(st.session_state.diagnosis)

    st.markdown("---")

    # 산출물 탭
    tab_names = ["분석 리포트"]
    if st.session_state.work_order:
        tab_names.append("작업지시서")

    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_report(st.session_state.report)

    if st.session_state.work_order and len(tabs) > 1:
        with tabs[1]:
            # PDF 다운로드 버튼 (우측 상단)
            try:
                src_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "src"
                )
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from utils.pdf import generate_work_order_pdf_bytes

                pdf_bytes = generate_work_order_pdf_bytes(st.session_state.work_order)
                wo_number = st.session_state.work_order.get(
                    "wo_number", "work_order"
                )
                _, btn_col = st.columns([4, 1])
                with btn_col:
                    st.download_button(
                        label="PDF 다운로드",
                        data=pdf_bytes,
                        file_name=f"{wo_number}_작업지시서.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"PDF 생성 실패: {e}")

            render_work_order(st.session_state.work_order)

elif st.session_state.status == "idle":
    st.info("좌측 사이드바에서 시나리오를 선택하고 '분석 시작'을 클릭하세요.")

elif st.session_state.status == "failed":
    error_msg = st.session_state.get("error_msg", "")
    st.error(f"분석 실패: {error_msg}" if error_msg else "분석 실패. API 서버 연결을 확인하세요.")
    # 실패 시에도 추론 과정이 있으면 표시
    if st.session_state.thoughts:
        st.markdown("### 에이전트 추론 과정")
        render_thoughts(st.session_state.thoughts)
