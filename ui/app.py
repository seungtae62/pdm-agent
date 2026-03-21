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

import markdown as md

from api_client import stream_chat, stream_events, submit_chat, submit_event
from styles import CHAT_PANEL_CSS, GLOBAL_CSS, SIDEBAR_CSS


def _md_to_html(text: str) -> str:
    """마크다운 텍스트를 HTML로 변환."""
    return md.markdown(text, extensions=["tables", "fenced_code"])


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
st.markdown(CHAT_PANEL_CSS, unsafe_allow_html=True)

# ──────────────────────────── 샘플 시나리오 ────────────────────────────

SAMPLE_SCENARIOS: dict[str, dict] = {
    "SC-001: 정상 구간 (Normal)": {
        "event_id": "EVT-20260314-0001",
        "timestamp": "2026-03-14T10:00:00+09:00",
        "event_type": "periodic_monitoring",
        "edge_node_id": "EDGE-IMS-01",
        "equipment_meta": {
            "equipment_id": "LINE-A",
            "equipment_name": "ZA2115 Pillow Block Line A",
            "location": "라인 A / 제품 AX-01",
            "shaft_rpm": 2000,
            "radial_load_lbs": 6000,
            "operation_start_date": "2026-01-01",
            "bearing": {
                "bearing_id": "AX-01",
                "position": "Drive End",
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
                    "rms": 0.08,
                    "peak": 0.25,
                    "peak_to_peak": 0.50,
                    "crest_factor": 3.1,
                    "kurtosis": 3.0,
                    "skewness": 0.01,
                    "standard_deviation": 0.08,
                    "mean": 0.001,
                    "shape_factor": 1.25,
                },
            },
            "frequency_domain": {
                "ch1_x": {
                    "bpfo_amplitude": 0.005,
                    "bpfi_amplitude": 0.004,
                    "bsf_amplitude": 0.003,
                    "ftf_amplitude": 0.002,
                    "bpfo_harmonics_2x": 0.002,
                    "bpfi_harmonics_2x": 0.001,
                    "spectral_energy_total": 0.1,
                    "spectral_energy_high_freq_band": 0.01,
                    "dominant_frequency_hz": 33.3,
                    "sideband_presence": False,
                    "sideband_spacing_hz": 0.0,
                    "sideband_count": 0,
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
            "equipment_id": "LINE-A",
            "equipment_name": "ZA2115 Pillow Block Line A",
            "location": "라인 A / 제품 AX-01",
            "shaft_rpm": 2000,
            "radial_load_lbs": 6000,
            "operation_start_date": "2026-01-01",
            "bearing": {
                "bearing_id": "AX-01",
                "position": "Drive End",
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
                    "rms": 0.15,
                    "peak": 0.55,
                    "peak_to_peak": 1.10,
                    "crest_factor": 3.7,
                    "kurtosis": 4.2,
                    "skewness": 0.3,
                    "standard_deviation": 0.15,
                    "mean": 0.002,
                    "shape_factor": 1.35,
                },
            },
            "frequency_domain": {
                "ch1_x": {
                    "bpfo_amplitude": 0.008,
                    "bpfi_amplitude": 0.045,
                    "bsf_amplitude": 0.006,
                    "ftf_amplitude": 0.003,
                    "bpfo_harmonics_2x": 0.003,
                    "bpfi_harmonics_2x": 0.020,
                    "spectral_energy_total": 0.25,
                    "spectral_energy_high_freq_band": 0.05,
                    "dominant_frequency_hz": 296.9,
                    "sideband_presence": True,
                    "sideband_spacing_hz": 33.3,
                    "sideband_count": 2,
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
            "equipment_id": "LINE-A",
            "equipment_name": "ZA2115 Pillow Block Line A",
            "location": "라인 A / 제품 AX-01",
            "shaft_rpm": 2000,
            "radial_load_lbs": 6000,
            "operation_start_date": "2026-01-01",
            "bearing": {
                "bearing_id": "AX-01",
                "position": "Drive End",
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
                    "rms": 0.35,
                    "peak": 1.40,
                    "peak_to_peak": 2.80,
                    "crest_factor": 4.0,
                    "kurtosis": 6.5,
                    "skewness": 0.8,
                    "standard_deviation": 0.35,
                    "mean": 0.005,
                    "shape_factor": 1.50,
                },
            },
            "frequency_domain": {
                "ch1_x": {
                    "bpfo_amplitude": 0.012,
                    "bpfi_amplitude": 0.120,
                    "bsf_amplitude": 0.015,
                    "ftf_amplitude": 0.005,
                    "bpfo_harmonics_2x": 0.005,
                    "bpfi_harmonics_2x": 0.055,
                    "spectral_energy_total": 0.60,
                    "spectral_energy_high_freq_band": 0.18,
                    "dominant_frequency_hz": 296.9,
                    "sideband_presence": True,
                    "sideband_spacing_hz": 33.3,
                    "sideband_count": 4,
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
            "equipment_id": "LINE-C",
            "equipment_name": "ZA2115 Pillow Block Line C",
            "location": "라인 C / 제품 CX-01",
            "shaft_rpm": 2000,
            "radial_load_lbs": 6000,
            "operation_start_date": "2026-01-01",
            "bearing": {
                "bearing_id": "CX-01",
                "position": "Drive End",
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
                    "rms": 0.80,
                    "peak": 3.50,
                    "peak_to_peak": 7.00,
                    "crest_factor": 4.4,
                    "kurtosis": 12.0,
                    "skewness": 1.5,
                    "standard_deviation": 0.80,
                    "mean": 0.010,
                    "shape_factor": 1.65,
                },
            },
            "frequency_domain": {
                "ch1_x": {
                    "bpfo_amplitude": 0.250,
                    "bpfi_amplitude": 0.030,
                    "bsf_amplitude": 0.040,
                    "ftf_amplitude": 0.010,
                    "bpfo_harmonics_2x": 0.120,
                    "bpfi_harmonics_2x": 0.010,
                    "spectral_energy_total": 1.50,
                    "spectral_energy_high_freq_band": 0.55,
                    "dominant_frequency_hz": 236.4,
                    "sideband_presence": True,
                    "sideband_spacing_hz": 33.3,
                    "sideband_count": 6,
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
        "chat_open": False,
        "chat_messages": [],  # list of {"role": "user"|"assistant", "content": str}
        "chat_session_id": None,
        "deep_search_enabled": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_session_state() -> None:
    """분석 결과 초기화 (채팅 상태는 유지)."""
    st.session_state.run_id = None
    st.session_state.status = "idle"
    st.session_state.thoughts = []
    st.session_state.diagnosis = {}
    st.session_state.report = ""
    st.session_state.work_order = {}
    st.session_state.error_msg = ""
    # 채팅은 분석과 독립적이므로 세션만 초기화
    st.session_state.chat_session_id = None


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

# ──────────────────────────── 메인 영역 레이아웃 ────────────────────────────


# 채팅 패널 토글 콜백
def _toggle_chat() -> None:
    st.session_state.chat_open = not st.session_state.chat_open


def _render_chat_messages_html(*, typing: bool = False) -> str:
    """채팅 메시지를 카카오톡 스타일 HTML로 렌더링."""
    if not st.session_state.chat_messages:
        return (
            '<div class="chat-empty-state">'
            '  <div class="chat-empty-icon">P</div>'
            '  <div class="chat-empty-text">PdM Agent에게 질문해 보세요<br>'
            "  진단, 진동 분석, 정비 등을 도와드립니다</div>"
            "</div>"
        )
    html_parts: list[str] = []
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            content = (
                msg["content"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )
            html_parts.append(
                f'<div class="chat-bubble-row-user">'
                f'  <div class="chat-msg-user">{content}</div>'
                f"</div>"
            )
        else:
            content = _md_to_html(msg["content"])
            html_parts.append(
                f'<div class="chat-bubble-row-assistant">'
                f'  <div class="chat-ai-avatar"><span class="chat-ai-avatar-text">PdM</span></div>'
                f'  <div class="chat-msg-assistant chat-md">{content}</div>'
                f"</div>"
            )
    if typing:
        html_parts.append(
            '<div class="chat-bubble-row-assistant">'
            '  <div class="chat-ai-avatar"><span class="chat-ai-avatar-text">PdM</span></div>'
            '  <div class="chat-msg-assistant">'
            '    <div class="chat-typing-indicator">'
            '      <div class="chat-typing-dot"></div>'
            '      <div class="chat-typing-dot"></div>'
            '      <div class="chat-typing-dot"></div>'
            "    </div>"
            "  </div>"
            "</div>"
        )
    # 자동 스크롤 anchor
    html_parts.append('<div class="chat-scroll-anchor" id="chat-scroll-anchor"></div>')
    return "\n".join(html_parts)


# 자동 스크롤 JS
_CHAT_AUTOSCROLL_JS = """
<script>
(function() {
    const el = document.getElementById('chat-scroll-anchor');
    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'end'}); }
})();
</script>
"""


# 채팅 패널 열림 상태에 따라 레이아웃 분할
if st.session_state.chat_open:
    main_col, chat_col = st.columns([3, 2])
else:
    main_col = st.container()
    chat_col = None

with main_col:
    title_col, chat_btn_col = st.columns([6, 1])
    with title_col:
        st.markdown("## PdM Agent 진단 대시보드")
    with chat_btn_col:
        if not st.session_state.chat_open:
            st.markdown('<div class="chat-toggle-btn">', unsafe_allow_html=True)
            st.button("Chat", key="chat_fab", on_click=_toggle_chat)
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")


# ──────────────────────────── 분석 실행 ────────────────────────────


def _run_analysis() -> None:
    """에이전트 분석 실행 (SSE 스트리밍)."""
    _reset_session_state()
    st.session_state.status = "running"

    # Placeholder 생성: 상태 표시를 상단에, thought를 하단에 배치
    with main_col:
        status_ph = st.empty()
        thoughts_ph = st.empty()

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
            thoughts.append(
                {
                    "type": "thought",
                    "text": current_text,
                    "tool_calls": [],
                    "status": "done",
                }
            )
        current_text = ""
        in_reasoning = True
        _render_live()

    def _finish_current_thought() -> None:
        """현재 thought를 완료 처리."""
        nonlocal current_text, in_reasoning
        if in_reasoning and current_text:
            thoughts.append(
                {
                    "type": "thought",
                    "text": current_text,
                    "tool_calls": [],
                    "status": "done",
                }
            )
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
                else:
                    if in_reasoning:
                        _finish_current_thought()
                    # 산출물 생성 노드를 타임라인에 표시
                    if node_name in ("generate_report", "generate_work_order"):
                        thoughts.append(
                            {
                                "type": "node",
                                "name": node_name,
                                "status": "thinking",
                            }
                        )
                        _render_live()

            elif event_type == "reasoning_token":
                if not in_reasoning:
                    continue
                token = event.get("token", "")
                current_text += token
                _render_live()

            elif event_type == "tool_call":
                pending_tool = {
                    "name": event.get("tool_name", ""),
                    "arguments": event.get("arguments", {}),
                    "result": None,
                }
                # Tool 호출을 별도 타임라인 step으로 추가
                thoughts.append(
                    {
                        "type": "skill",
                        "name": pending_tool["name"],
                        "arguments": pending_tool["arguments"],
                        "result": None,
                        "status": "thinking",
                    }
                )
                _render_live()

            elif event_type == "tool_result":
                if pending_tool:
                    result = event.get("result", "")
                    # 마지막 tool step 업데이트
                    for step in reversed(thoughts):
                        if (
                            step.get("type") == "skill"
                            and step.get("name") == pending_tool["name"]
                            and step.get("status") == "thinking"
                        ):
                            step["result"] = result
                            step["status"] = "done"
                            break
                    _render_live()
                    pending_tool = None

            elif event_type == "diagnosis":
                st.session_state.diagnosis = event.get("diagnosis", {})

            elif event_type == "report_generated":
                st.session_state.report = event.get("report", "")
                for step in reversed(thoughts):
                    if (
                        step.get("type") == "node"
                        and step.get("name") == "generate_report"
                    ):
                        step["status"] = "done"
                        break
                _render_live()

            elif event_type == "work_order_generated":
                st.session_state.work_order = event.get("work_order", {})
                for step in reversed(thoughts):
                    if (
                        step.get("type") == "node"
                        and step.get("name") == "generate_work_order"
                    ):
                        step["status"] = "done"
                        break
                _render_live()

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

with main_col:
    if st.session_state.status == "completed":
        # 추론 과정 (Thought 단위)
        if st.session_state.thoughts:
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

                    pdf_bytes = generate_work_order_pdf_bytes(
                        st.session_state.work_order
                    )
                    wo_number = st.session_state.work_order.get(
                        "wo_number", "work_order"
                    )
                    _, dl_btn_col = st.columns([4, 1])
                    with dl_btn_col:
                        st.download_button(
                            label="PDF 다운로드",
                            data=pdf_bytes,
                            file_name=f"{wo_number}_작업지시서.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.warning(f"PDF 생 실패: {e}")

                render_work_order(st.session_state.work_order)

    elif st.session_state.status == "idle":
        st.info("좌측 사이드바에서 시나리오를 선택하고 '분석 시작'을 클릭하세요.")

    elif st.session_state.status == "failed":
        error_msg = st.session_state.get("error_msg", "")
        st.error(
            f"분석 실패: {error_msg}"
            if error_msg
            else "분석 실패. API 서버 연결을 확인하세요."
        )
        # 실패 시에도 추론 과정이 있으면 표시
        if st.session_state.thoughts:
            st.markdown("### 에이전트 추론 과정")
            render_thoughts(st.session_state.thoughts)


# ──────────────────────────── 채팅 패널 ────────────────────────────


def _handle_chat_submit() -> None:
    """채팅 메시지 제출 콜백 (session_state에서 입력값 읽기)."""
    msg = st.session_state.get("_chat_text_input", "").strip()
    if not msg:
        return
    st.session_state._pending_chat_msg = msg
    st.session_state._chat_text_input = ""


if st.session_state.chat_open and chat_col is not None:
    with chat_col:
        # ── 헤더: 제목 + 닫기 ──
        hdr_col, ds_col, close_col = st.columns([2, 3, 1], vertical_alignment="center")
        with hdr_col:
            st.markdown(
                '<span class="chat-header-title">PdM Agent</span>',
                unsafe_allow_html=True,
            )
        with ds_col:
            st.session_state.deep_search_enabled = st.toggle(
                "Deep Search",
                value=st.session_state.get("deep_search_enabled", False),
                key="ds_toggle",
            )
        with close_col:
            st.button("X", key="chat_close", on_click=_toggle_chat)

        st.markdown(
            '<div class="chat-header-divider"></div>',
            unsafe_allow_html=True,
        )

        # ── 메시지 영역 ──
        chat_messages_ph = st.empty()

        def _render_full_panel(messages_html: str) -> None:
            """채팅 메시지 영역을 렌더링."""
            chat_messages_ph.markdown(
                f'<div class="chat-messages">{messages_html}</div>'
                f"{_CHAT_AUTOSCROLL_JS}",
                unsafe_allow_html=True,
            )

        _render_full_panel(_render_chat_messages_html())

        # ── 입력 영역 ──
        st.text_input(
            "질문 입력",
            key="_chat_text_input",
            placeholder="메시지를 입력하세요...",
            on_change=_handle_chat_submit,
            label_visibility="collapsed",
        )

        # ── 보류 중인 메시지 처리 ──
        pending_msg = st.session_state.pop("_pending_chat_msg", None)
        if pending_msg:
            ds_thinking_html = ""
            ds_pipeline_stage = ""
            ds_current_researcher = -1
            # Deep Search 상태 추적 (재렌더링용)
            ds_state = {
                "stage": "",
                "perspectives": "",
                "total_perspectives": 0,
                "researchers": [],  # [{"label": str, "confidence": float, "preview": str}]
                "critic_summary": "",
                "critic_items": [],  # [{"text": str, "passed": bool}]
                "voting_weights": [],
                "done": False,
            }
            st.session_state.chat_messages.append(
                {"role": "user", "content": pending_msg}
            )
            # 타이핑 인디케이터 표시
            _render_full_panel(_render_chat_messages_html(typing=True))

            try:
                _deep_search = st.session_state.get("deep_search_enabled", False)
                try:
                    chat_resp = submit_chat(
                        api_url,
                        st.session_state.run_id,
                        pending_msg,
                        st.session_state.chat_session_id,
                        deep_search=_deep_search,
                    )
                except Exception:
                    # run_id가 유효하지 않으면 run_id 없이 재시도
                    st.session_state.run_id = None
                    st.session_state.chat_session_id = None
                    chat_resp = submit_chat(
                        api_url,
                        None,
                        pending_msg,
                        None,
                        deep_search=_deep_search,
                    )
                session_id = chat_resp["session_id"]
                st.session_state.chat_session_id = session_id

                assistant_text = ""
                for evt in stream_chat(api_url, session_id):
                    evt_type = evt.get("event", "")
                    if evt_type == "chat_token":
                        assistant_text += evt.get("token", "")
                        # 스트리밍 중: 기존 메시지 + 현재 스트리밍 텍스트
                        streaming_html = _render_chat_messages_html()
                        streaming_html += (
                            f'\n<div class="chat-bubble-row-assistant">'
                            f'  <div class="chat-ai-avatar"><span class="chat-ai-avatar-text">PdM</span></div>'
                            f'  <div class="chat-msg-assistant">'
                            f"    {assistant_text}"
                            f'    <span class="chat-streaming-dot"></span>'
                            f"  </div>"
                            f"</div>"
                            f'<div class="chat-scroll-anchor" id="chat-scroll-anchor"></div>'
                        )
                        _render_full_panel(streaming_html)
                    elif evt_type == "deep_search_step":
                        step_type = evt.get("step_type", "")
                        role = evt.get("role", "")
                        content = evt.get("content", "")
                        confidence = evt.get("confidence")
                        voting_weights = evt.get("voting_weights")

                        # -- 상태 업데이트 (append 대신 갱신) --
                        if step_type == "started":
                            ds_state["stage"] = "started"
                        elif step_type == "decompose":
                            ds_state["stage"] = "decompose"
                            total = evt.get("total_perspectives", 0)
                            if total:
                                ds_state["total_perspectives"] = total
                            if content.strip():
                                ds_state["perspectives"] = content
                        elif step_type == "researcher":
                            ds_state["stage"] = "research"
                            label = role.split(": ", 1)[1] if ": " in role else role
                            conf = confidence if confidence is not None else 0.0
                            preview = ""
                            if content.strip():
                                preview = (
                                    content[:100] + "..."
                                    if len(content) > 100
                                    else content
                                )
                            idx = evt.get("perspective_index", 0)
                            # 같은 researcher면 업데이트, 아니면 추가
                            found = False
                            for r in ds_state["researchers"]:
                                if r["label"] == label:
                                    r["confidence"] = conf
                                    if preview:
                                        r["preview"] = preview
                                    found = True
                                    break
                            if not found:
                                ds_state["researchers"].append(
                                    {
                                        "label": label,
                                        "confidence": conf,
                                        "preview": preview,
                                    }
                                )
                        elif step_type == "critic":
                            ds_state["stage"] = "critic"
                            lines = content.split("\n") if content else []
                            if lines:
                                ds_state["critic_summary"] = lines[0]
                            items = []
                            for line in lines[1:]:
                                if line.strip():
                                    passed = "[Pass]" in line
                                    items.append(
                                        {"text": line.strip(), "passed": passed}
                                    )
                            if items:
                                ds_state["critic_items"] = items
                        elif step_type == "voting":
                            ds_state["stage"] = "voting"
                            if voting_weights:
                                ds_state["voting_weights"] = voting_weights
                        elif step_type == "synthesis":
                            ds_state["stage"] = "synthesis"
                            ds_state["done"] = True

                        # -- 상태에서 HTML 재구성 --
                        ds_html = '<div class="ds-pipeline">'
                        ds_html += (
                            '<div class="ds-pipeline-title">' "Deep Research</div>"
                        )

                        # 1. Decompose
                        stage = ds_state["stage"]
                        active_1 = stage == "decompose" or stage == "started"
                        done_1 = stage not in ("started", "decompose")
                        num_cls_1 = (
                            " ds-num-done"
                            if done_1
                            else (" ds-num-active" if active_1 else "")
                        )
                        ds_html += '<div class="ds-stage">'
                        ds_html += (
                            f'<div class="ds-stage-header">'
                            f'<span class="ds-stage-num{num_cls_1}">1</span>'
                            f" Decompose"
                        )
                        if active_1 and not ds_state["perspectives"]:
                            ds_html += ' <span class="ds-stage-spinner">...</span>'
                        ds_html += "</div>"
                        if ds_state["perspectives"]:
                            ds_html += (
                                f'<div class="ds-stage-detail">'
                                f'{ds_state["total_perspectives"]}개 전문가 관점: '
                                f'{ds_state["perspectives"]}</div>'
                            )
                        ds_html += "</div>"

                        # 2. Research
                        if (
                            stage
                            in (
                                "research",
                                "critic",
                                "voting",
                                "synthesis",
                            )
                            or ds_state["researchers"]
                        ):
                            active_2 = stage == "research"
                            done_2 = stage in ("critic", "voting", "synthesis")
                            num_cls_2 = (
                                " ds-num-done"
                                if done_2
                                else (" ds-num-active" if active_2 else "")
                            )
                            ds_html += '<div class="ds-stage">'
                            ds_html += (
                                f'<div class="ds-stage-header">'
                                f'<span class="ds-stage-num{num_cls_2}">2</span>'
                                f" Research</div>"
                            )
                            total_r = ds_state["total_perspectives"] or len(
                                ds_state["researchers"]
                            )
                            for i, r in enumerate(ds_state["researchers"]):
                                conf = r["confidence"]
                                bar_filled = int(conf * 5)
                                # 신뢰도 수준별 색상 클래스
                                if conf >= 0.7:
                                    conf_cls = "ds-conf-high"
                                elif conf >= 0.4:
                                    conf_cls = "ds-conf-mid"
                                else:
                                    conf_cls = "ds-conf-low"
                                bar_html = (
                                    f'<span class="ds-conf-bar {conf_cls}">'
                                    f'{"●" * bar_filled}'
                                    f'{"○" * (5 - bar_filled)}</span>'
                                )
                                ds_html += '<div class="ds-researcher-item ds-done">'
                                ds_html += '<div class="ds-researcher-header">'
                                ds_html += (
                                    f'<span class="ds-researcher-label">'
                                    f"{r['label']}</span>"
                                )
                                ds_html += (
                                    f'<span class="ds-conf {conf_cls}">'
                                    f"{bar_html} conf {conf:.2f}</span>"
                                )
                                ds_html += "</div>"
                                if r["preview"]:
                                    ds_html += (
                                        f'<div class="ds-researcher-preview">'
                                        f'{r["preview"]}</div>'
                                    )
                                ds_html += "</div>"
                            # 아직 도착하지 않은 researcher 표시
                            remaining = total_r - len(ds_state["researchers"])
                            if active_2 and remaining > 0:
                                ds_html += (
                                    '<div class="ds-researcher-item ds-pending">'
                                    '<div class="ds-researcher-header">'
                                    '<span class="ds-researcher-label">'
                                    "조사 중...</span>"
                                    '<span class="ds-conf ds-conf-low">'
                                    '<span class="ds-conf-bar ds-conf-low">'
                                    "○○○○○</span> conf --</span>"
                                    "</div></div>"
                                )
                            ds_html += "</div>"

                        # 3. Review
                        if stage in ("critic", "voting", "synthesis"):
                            done_3 = stage in ("voting", "synthesis")
                            num_cls_3 = " ds-num-done" if done_3 else (" ds-num-active")
                            ds_html += '<div class="ds-stage">'
                            ds_html += (
                                f'<div class="ds-stage-header">'
                                f'<span class="ds-stage-num{num_cls_3}">3</span>'
                                f" Review</div>"
                            )
                            if ds_state["critic_summary"]:
                                ds_html += (
                                    f'<div class="ds-stage-detail">'
                                    f'{ds_state["critic_summary"]}</div>'
                                )
                            for ci in ds_state["critic_items"]:
                                cls = "ds-pass" if ci["passed"] else "ds-revise"
                                ds_html += (
                                    f'<div class="ds-critic-item {cls}">'
                                    f'{ci["text"]}</div>'
                                )
                            ds_html += "</div>"

                        # 4. Synthesize + Voting
                        if stage in ("voting", "synthesis"):
                            num_cls_4 = (
                                " ds-num-done" if ds_state["done"] else " ds-num-active"
                            )
                            ds_html += '<div class="ds-stage">'
                            ds_html += (
                                f'<div class="ds-stage-header">'
                                f'<span class="ds-stage-num{num_cls_4}">4</span>'
                                f" Synthesize</div>"
                            )
                            if ds_state["voting_weights"]:
                                ds_html += '<div class="ds-voting">'
                                ds_html += (
                                    '<div class="ds-voting-title">'
                                    "Weighted Voting</div>"
                                )
                                for vw in ds_state["voting_weights"]:
                                    name = vw.get("perspective", "")
                                    weight = vw.get("weight", 0)
                                    bar_w = int(weight)
                                    p_cls = (
                                        "ds-vote-pass"
                                        if vw.get("passed", True)
                                        else "ds-vote-revise"
                                    )
                                    ds_html += '<div class="ds-vote-row">'
                                    ds_html += (
                                        f'<div class="ds-vote-label">' f"{name}</div>"
                                    )
                                    ds_html += (
                                        f'<div class="ds-vote-bar-bg">'
                                        f'<div class="ds-vote-bar-fill {p_cls}"'
                                        f' style="width:{bar_w}%"></div></div>'
                                    )
                                    ds_html += (
                                        f'<div class="ds-vote-pct">' f"{weight}%</div>"
                                    )
                                    ds_html += "</div>"
                                ds_html += "</div>"
                            ds_html += "</div>"

                        ds_html += "</div>"

                        # -- 렌더링 --
                        streaming_html = _render_chat_messages_html()
                        streaming_html += (
                            f'\n<div class="chat-bubble-row-assistant">'
                            f'  <div class="chat-ai-avatar">'
                            f'<span class="chat-ai-avatar-text">PdM</span>'
                            f"</div>"
                            f'  <div class="chat-msg-assistant">'
                            f"{ds_html}</div>"
                            f"</div>"
                            f'<div class="chat-scroll-anchor"'
                            f' id="chat-scroll-anchor"></div>'
                        )
                        _render_full_panel(streaming_html)
                    elif evt_type == "chat_completed":
                        assistant_text = evt.get("content", assistant_text)
                    elif evt_type == "chat_error":
                        assistant_text = (
                            f"오류: {evt.get('message', '알 수 없는 오류')}"
                        )

                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                _render_full_panel(_render_chat_messages_html())
            except Exception as e:
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": f"오류: {e}"}
                )
                st.rerun()

else:
    pass  # Chat 버튼은 제목 우측에 표시됨
