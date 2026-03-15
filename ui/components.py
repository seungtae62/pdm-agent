"""재사용 UI 컴포넌트."""

from __future__ import annotations

import json

import streamlit as st

from styles import RISK_COLORS, THOUGHT_CSS


def _extract_summary(text: str, max_len: int = 80) -> str:
    """추론 텍스트에서 요약 한 줄 추출."""
    if not text:
        return "추론 중..."
    # 첫 번째 의미 있는 줄
    for line in text.strip().splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            if len(line) > max_len:
                return line[:max_len] + "..."
            return line
    return text[:max_len] + "..." if len(text) > max_len else text


def render_thoughts(thoughts: list[dict], is_streaming: bool = False) -> None:
    """Thought 단계별 추론 과정 표시.

    각 thought: {text, tool_calls, status}
    status: "thinking" / "done"
    """
    st.markdown(THOUGHT_CSS, unsafe_allow_html=True)

    for i, thought in enumerate(thoughts):
        num = i + 1
        status = thought.get("status", "done")
        text = thought.get("text", "")
        tool_calls = thought.get("tool_calls", [])
        summary = _extract_summary(text)

        # 상태 아이콘
        if status == "thinking":
            icon = "\u23f3"  # hourglass
            label_class = "thought-active"
        else:
            icon = "\u2705"  # checkmark
            label_class = "thought-done"

        # Thought 헤더 + expander
        with st.expander(
            f"{icon}  Thought {num}: {summary}",
            expanded=(status == "thinking"),
        ):
            if text:
                st.markdown(text)

            # 해당 thought에서 발생한 Tool 호출
            for tc in tool_calls:
                render_tool_call(
                    tc["name"],
                    tc.get("arguments"),
                    tc.get("result"),
                )


def render_thought_streaming(
    thought_num: int,
    text: str,
    placeholder,
) -> None:
    """스트리밍 중인 현재 Thought를 placeholder에 실시간 갱신."""
    summary = _extract_summary(text)
    with placeholder.container():
        st.markdown(THOUGHT_CSS, unsafe_allow_html=True)
        with st.expander(
            f"\u23f3  Thought {thought_num}: {summary}",
            expanded=True,
        ):
            st.markdown(text)


def render_diagnosis_cards(diagnosis: dict) -> None:
    """결함유형, 단계, 위험도, RUL, 정비 권고 카드 렌더링."""
    if not diagnosis:
        return

    risk_level = diagnosis.get("risk_level", "normal")
    color = RISK_COLORS.get(risk_level, "#6c757d")

    col1, col2, col3 = st.columns(3)

    with col1:
        _render_card(
            "결함 유형",
            diagnosis.get("fault_type", "-"),
            "#0d6efd",
        )
    with col2:
        _render_card(
            "결함 단계",
            diagnosis.get("fault_stage", "-"),
            "#6610f2",
        )
    with col3:
        _render_card(
            "위험도",
            risk_level.upper(),
            color,
        )

    col4, col5 = st.columns(2)

    with col4:
        rul = diagnosis.get("rul_assessment", "-")
        _render_card("잔여수명 (RUL)", str(rul), "#0dcaf0")
    with col5:
        rec = diagnosis.get("recommendation", "-")
        _render_card("정비 권고", str(rec), "#198754")


def _render_card(label: str, value: str, bg_color: str) -> None:
    """단일 진단 카드."""
    st.markdown(
        f'<div class="diag-card" style="background-color: {bg_color};">'
        f'<div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_equipment_info(payload: dict) -> None:
    """사이드바 설비/베어링/이상감지 요약."""
    meta = payload.get("equipment_meta", {})
    anomaly = payload.get("anomaly_detection_result", {})
    bearing = meta.get("bearing", {})

    st.markdown("#### 설비 정보")
    st.markdown(f"**설비**: {meta.get('equipment_id', '-')}")
    st.markdown(f"**설비명**: {meta.get('equipment_name', '-')}")
    st.markdown(f"**베어링**: {bearing.get('bearing_id', '-')}")
    st.markdown(f"**위치**: {bearing.get('position', '-')}")
    st.markdown(f"**RPM**: {meta.get('shaft_rpm', '-')}")
    st.markdown(f"**가동 시간**: {meta.get('total_running_hours', '-')}h")

    st.markdown("---")
    st.markdown("#### 이상 감지")

    health_state = anomaly.get("health_state", "-")
    risk_color = RISK_COLORS.get(health_state, "#6c757d")
    st.markdown(
        f"**상태**: <span style='color:{risk_color}; font-weight:bold;'>"
        f"{health_state.upper()}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**이상 점수**: {anomaly.get('anomaly_score', '-')}")
    st.markdown(f"**임계값**: {anomaly.get('anomaly_threshold', '-')}")
    st.markdown(f"**신뢰도**: {anomaly.get('confidence', '-')}")


def render_work_order(work_order: dict) -> None:
    """구조화된 작업지시서 dict를 Streamlit 컴포넌트로 렌더링."""
    if not work_order:
        st.info("작업지시서가 생성되지 않았습니다.")
        return

    # 기본 정보
    st.markdown("#### 기본 정보")
    info_fields = [
        ("작업지시 번호", "wo_number"),
        ("설비명 / 설비번호", "equipment"),
        ("설비 위치", "location"),
        ("작업 유형", "work_type"),
        ("작업 요청일", "request_date"),
        ("작업 예정일", "scheduled_date"),
        ("완료 예정일", "due_date"),
        ("작업 담당자", "assignee"),
    ]
    for label, key in info_fields:
        value = work_order.get(key, "-") or "-"
        st.markdown(f"**{label}**: {value}")

    st.markdown("---")

    # 작업 내용
    st.markdown("#### 작업 내용")
    if work_order.get("summary"):
        st.markdown(f"**작업 내용 요약**: {work_order['summary']}")
    if work_order.get("safety"):
        st.markdown(f"**안전사항**: {work_order['safety']}")

    # 체크리스트
    if work_order.get("checklist"):
        st.markdown("---")
        st.markdown("#### 작업 상세 체크리스트")
        for i, item in enumerate(work_order["checklist"], 1):
            st.markdown(f"{i}. {item}")

    # 필요 자재
    if work_order.get("materials"):
        st.markdown("---")
        st.markdown("#### 필요 자재")
        import pandas as pd

        df = pd.DataFrame(work_order["materials"])
        column_map = {
            "code": "코드",
            "name": "자재명",
            "spec": "규격",
            "qty": "수량",
            "unit": "단위",
            "note": "비고",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        st.dataframe(df, use_container_width=True, hide_index=True)

    # 필요 공구
    if work_order.get("tools"):
        st.markdown("---")
        st.markdown("#### 필요 공구 및 장비")
        import pandas as pd

        df = pd.DataFrame(work_order["tools"])
        column_map = {
            "code": "코드",
            "name": "명칭",
            "spec": "규격",
            "qty": "수량",
            "unit": "단위",
            "note": "비고",
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        st.dataframe(df, use_container_width=True, hide_index=True)

    # 작업 후 확인사항
    if work_order.get("post_checks"):
        st.markdown("---")
        st.markdown("#### 작업 후 확인사항")
        for item in work_order["post_checks"]:
            st.markdown(f"- {item}")

    # 승인 및 결과
    approval_fields = [
        ("작업 승인자", "approver"),
        ("작업 완료일시", "completion_date"),
        ("작업 결과 요약", "result_summary"),
    ]
    has_approval = any(work_order.get(k) for _, k in approval_fields)
    if has_approval:
        st.markdown("---")
        st.markdown("#### 승인 및 결과")
        for label, key in approval_fields:
            value = work_order.get(key, "-") or "-"
            st.markdown(f"**{label}**: {value}")


def render_tool_call(
    tool_name: str,
    arguments: dict | None = None,
    result: str | None = None,
) -> None:
    """Tool 호출 expander."""
    with st.expander(f"\U0001f527 Tool: {tool_name}", expanded=False):
        if arguments:
            st.markdown("**Arguments:**")
            st.code(json.dumps(arguments, ensure_ascii=False, indent=2), language="json")
        if result:
            st.markdown("**Result:**")
            if isinstance(result, dict):
                st.code(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    language="json",
                )
            else:
                st.text(str(result)[:2000])
