"""재사용 UI 컴포넌트."""

from __future__ import annotations

import html
import json

import streamlit as st

from styles import DIAG_CARD_CSS, REPORT_CSS, RISK_COLORS, THOUGHT_CSS


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


def _md_to_html_simple(text: str) -> str:
    """마크다운 텍스트를 간단한 HTML로 변환 (p 태그 기반)."""
    if not text:
        return ""
    escaped = html.escape(text)
    paragraphs = escaped.split("\n\n")
    parts = []
    for para in paragraphs:
        # 줄바꿈을 <br>로
        para = para.strip()
        if not para:
            continue
        # **bold** 처리
        import re

        para = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", para)
        # 단일 줄바꿈 → <br>
        para = para.replace("\n", "<br>")
        parts.append(f"<p>{para}</p>")
    return "".join(parts)


def _render_tool_html(tc: dict) -> str:
    """Tool 호출을 HTML로 렌더링."""
    name = html.escape(tc.get("name", ""))
    parts = [f'<span class="tool-badge"><span class="tool-icon">&gt;</span>{name}</span>']

    args = tc.get("arguments")
    result = tc.get("result")
    if args or result:
        detail_parts = []
        if args:
            args_str = html.escape(json.dumps(args, ensure_ascii=False, indent=2))
            detail_parts.append(
                f"<details><summary>Arguments</summary><pre>{args_str}</pre></details>"
            )
        if result:
            if isinstance(result, dict):
                result_str = html.escape(
                    json.dumps(result, ensure_ascii=False, indent=2)
                )
            else:
                result_str = html.escape(str(result)[:2000])
            detail_parts.append(
                f"<details><summary>Result</summary><pre>{result_str}</pre></details>"
            )
        parts.append(f'<div class="tool-detail">{"".join(detail_parts)}</div>')

    return "".join(parts)


def _build_step_html(
    label: str,
    summary: str,
    body_html: str,
    tools_html: str,
    dot_class: str,
    is_open: bool,
) -> str:
    """단일 step의 HTML 생성. dot은 details 밖에 배치."""
    open_attr = " open" if is_open else ""
    return (
        f'<div class="thought-step">'
        f'  <div class="thought-dot {dot_class}"></div>'
        f'  <details class="thought-accordion"{open_attr}>'
        f"    <summary>{label} &mdash; {html.escape(summary)}</summary>"
        f'    <div class="thought-body">{body_html}{tools_html}</div>'
        f"  </details>"
        f"</div>"
    )


def _build_tool_body_html(step: dict) -> str:
    """Tool step의 body HTML 생성 (arguments + result)."""
    parts = []
    args = step.get("arguments")
    result = step.get("result")
    if args:
        args_str = html.escape(json.dumps(args, ensure_ascii=False, indent=2))
        parts.append(
            f"<details open><summary>Arguments</summary><pre>{args_str}</pre></details>"
        )
    if result is not None:
        if isinstance(result, dict):
            result_str = html.escape(
                json.dumps(result, ensure_ascii=False, indent=2)
            )
        else:
            result_str = html.escape(str(result)[:2000])
        parts.append(
            f"<details><summary>Result</summary><pre>{result_str}</pre></details>"
        )
    elif step.get("status") == "thinking":
        parts.append('<p style="color:rgba(128,128,128,0.6);">실행 중...</p>')
    return "".join(parts)


_NODE_LABELS: dict[str, str] = {
    "generate_report": "리포트 생성",
    "generate_work_order": "작업지시서 생성",
}


def _render_single_step(step: dict, thought_num: list[int]) -> str:
    """단일 step(thought, mcp, node)의 HTML을 반환. thought_num은 mutable counter."""
    step_type = step.get("type", "thought")
    status = step.get("status", "done")

    if step_type == "mcp":
        name = step.get("name", "unknown")
        dot_class = "thought-dot-tool" if status == "done" else "thought-dot-active"
        body_html = _build_tool_body_html(step)
        return _build_step_html(
            "MCP", name, body_html, "", dot_class, is_open=(status == "thinking")
        )
    elif step_type == "node":
        name = step.get("name", "")
        label = _NODE_LABELS.get(name, name)
        dot_class = "thought-dot-node" if status == "done" else "thought-dot-active"
        status_text = "완료" if status == "done" else "생성 중..."
        body_html = f'<p style="color:rgba(128,128,128,0.7);">{status_text}</p>'
        return _build_step_html(
            label, "", body_html, "", dot_class, is_open=(status == "thinking")
        )
    else:
        thought_num[0] += 1
        text = step.get("text", "")
        tool_calls = step.get("tool_calls", [])
        summary = _extract_summary(text)
        dot_class = "thought-dot-active" if status == "thinking" else "thought-dot-done"
        body_html = _md_to_html_simple(text)

        tools_html = ""
        if tool_calls:
            tool_items = "".join(_render_tool_html(tc) for tc in tool_calls)
            tools_html = f'<div style="margin-top:8px;">{tool_items}</div>'

        return _build_step_html(
            f"Step {thought_num[0]}", summary, body_html, tools_html,
            dot_class, is_open=(status == "thinking"),
        )


def render_thoughts(thoughts: list[dict]) -> None:
    """Thought/Tool 단계별 추론 과정을 타임라인 형태로 표시.

    각 step: {type: "thought"|"tool", ...}
    - thought: {text, tool_calls, status}
    - tool: {name, arguments, result, status}

    Note: 각 step을 개별 st.markdown()으로 분리하여
    st.rerun() 후에도 모든 <details> 태그가 정상 렌더링되도록 한다.
    """
    st.markdown(THOUGHT_CSS, unsafe_allow_html=True)

    thought_num = [0]
    for step in thoughts:
        step_html = (
            f'<div class="thought-timeline">'
            + _render_single_step(step, thought_num)
            + "</div>"
        )
        st.markdown(step_html, unsafe_allow_html=True)


def render_thoughts_live(
    completed: list[dict],
    current_text: str,
    placeholder,
) -> None:
    """완료된 steps(접힌 상태) + 현재 streaming thought(펼친 상태)를 하나의 placeholder에 누적 렌더링."""
    steps_html = []
    thought_num = [0]

    # 완료된 steps (접힌 상태)
    for step in completed:
        steps_html.append(_render_single_step(step, thought_num))

    # 현재 streaming thought (펼친 상태)
    if current_text:
        thought_num[0] += 1
        summary = _extract_summary(current_text)
        body_html = _md_to_html_simple(current_text)
        steps_html.append(
            _build_step_html(
                f"Step {thought_num[0]}", summary, body_html, "",
                "thought-dot-active", is_open=True,
            )
        )

    timeline_html = f'<div class="thought-timeline">{"".join(steps_html)}</div>'
    with placeholder.container():
        st.markdown(THOUGHT_CSS, unsafe_allow_html=True)
        st.markdown(timeline_html, unsafe_allow_html=True)


def _format_rul(rul) -> str:
    """rul_assessment 값을 사람이 읽기 좋은 문자열로 변환."""
    if isinstance(rul, dict):
        parts = []
        ml_hours = rul.get("ml_rul_hours")
        if ml_hours is not None:
            parts.append(f"{ml_hours}h")
        assessment = rul.get("agent_assessment")
        if assessment:
            parts.append(html.escape(str(assessment)))
        confidence = rul.get("confidence_level")
        if confidence:
            parts.append(f"(신뢰도: {html.escape(str(confidence))})")
        return " / ".join(parts) if parts else "-"
    return html.escape(str(rul)) if rul else "-"


def render_diagnosis_cards(diagnosis: dict) -> None:
    """진단 결과를 단일 박스에 key-value 리스트로 렌더링."""
    if not diagnosis:
        return

    st.markdown(DIAG_CARD_CSS, unsafe_allow_html=True)

    risk_level = diagnosis.get("risk_level") or diagnosis.get("severity", "normal")
    risk_color = RISK_COLORS.get(risk_level, "#6c757d")

    fault_type = html.escape(str(diagnosis.get("fault_type", "-")))
    fault_stage = html.escape(str(diagnosis.get("fault_stage", "-")))
    rul = _format_rul(diagnosis.get("rul_assessment", "-"))
    rec = html.escape(str(diagnosis.get("recommendation", "-")))

    risk_text_color = "#212529" if risk_level in ("watch", "warning") else "white"
    risk_html = f'<span class="risk-badge" style="background:{risk_color};color:{risk_text_color};">{risk_level.upper()}</span>'

    rows = [
        ("결함 유형", fault_type),
        ("결함 단계", fault_stage),
        ("위험도", risk_html),
        ("잔여수명 (RUL)", rul),
        ("정비 권고", rec),
    ]

    rows_html = "".join(
        f'<div class="diag-row">'
        f'<span class="diag-label">{label}</span>'
        f'<span class="diag-value">{value}</span>'
        f"</div>"
        for label, value in rows
    )

    st.markdown(f'<div class="diag-box">{rows_html}</div>', unsafe_allow_html=True)


def render_report(report: str) -> None:
    """분석 리포트를 문서 스타일 컨테이너로 렌더링."""
    if not report:
        st.info("리포트가 생성되지 않았습니다.")
        return

    st.markdown(REPORT_CSS, unsafe_allow_html=True)

    import re

    report_html = html.escape(report)

    # markdown → HTML 변환 (순서 중요)
    # 헤더 (#### → h4, ### → h3, ## → h2, # → h1)
    report_html = re.sub(
        r"^####\s+(.+)$", r"<h4>\1</h4>", report_html, flags=re.MULTILINE
    )
    report_html = re.sub(
        r"^###\s+(.+)$", r"<h3>\1</h3>", report_html, flags=re.MULTILINE
    )
    report_html = re.sub(
        r"^##\s+(.+)$", r"<h2>\1</h2>", report_html, flags=re.MULTILINE
    )
    report_html = re.sub(
        r"^#\s+(.+)$", r"<h1>\1</h1>", report_html, flags=re.MULTILINE
    )

    # bold / italic
    report_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", report_html)
    report_html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", report_html)

    # 리스트 아이템 (- 또는 숫자.)
    report_html = re.sub(
        r"^[-•]\s+(.+)$", r"<li>\1</li>", report_html, flags=re.MULTILINE
    )
    report_html = re.sub(
        r"^(\d+)\.\s+(.+)$", r"<li>\2</li>", report_html, flags=re.MULTILINE
    )
    # 연속 <li>를 <ul>로 감싸기
    report_html = re.sub(
        r"((?:<li>.*?</li>\n?)+)",
        r"<ul>\1</ul>",
        report_html,
    )

    # 수평선
    report_html = re.sub(r"^-{3,}$", r"<hr>", report_html, flags=re.MULTILINE)

    # 빈 줄로 구분된 단락 → <p>
    paragraphs = report_html.split("\n\n")
    processed = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 이미 HTML 태그로 시작하면 그대로
        if re.match(r"<(h[1-4]|ul|li|hr|p)", para):
            processed.append(para)
        else:
            # 단일 줄바꿈 → <br>
            para = para.replace("\n", "<br>")
            processed.append(f"<p>{para}</p>")
    report_html = "\n".join(processed)

    st.markdown(
        f'<div class="report-container">{report_html}</div>',
        unsafe_allow_html=True,
    )


def render_equipment_info(payload: dict) -> None:
    """사이드바 설비/베어링/이상감지 요약 (compact HTML table)."""
    meta = payload.get("equipment_meta", {})
    anomaly = payload.get("anomaly_detection_result", {})
    bearing = meta.get("bearing", {})
    health_state = anomaly.get("health_state", "-")
    risk_color = RISK_COLORS.get(health_state, "#6c757d")

    sidebar_html = f"""
<div class="sidebar-section">
  <div class="sidebar-heading">설비 정보</div>
  <table class="sidebar-table">
    <tr><td class="label">설비</td><td colspan="3">{meta.get('equipment_id', '-')}<br><span style="color:rgba(128,128,128,0.9);font-size:12px;">{meta.get('equipment_name', '-')}</span></td></tr>
    <tr><td class="label">베어링</td><td>{bearing.get('bearing_id', '-')}</td>
        <td class="label">위치</td><td>{bearing.get('position', '-')}</td></tr>
    <tr><td class="label">RPM</td><td>{meta.get('shaft_rpm', '-')}</td>
        <td class="label">가동</td><td>{meta.get('total_running_hours', '-')}h</td></tr>
  </table>
  <div class="sidebar-heading" style="margin-top:10px;">이상 감지</div>
  <table class="sidebar-table">
    <tr><td class="label">상태</td>
        <td><span style="color:{risk_color};font-weight:700;">{health_state.upper()}</span></td>
        <td class="label">점수</td><td>{anomaly.get('anomaly_score', '-')}</td></tr>
    <tr><td class="label">임계값</td><td>{anomaly.get('anomaly_threshold', '-')}</td>
        <td class="label">신뢰도</td><td>{anomaly.get('confidence', '-')}</td></tr>
  </table>
</div>
"""
    st.markdown(sidebar_html, unsafe_allow_html=True)


def _esc(value) -> str:
    """값을 HTML escape된 문자열로 변환."""
    return html.escape(str(value)) if value else "-"


def _build_info_rows(work_order: dict, fields: list[tuple[str, str]]) -> str:
    """key-value 필드를 diag-row 스타일 HTML로 변환."""
    return "".join(
        f'<div class="diag-row">'
        f'<span class="diag-label">{label}</span>'
        f'<span class="diag-value">{_esc(work_order.get(key))}</span>'
        f"</div>"
        for label, key in fields
    )


def _build_table_html(items: list[dict], column_map: dict[str, str]) -> str:
    """dict 리스트를 HTML table로 변환."""
    if not items:
        return ""
    cols = [k for k in column_map if k in items[0]]
    header = "".join(f"<th>{column_map[k]}</th>" for k in cols)
    rows = ""
    for item in items:
        cells = "".join(f"<td>{_esc(item.get(k))}</td>" for k in cols)
        rows += f"<tr>{cells}</tr>"
    return (
        '<table class="wo-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def render_work_order(work_order: dict) -> None:
    """구조화된 작업지시서 dict를 문서 스타일 HTML로 렌더링."""
    if not work_order:
        st.info("작업지시서가 생성되지 않았습니다.")
        return

    st.markdown(REPORT_CSS, unsafe_allow_html=True)

    parts: list[str] = []

    # 기본 정보
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
    parts.append(f"<h2>기본 정보</h2><div class='diag-box'>{_build_info_rows(work_order, info_fields)}</div>")

    # 작업 내용
    work_lines = []
    if work_order.get("summary"):
        work_lines.append(f"<p><strong>작업 내용 요약</strong><br>{_esc(work_order['summary'])}</p>")
    if work_order.get("safety"):
        work_lines.append(f"<p><strong>안전사항</strong><br>{_esc(work_order['safety'])}</p>")
    if work_lines:
        parts.append(f"<h2>작업 내용</h2>{''.join(work_lines)}")

    # 체크리스트
    if work_order.get("checklist"):
        items = "".join(f"<li>{_esc(item)}</li>" for item in work_order["checklist"])
        parts.append(f"<h2>작업 상세 체크리스트</h2><ol>{items}</ol>")

    # 필요 자재
    material_cols = {"code": "코드", "name": "자재명", "spec": "규격", "qty": "수량", "unit": "단위", "note": "비고"}
    if work_order.get("materials"):
        parts.append(f"<h2>필요 자재</h2>{_build_table_html(work_order['materials'], material_cols)}")

    # 필요 공구
    tool_cols = {"code": "코드", "name": "명칭", "spec": "규격", "qty": "수량", "unit": "단위", "note": "비고"}
    if work_order.get("tools"):
        parts.append(f"<h2>필요 공구 및 장비</h2>{_build_table_html(work_order['tools'], tool_cols)}")

    # 작업 후 확인사항
    if work_order.get("post_checks"):
        items = "".join(f"<li>{_esc(item)}</li>" for item in work_order["post_checks"])
        parts.append(f"<h2>작업 후 확인사항</h2><ul>{items}</ul>")

    # 승인 및 결과
    approval_fields = [
        ("작업 승인자", "approver"),
        ("작업 완료일시", "completion_date"),
        ("작업 결과 요약", "result_summary"),
    ]
    if any(work_order.get(k) for _, k in approval_fields):
        parts.append(f"<h2>승인 및 결과</h2><div class='diag-box'>{_build_info_rows(work_order, approval_fields)}</div>")

    st.markdown(
        f'<div class="report-container">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )
