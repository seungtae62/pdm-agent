"""CSS 스타일 상수."""

from __future__ import annotations

# 위험도별 색상
RISK_COLORS: dict[str, str] = {
    "normal": "#28a745",
    "watch": "#ffc107",
    "warning": "#fd7e14",
    "critical": "#dc3545",
}

GLOBAL_CSS = """
<style>
/* 상단 여백 축소 */
.stMainBlockContainer {
    padding-top: 1rem !important;
}
/* 버튼 공통: 덜 둥글고 compact하게 */
button[kind="secondary"],
button[kind="primary"],
.stButton > button,
.stDownloadButton > button {
    border-radius: 4px !important;
    padding: 0.3rem 0.8rem !important;
    font-size: 14px !important;
    min-height: 0 !important;
    line-height: 1.4 !important;
}
</style>
"""

SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
[data-testid="stSidebar"] p {
    margin-bottom: 0.3rem;
}
.sidebar-section {
    margin: 4px 0;
}
.sidebar-heading {
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 6px;
    color: inherit;
}
.sidebar-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    line-height: 1.4;
}
.sidebar-table td {
    padding: 2px 4px 2px 0;
    vertical-align: top;
}
.sidebar-table td.label {
    color: rgba(128,128,128,0.9);
    white-space: nowrap;
    width: 20%;
    padding-right: 4px;
}
</style>
"""

DIAG_CARD_CSS = """
<style>
/* 진단 결과 — 단일 박스 */
.diag-box {
    background: rgba(128,128,128,0.06);
    border-radius: 8px;
    padding: 16px 20px;
}
.diag-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 10px 0;
    border-bottom: 1px solid rgba(128,128,128,0.1);
}
.diag-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.diag-row:first-child {
    padding-top: 0;
}
.diag-label {
    font-size: 13px;
    color: rgba(128,128,128,0.75);
    font-weight: 600;
    flex-shrink: 0;
    margin-right: 16px;
}
.diag-value {
    font-size: 14px;
    font-weight: 600;
    color: inherit;
    text-align: right;
}
.risk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 700;
    color: white;
}
</style>
"""

REPORT_CSS = """
<style>
/* 분석 리포트 문서 컨테이너 */
.report-container {
    background: rgba(128,128,128,0.04);
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 8px;
    padding: 28px 32px;
    font-size: 14px;
    line-height: 1.75;
    color: inherit;
}
.report-container h1 {
    font-size: 20px;
    font-weight: 700;
    margin: 0 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(128,128,128,0.15);
}
.report-container h2 {
    font-size: 16px;
    font-weight: 700;
    margin: 20px 0 10px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(128,128,128,0.1);
}
.report-container h3 {
    font-size: 15px;
    font-weight: 600;
    margin: 16px 0 8px 0;
}
.report-container h4 {
    font-size: 14px;
    font-weight: 600;
    margin: 12px 0 6px 0;
    color: rgba(128,128,128,0.85);
}
.report-container p {
    margin: 0 0 10px 0;
}
.report-container ul {
    margin: 6px 0 12px 0;
    padding-left: 20px;
}
.report-container li {
    margin-bottom: 4px;
}
.report-container hr {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.12);
    margin: 16px 0;
}
.report-container strong {
    font-weight: 600;
}
.report-container ol {
    margin: 6px 0 12px 0;
    padding-left: 20px;
}
.report-container ol li {
    margin-bottom: 4px;
}
/* 작업지시서 테이블 */
.wo-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 8px 0 12px 0;
}
.wo-table th {
    background: rgba(128,128,128,0.08);
    font-weight: 600;
    text-align: left;
    padding: 8px 10px;
    border-bottom: 2px solid rgba(128,128,128,0.15);
    font-size: 12px;
    color: rgba(128,128,128,0.75);
}
.wo-table td {
    padding: 7px 10px;
    border-bottom: 1px solid rgba(128,128,128,0.08);
}
.wo-table tbody tr:last-child td {
    border-bottom: none;
}
/* diag-box inside report-container */
.report-container .diag-box {
    margin: 4px 0 12px 0;
}
</style>
"""

THOUGHT_CSS = """
<style>
/* ── 스트리밍 중 타임라인 (render_thoughts_live, HTML 기반) ── */
.thought-timeline {
    position: relative;
    padding-left: 24px;
    margin: 8px 0;
}
.thought-timeline::before {
    content: '';
    position: absolute;
    left: 7px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: rgba(128,128,128,0.2);
}
.thought-step {
    position: relative;
    margin-bottom: 12px;
}
.thought-step:last-child {
    margin-bottom: 0;
}
.thought-dot {
    position: absolute;
    left: -24px;
    top: 6px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1;
}
.thought-dot-done {
    background: rgba(40,167,69,0.15);
    border: 2px solid #28a745;
}
.thought-dot-done::after {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #28a745;
}
.thought-dot-active {
    background: rgba(13,110,253,0.15);
    border: 2px solid #0d6efd;
    animation: pulse-dot 1.5s ease-in-out infinite;
}
.thought-dot-active::after {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #0d6efd;
}
.thought-dot-tool {
    background: rgba(111,66,193,0.15);
    border: 2px solid #6f42c1;
}
.thought-dot-tool::after {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6f42c1;
}
.thought-dot-node {
    background: rgba(32,201,151,0.15);
    border: 2px solid #20c997;
}
.thought-dot-node::after {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #20c997;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}
.thought-accordion {
    margin: 0;
}
.thought-accordion summary {
    font-size: 13px;
    font-weight: 600;
    color: rgba(128,128,128,0.8);
    cursor: pointer;
    user-select: none;
    list-style: none;
    padding: 4px 0;
}
.thought-accordion summary::-webkit-details-marker {
    display: none;
}
.thought-accordion summary::before {
    content: '>';
    display: inline-block;
    margin-right: 6px;
    font-size: 11px;
    transition: transform 0.15s ease;
}
.thought-accordion[open] summary::before {
    transform: rotate(90deg);
}
.thought-accordion summary:hover {
    color: inherit;
}
.thought-body {
    background: rgba(128,128,128,0.06);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.6;
    color: inherit;
    margin-top: 6px;
}
.thought-body p {
    margin: 0 0 8px 0;
}
.thought-body p:last-child {
    margin-bottom: 0;
}
/* Tool badge (스트리밍용) */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(13,110,253,0.08);
    border: 1px solid rgba(13,110,253,0.2);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: #0d6efd;
    margin: 4px 4px 4px 0;
    cursor: default;
}
.tool-badge .tool-icon {
    font-size: 11px;
    opacity: 0.7;
}
.tool-detail {
    margin-top: 6px;
    border-left: 2px solid rgba(13,110,253,0.2);
    padding-left: 12px;
}
.tool-detail summary {
    font-size: 12px;
    color: rgba(128,128,128,0.7);
    cursor: pointer;
    user-select: none;
    padding: 2px 0;
}
.tool-detail summary:hover {
    color: #0d6efd;
}
.tool-detail pre {
    background: rgba(128,128,128,0.06);
    border-radius: 4px;
    padding: 8px 10px;
    font-size: 12px;
    line-height: 1.4;
    overflow-x: auto;
    margin: 4px 0;
    white-space: pre-wrap;
    word-break: break-word;
}

</style>
"""

CHAT_PANEL_CSS = """
<style>
/* ── 채팅 토글 버튼 ── */
.chat-toggle-btn button {
    border-radius: 20px !important;
    padding: 4px 16px !important;
    font-size: 13px !important;
    min-height: 0 !important;
    line-height: 1.4 !important;
}

/* ── 채팅 우측 컬럼 고정 ── */
[data-testid="stHorizontalBlock"]:has(.chat-messages)
    > [data-testid="stColumn"]:last-child {
    position: sticky;
    top: 1rem;
    align-self: flex-start;
    border-left: 1px solid rgba(128,128,128,0.12);
    padding-left: 12px !important;
}

/* ── 채팅 헤더 ── */
.chat-header-title {
    font-size: 14px;
    font-weight: 700;
    color: inherit;
}
.chat-header-divider {
    border-bottom: 1px solid rgba(128,128,128,0.12);
    margin: 0 0 8px 0;
}

/* ── 메시지 영역 ── */
.chat-messages {
    height: calc(100vh - 260px);
    max-height: calc(100vh - 260px);
    overflow-y: auto;
    padding: 16px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: rgba(128,128,128,0.02);
    border: 1px solid rgba(128,128,128,0.1);
    border-radius: 8px;
    scroll-behavior: smooth;
}

/* 빈 상태 */
.chat-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: rgba(128,128,128,0.45);
}
.chat-empty-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: rgba(128,128,128,0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 700;
    color: rgba(128,128,128,0.3);
}
.chat-empty-text {
    font-size: 13px;
    text-align: center;
    line-height: 1.5;
}

/* ── 사용자 메시지 (우측, 노란 말풍선) ── */
.chat-bubble-row-user {
    display: flex;
    justify-content: flex-end;
    padding: 0 4px;
}
.chat-msg-user {
    background: #FEE500;
    color: #1a1a1a;
    border-radius: 16px 16px 4px 16px;
    padding: 9px 14px;
    font-size: 14px;
    line-height: 1.55;
    max-width: 82%;
    word-break: break-word;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
}

/* 다크모드 사용자 메시지 */
@media (prefers-color-scheme: dark) {
    .chat-msg-user {
        background: #d4c100;
        color: #1a1a1a;
    }
}
/* Streamlit 다크모드 대응 */
[data-testid="stAppViewContainer"][data-theme="dark"] .chat-msg-user,
html[data-theme="dark"] .chat-msg-user {
    background: #d4c100;
    color: #1a1a1a;
}

/* ── AI 메시지 (좌측, 프로필 아이콘 + 흰 말풍선) ── */
.chat-bubble-row-assistant {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 8px;
    padding: 0 4px;
}
.chat-ai-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.chat-ai-avatar-text {
    font-size: 11px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.chat-msg-assistant {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.08);
    border-radius: 16px 16px 16px 4px;
    padding: 9px 14px;
    font-size: 14px;
    line-height: 1.55;
    max-width: 78%;
    word-break: break-word;
    color: inherit;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

/* ── 타이핑 인디케이터 ── */
.chat-typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 0;
}
.chat-typing-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: rgba(128,128,128,0.4);
    animation: chat-typing-bounce 1.4s ease-in-out infinite;
}
.chat-typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}
.chat-typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}
@keyframes chat-typing-bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
    30% { transform: translateY(-4px); opacity: 1; }
}

/* ── 스트리밍 커서 ── */
.chat-streaming-dot::after {
    content: '';
    display: inline-block;
    width: 2px;
    height: 14px;
    background: currentColor;
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: chat-cursor-blink 0.8s step-end infinite;
}
@keyframes chat-cursor-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

/* ── 입력 영역 ── */
.chat-input-area {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-top: 1px solid rgba(128,128,128,0.15);
    background: rgba(128,128,128,0.04);
    flex-shrink: 0;
}

/* (전송 버튼 제거: Enter 키로 전송) */

/* ── 마크다운 렌더링 (AI 말풍선 내부) ── */
.chat-md h4 {
    font-size: 13px;
    font-weight: 700;
    margin: 10px 0 6px 0;
    color: inherit;
}
.chat-md h4:first-child {
    margin-top: 0;
}
.chat-md p {
    margin: 0 0 6px 0;
    line-height: 1.55;
}
.chat-md p:last-child {
    margin-bottom: 0;
}
.chat-md ul, .chat-md ol {
    margin: 4px 0 8px 0;
    padding-left: 18px;
}
.chat-md li {
    margin-bottom: 2px;
    font-size: 13px;
    line-height: 1.5;
}
.chat-md code {
    background: rgba(128,128,128,0.1);
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 12px;
}
.chat-md pre {
    background: rgba(128,128,128,0.08);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 12px;
    overflow-x: auto;
    margin: 6px 0;
}
.chat-md hr {
    border: none;
    border-top: 1px solid rgba(128,128,128,0.15);
    margin: 8px 0;
}
.chat-md strong {
    font-weight: 600;
}
.chat-md table {
    border-collapse: collapse;
    font-size: 12px;
    margin: 6px 0;
    width: 100%;
}
.chat-md th, .chat-md td {
    border: 1px solid rgba(128,128,128,0.15);
    padding: 4px 8px;
    text-align: left;
}
.chat-md th {
    background: rgba(128,128,128,0.06);
    font-weight: 600;
}

/* ── 자동 스크롤 anchor ── */
.chat-scroll-anchor {
    height: 1px;
    width: 100%;
}

/* Deep Search Think Mode */
.ds-think-container {
    background: #f0f4f8;
    border-left: 3px solid #4a90d9;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    font-size: 13px;
}
.ds-think-header {
    font-weight: 700;
    color: #2c5282;
    margin-bottom: 8px;
    font-size: 14px;
}
.ds-think-step {
    margin: 6px 0;
    padding: 8px;
    background: white;
    border-radius: 6px;
}
.ds-think-role {
    font-weight: 600;
    color: #4a5568;
    margin-bottom: 4px;
    font-size: 12px;
}
.ds-think-content {
    color: #2d3748;
    line-height: 1.5;
}
.ds-researcher {
    border-left: 2px solid #48bb78;
}
.ds-critic {
    border-left: 2px solid #ed8936;
}
</style>
"""
