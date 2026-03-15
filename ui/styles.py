"""CSS 스타일 상수."""

from __future__ import annotations

# 위험도별 색상
RISK_COLORS: dict[str, str] = {
    "normal": "#28a745",
    "watch": "#ffc107",
    "warning": "#fd7e14",
    "critical": "#dc3545",
}

THOUGHT_CSS = """
<style>
/* 진단 카드 */
.diag-card {
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    color: white;
    font-weight: bold;
}
.diag-card .card-label {
    font-size: 12px;
    opacity: 0.85;
    margin-bottom: 4px;
}
.diag-card .card-value {
    font-size: 20px;
}
</style>
"""
