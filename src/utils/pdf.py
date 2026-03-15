"""PDF generation utilities for work order documents.

Extracted from scripts/parse_maintenance_xlsx.py — reuses the same
MaintenancePDF class and rendering logic so that agent-generated work orders
produce PDFs visually identical to the RAG knowledge-base originals.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

FONT_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "fonts" / "NotoSansKR.ttf"
)


class MaintenancePDF(FPDF):
    """Custom PDF with Korean font support and proper text wrapping."""

    LABEL_W = 45
    LINE_H = 5.5
    PAGE_W = 210
    MARGIN = 10

    def __init__(self) -> None:
        super().__init__()
        self.add_font("NotoSansKR", "", FONT_PATH)
        self.add_font("NotoSansKR", "B", FONT_PATH)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_left_margin(self.MARGIN)
        self.set_right_margin(self.MARGIN)

    @property
    def content_w(self) -> float:
        return self.PAGE_W - self.l_margin - self.r_margin

    def header(self) -> None:
        pass

    def section_title(self, title: str) -> None:
        self._check_space(15)
        self.set_font("NotoSansKR", "B", 10)
        self.set_fill_color(230, 230, 230)
        self.cell(self.content_w, 7, f"  {title}", ln=True, fill=True)
        self.ln(1)

    def _check_space(self, min_h: float = 12.0) -> None:
        if self.get_y() + min_h > self.h - self.b_margin:
            self.add_page()

    def field(self, label: str, value: str) -> None:
        value = value or "-"
        self.set_font("NotoSansKR", "", 9)
        val_w = self.content_w - self.LABEL_W
        lines = max(1, len(value) / max(val_w / 2.5, 1))
        est_h = max(self.LINE_H, lines * self.LINE_H)
        self._check_space(min(est_h + 2, 40))

        y_before = self.get_y()
        x0 = self.l_margin

        self.set_auto_page_break(auto=False)

        self.set_font("NotoSansKR", "B", 9)
        self.set_xy(x0, y_before)
        self.cell(self.LABEL_W, self.LINE_H, label)

        self.set_font("NotoSansKR", "", 9)
        self.set_xy(x0 + self.LABEL_W, y_before)
        self.multi_cell(val_w, self.LINE_H, value)

        self.set_auto_page_break(auto=True, margin=15)

        y_after = self.get_y()
        if y_after < y_before + self.LINE_H:
            self.set_y(y_before + self.LINE_H)

    def items_list(self, items: list[str]) -> None:
        self.set_font("NotoSansKR", "", 9)
        indent = 5
        for item in items:
            self._check_space(self.LINE_H * 2)
            self.set_x(self.l_margin + indent)
            w = self.content_w - indent
            self.multi_cell(w, self.LINE_H, f"\u2022 {item}")
            self.ln(0.5)

    def small_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        col_widths: list[float],
    ) -> None:
        row_h = 6

        self.set_font("NotoSansKR", "B", 8)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], row_h, h, border=1, align="C")
        self.ln()

        self.set_font("NotoSansKR", "", 8)
        for row_data in rows:
            max_lines = 1
            for i, val in enumerate(row_data):
                char_w = col_widths[i] / 4.5
                lines = max(1, len(val) / max(char_w, 1))
                max_lines = max(max_lines, lines)
            cell_h = max(row_h, int(max_lines) * row_h)

            if self.get_y() + cell_h > self.h - self.b_margin:
                self.add_page()
                self.set_font("NotoSansKR", "B", 8)
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], row_h, h, border=1, align="C")
                self.ln()
                self.set_font("NotoSansKR", "", 8)

            y0 = self.get_y()
            x0 = self.l_margin
            actual_max_y = y0

            for i, val in enumerate(row_data):
                self.set_xy(x0, y0)
                self.multi_cell(col_widths[i], row_h, val, border=0)
                actual_max_y = max(actual_max_y, self.get_y())
                x0 += col_widths[i]

            actual_h = actual_max_y - y0
            if actual_h < row_h:
                actual_h = row_h
            x0 = self.l_margin
            for i in range(len(row_data)):
                self.rect(x0, y0, col_widths[i], actual_h)
                x0 += col_widths[i]

            self.set_y(y0 + actual_h)


def generate_work_order_pdf_bytes(work_order: dict) -> bytes:
    """Generate a work-order PDF from a structured dict.

    Args:
        work_order: Dict with the same schema as RAG knowledge-base work orders.

    Returns:
        PDF file content as bytes.
    """
    pdf = MaintenancePDF()
    pdf.LINE_H = 5.0
    pdf.set_auto_page_break(auto=True, margin=10)

    pdf.add_page()
    pdf.set_font("NotoSansKR", "B", 14)
    pdf.cell(0, 10, "\uacf5\uc7a5\uc124\ube44 \ubcf4\uc804\uc791\uc5c5 \uc791\uc5c5\uc9c0\uc2dc\uc11c", ln=True, align="C")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # 1. 기본 정보
    pdf.section_title("\uae30\ubcf8 \uc815\ubcf4")
    pdf.field("\uc791\uc5c5\uc9c0\uc2dc \ubc88\ud638", work_order.get("wo_number", ""))
    pdf.field("\uc124\ube44\uba85 / \uc124\ube44\ubc88\ud638", work_order.get("equipment", ""))
    pdf.field("\uc124\ube44 \uc704\uce58", work_order.get("location", ""))
    pdf.field("\uc791\uc5c5 \uc720\ud615", work_order.get("work_type", ""))
    pdf.field("\uc791\uc5c5 \uc694\uccad\uc77c", work_order.get("request_date", ""))
    pdf.field("\uc791\uc5c5 \uc608\uc815\uc77c", work_order.get("scheduled_date", ""))
    pdf.field("\uc644\ub8cc \uc608\uc815\uc77c", work_order.get("due_date", ""))
    pdf.field("\uc791\uc5c5 \ub2f4\ub2f9\uc790", work_order.get("assignee", ""))

    # 2. 작업 내용
    pdf.ln(1)
    pdf.section_title("\uc791\uc5c5 \ub0b4\uc6a9")
    pdf.field("\uc791\uc5c5 \ub0b4\uc6a9 \uc694\uc57d", work_order.get("summary", ""))
    pdf.field("\uc548\uc804\uc0ac\ud56d", work_order.get("safety", ""))

    # 3. 체크리스트
    if work_order.get("checklist"):
        pdf.ln(1)
        pdf.section_title("\uc791\uc5c5 \uc0c1\uc138 \uccb4\ud06c\ub9ac\uc2a4\ud2b8")
        pdf.items_list(work_order["checklist"])

    # 4. 필요 자재
    if work_order.get("materials"):
        pdf.ln(1)
        pdf.section_title("\ud544\uc694 \uc790\uc7ac")
        widths = [28.0, 32.0, 32.0, 14.0, 14.0, 70.0]
        headers = ["\ucf54\ub4dc", "\uc790\uc7ac\uba85", "\uaddc\uaca9", "\uc218\ub7c9", "\ub2e8\uc704", "\ube44\uace0"]
        rows = [
            [
                m.get("code", ""),
                m.get("name", ""),
                m.get("spec", ""),
                str(m.get("qty", "")),
                m.get("unit", ""),
                m.get("note", ""),
            ]
            for m in work_order["materials"]
        ]
        pdf.small_table(headers, rows, widths)

    # 5. 필요 공구
    if work_order.get("tools"):
        pdf.ln(1)
        pdf.section_title("\ud544\uc694 \uacf5\uad6c \ubc0f \uc7a5\ube44")
        widths = [28.0, 32.0, 38.0, 14.0, 14.0, 64.0]
        headers = ["\ucf54\ub4dc", "\uba85\uce6d", "\uaddc\uaca9", "\uc218\ub7c9", "\ub2e8\uc704", "\ube44\uace0"]
        rows = [
            [
                t.get("code", ""),
                t.get("name", ""),
                t.get("spec", ""),
                str(t.get("qty", "")),
                t.get("unit", ""),
                t.get("note", ""),
            ]
            for t in work_order["tools"]
        ]
        pdf.small_table(headers, rows, widths)

    # 6. 작업 후 확인사항
    if work_order.get("post_checks"):
        pdf.ln(1)
        pdf.section_title("\uc791\uc5c5 \ud6c4 \ud655\uc778\uc0ac\ud56d")
        pdf.items_list(work_order["post_checks"])

    # 7. 승인 및 결과
    pdf.ln(1)
    pdf.section_title("\uc2b9\uc778 \ubc0f \uacb0\uacfc")
    pdf.field("\uc791\uc5c5 \uc2b9\uc778\uc790", work_order.get("approver", ""))
    pdf.field("\uc791\uc5c5 \uc644\ub8cc\uc77c\uc2dc", work_order.get("completion_date", ""))
    pdf.field("\uc791\uc5c5 \uacb0\uacfc \uc694\uc57d", work_order.get("result_summary", ""))

    if work_order.get("attachments"):
        pdf.field("\ucca8\ubd80\ud30c\uc77c", ", ".join(work_order["attachments"]))

    return bytes(pdf.output())
