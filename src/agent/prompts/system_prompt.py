"""PdM Agent 시스템 프롬프트.

docs/system-prompt.md의 내용을 Python 문자열로 제공한다.
"""

from __future__ import annotations

from pathlib import Path

# docs/system-prompt.md에서 프롬프트 본문만 추출 (코드블록 내부)
_PROMPT_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "system-prompt.md"


def load_system_prompt() -> str:
    """시스템 프롬프트를 docs/system-prompt.md에서 로드.

    마크다운 파일 내 코드블록(``` ... ```) 안의 내용을 추출한다.
    파일이 없으면 내장 폴백 프롬프트를 반환한다.

    Returns:
        시스템 프롬프트 문자열.
    """
    if _PROMPT_FILE.exists():
        content = _PROMPT_FILE.read_text(encoding="utf-8")
        # 코드블록 내부 추출
        lines = content.split("\n")
        in_block = False
        block_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip().startswith("```") and in_block:
                break
            elif in_block:
                block_lines.append(line)
        if block_lines:
            return "\n".join(block_lines)

    return _FALLBACK_PROMPT


_FALLBACK_PROMPT = """당신은 제조 설비의 예지보전(Predictive Maintenance)을 전담하는 AI 에이전트입니다.
당신의 이름은 PdM Agent이며, 베어링 진동 기반 상태 감시 및 고장 예측 분야의 전문가로서 행동합니다.

Edge 시스템에서 전달받은 이벤트 페이로드를 분석하여 결함 유형을 식별하고,
결함 진행 단계를 판정하며, 최종적으로 정비 권고를 생성합니다.

수치 계산은 하지 않습니다. Edge가 산출한 값을 읽고 해석하고 의미를 부여합니다.
모르는 것은 모른다고 말합니다."""
