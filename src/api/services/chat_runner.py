"""Chat runner interface and mock implementation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from api.models.stream import ChatCompletedEvent, ChatErrorEvent, ChatTokenEvent
from api.services.run_manager import RunManager


@runtime_checkable
class ChatRunner(Protocol):
    """Protocol for chat runners."""

    async def run(
        self,
        run_id: str,
        session_id: str,
        message: str,
        run_manager: RunManager,
        *,
        deep_search: bool = False,
    ) -> None:
        """Process a chat message and stream response."""
        ...


# Keyword-based mock responses
_MOCK_RESPONSES: dict[str, str] = {
    # 분석 결과 관련
    "rul": (
        "현재 ML 모델(RUL-LSTM-v1)의 잔여수명 예측 결과를 기반으로 설명드리겠습니다. "
        "예측된 RUL은 운전 조건(회전속도, 하중)이 현재와 동일하게 유지된다는 가정 하에 산출되었습니다. "
        "신뢰구간을 고려하면 하한값을 기준으로 정비 계획을 수립하시는 것을 권장합니다. "
        "실제 잔여수명은 윤활 상태, 온도 변화, 부하 변동 등에 따라 달라질 수 있으므로 "
        "모니터링 주기를 단축하여 추세를 관찰하시기 바랍니다."
    ),
    "정비": (
        "권장 정비 조치에 대해 상세히 안내드리겠습니다. "
        "베어링 교체 작업 시 다음 사항을 고려해 주세요:\n"
        "1. 교체 전 축 정렬 상태를 반드시 확인하세요.\n"
        "2. 신규 베어링 장착 시 제조사 권장 토크값을 준수하세요.\n"
        "3. 윤활유는 ISO VG 68 등급 이상을 사용하세요.\n"
        "4. 교체 후 시운전 시 최소 30분간 진동/온도를 모니터링하세요."
    ),
    "원인": (
        "결함의 근본 원인에 대해 분석 결과를 설명드리겠습니다. "
        "주파수 스펙트럼 분석에서 결함 특성 주파수와 그 고조파 성분이 확인되었습니다. "
        "이는 베어링 레이스웨이의 표면 손상(피팅 또는 스폴링)으로 인한 "
        "충격 진동이 발생하고 있음을 나타냅니다. "
        "가능한 원인으로는 과부하, 윤활 부족, 오염물 침입, 또는 피로 마모가 있습니다."
    ),
    # RAG / 도메인 지식 관련
    "bpfo": (
        "BPFO(Ball Pass Frequency Outer race)는 외륜 결함 특성 주파수입니다. "
        "전동체가 외륜의 손상 부위를 통과할 때마다 충격이 발생하며, "
        "이 충격의 반복 주파수가 BPFO입니다. "
        "BPFO = (N/2) * (1 - Bd/Pd * cos(alpha)) * RPM/60 으로 계산됩니다. "
        "BPFO의 2x, 3x 고조파가 함께 나타나면 결함 진행을 의미합니다."
    ),
    "bpfi": (
        "BPFI(Ball Pass Frequency Inner race)는 내륜 결함 특성 주파수입니다. "
        "전동체가 내륜의 손상 부위를 통과할 때 발생하는 충격의 반복 주파수이며, "
        "BPFI = (N/2) * (1 + Bd/Pd * cos(alpha)) * RPM/60 으로 계산됩니다. "
        "내륜 결함의 특징은 BPFI 주변에 회전속도 간격의 사이드밴드가 나타나는 것입니다."
    ),
    "주파수": (
        "베어링 결함 특성 주파수는 4가지가 있습니다:\n"
        "- BPFO: 외륜 결함 주파수\n"
        "- BPFI: 내륜 결함 주파수\n"
        "- BSF: 전동체(볼) 결함 주파수\n"
        "- FTF: 케이지(보유기) 결함 주파수\n"
        "각 주파수는 베어링 기하학적 사양(볼 직경, 피치 직경, 접촉각, 볼 수)과 "
        "회전속도로부터 계산됩니다."
    ),
    "kurtosis": (
        "Kurtosis(첨도)는 진동 신호의 충격성을 나타내는 통계 지표입니다. "
        "정상 베어링은 kurtosis가 약 3.0(가우시안 분포)이며, "
        "결함 초기에는 간헐적 충격으로 인해 kurtosis가 급격히 증가합니다. "
        "일반적으로 kurtosis > 4.0이면 주의, > 6.0이면 결함 진행으로 판단합니다. "
        "다만 결함이 심하게 진행되면 연속적 충격으로 오히려 kurtosis가 감소할 수 있습니다."
    ),
    "윤활": (
        "베어링 윤활 관련 주요 사항입니다:\n"
        "- 윤활 부족 시 고주파 대역(1-5kHz) 에너지가 증가합니다.\n"
        "- 적정 윤활유 등급: ISO VG 68 (일반 산업용 베어링 기준)\n"
        "- 재윤활 주기는 운전 조건(온도, 속도, 하중)에 따라 결정합니다.\n"
        "- 과도한 윤활도 발열과 성능 저하를 유발할 수 있습니다.\n"
        "- 윤활 상태 모니터링에는 초음파 측정이 효과적입니다."
    ),
    "진동": (
        "진동 분석은 베어링 상태 진단의 핵심 기법입니다. "
        "시간 영역에서는 RMS, Peak, Crest Factor, Kurtosis 등의 통계 지표를 활용하고, "
        "주파수 영역에서는 결함 특성 주파수(BPFO, BPFI, BSF, FTF)의 진폭과 고조파를 분석합니다. "
        "ISO 10816 기준으로 진동 심각도를 분류하며, "
        "RMS 속도 기준 4.5mm/s 초과 시 경고, 11.2mm/s 초과 시 위험으로 판단합니다."
    ),
}

_DEFAULT_RESPONSE = (
    "베어링 상태 진단 및 예지보전에 대해 도움을 드릴 수 있습니다. "
    "진동 분석, 결함 주파수, RUL 예측, 정비 계획, 윤활 관리 등 "
    "궁금한 내용을 질문해 주세요."
)


def _select_response(message: str) -> str:
    """Select mock response based on keywords in the message."""
    lower = message.lower()
    for keyword, response in _MOCK_RESPONSES.items():
        if keyword in lower:
            return response
    return _DEFAULT_RESPONSE


class MockChatRunner:
    """Mock chat runner for demo/testing.

    Simulates chat response with keyword-based mock answers
    and token-level streaming.
    """

    async def run(
        self,
        run_id: str,
        session_id: str,
        message: str,
        run_manager: RunManager,
        *,
        deep_search: bool = False,
    ) -> None:
        """Simulate chat response streaming."""
        now = lambda: datetime.now(timezone.utc).isoformat()

        try:
            response = _select_response(message)

            # Stream tokens (word by word)
            for token in response.split():
                await run_manager.emit_chat_event(
                    session_id,
                    ChatTokenEvent(
                        run_id=run_id,
                        session_id=session_id,
                        token=token + " ",
                    ),
                )
                await asyncio.sleep(0.05)

            # Chat completed
            await run_manager.emit_chat_event(
                session_id,
                ChatCompletedEvent(
                    run_id=run_id,
                    session_id=session_id,
                    content=response,
                    timestamp=now(),
                ),
            )

        except Exception as e:
            await run_manager.emit_chat_event(
                session_id,
                ChatErrorEvent(
                    run_id=run_id,
                    session_id=session_id,
                    message=str(e),
                    timestamp=now(),
                ),
            )

        finally:
            await run_manager.end_chat_stream(session_id)
