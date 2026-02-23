"""알림 MCP Server.

정비 담당자에게 분석 결과 및 정비 권고 알림을 전송한다.
PoC에서는 로그 출력으로 대체한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotificationResult:
    """알림 전송 결과."""

    success: bool
    timestamp: str
    message: str


class NotificationServer:
    """정비 담당자 알림 서버.

    PoC에서는 로그 출력으로 대체한다.
    Production 환경에서는 이메일, SMS, Slack 등으로 확장 가능.
    """

    def notify_maintenance_staff(
        self,
        message: str,
        risk_level: str,
        equipment_id: str,
    ) -> NotificationResult:
        """정비 담당자에게 알림을 전송.

        Args:
            message: 알림 메시지 (분석 결과 요약 + 정비 권고).
            risk_level: 위험도 (normal / watch / warning / critical).
            equipment_id: 설비 ID.

        Returns:
            NotificationResult with success status.
        """
        timestamp = datetime.now().isoformat()

        logger.warning(
            f"\n{'=' * 60}\n"
            f"[NOTIFICATION] 정비 담당자 알림 전송\n"
            f"  시각: {timestamp}\n"
            f"  설비: {equipment_id}\n"
            f"  위험도: {risk_level.upper()}\n"
            f"  내용: {message}\n"
            f"{'=' * 60}"
        )

        return NotificationResult(
            success=True,
            timestamp=timestamp,
            message=f"알림 전송 완료 (설비: {equipment_id}, 위험도: {risk_level})",
        )
