"""알림 Action Skill.

NotificationServer를 LangGraph Tool로 등록한다.
PoC에서는 로그 출력, Production에서는 이메일/Slack 등으로 확장 가능.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_notification_server = None


def _get_server():
    """NotificationServer 인스턴스를 지연 초기화로 반환."""
    global _notification_server
    if _notification_server is None:
        from mcp_servers.notification_server import NotificationServer

        _notification_server = NotificationServer()
        logger.info("[action-skill] NotificationServer 초기화 완료")
    return _notification_server


@tool
def notify_maintenance_staff(
    message: str,
    risk_level: str,
    equipment_id: str,
) -> str:
    """정비 담당자에게 분석 결과 및 정비 권고 알림을 전송합니다. 위험도 Watch 이상이거나 인간의 확인이 필요할 때 호출합니다.

    Args:
        message: 알림 메시지 (분석 결과 요약 + 정비 권고).
        risk_level: 위험도 (normal / watch / warning / critical).
        equipment_id: 설비 ID.
    """
    server = _get_server()
    result = server.notify_maintenance_staff(
        message=message, risk_level=risk_level, equipment_id=equipment_id
    )
    logger.info(
        "[action-skill] notify_maintenance_staff: equipment=%s, risk=%s",
        equipment_id,
        risk_level,
    )
    return json.dumps(
        {"success": result.success, "message": result.message},
        ensure_ascii=False,
    )


def get_notification_tools() -> list:
    """알림 Action Skill Tool 리스트를 반환."""
    return [notify_maintenance_staff]
