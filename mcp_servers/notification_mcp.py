"""알림 FastMCP Server.

NotificationServer의 알림 전송 메서드를 MCP Tool로 래핑한다.
stdio transport로 서브프로세스에서 실행된다.
"""

from __future__ import annotations

import json
import os
import sys

# 서브프로세스에서 프로젝트 루트 import 보장
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("notification-server")

_server_instance: "NotificationServer | None" = None


def _get_server():
    """지연 초기화로 NotificationServer 인스턴스 반환."""
    global _server_instance
    if _server_instance is None:
        from mcp_servers.notification_server import NotificationServer

        _server_instance = NotificationServer()
    return _server_instance


@mcp.tool()
def notify_maintenance_staff(
    message: str,
    risk_level: str,
    equipment_id: str,
) -> str:
    """정비 담당자에게 분석 결과 및 정비 권고 알림을 전송합니다. 위험도 Watch 이상이거나 인간의 확인이 필요할 때 호출합니다.

    Args:
        message: 알림 메시지 (분석 결과 요약 + 정비 권고).
        risk_level: 위험도 (watch / warning / critical).
        equipment_id: 설비 ID.
    """
    server = _get_server()
    result = server.notify_maintenance_staff(
        message=message, risk_level=risk_level, equipment_id=equipment_id
    )
    return json.dumps(
        {"success": result.success, "message": result.message},
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run()
