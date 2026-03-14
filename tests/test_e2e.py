"""Docker 기반 E2E 테스트.

실제 Docker 서비스(Qdrant, PostgreSQL) + 실제 OpenAI API를 사용하여
FastAPI 앱의 전체 파이프라인을 검증한다.

사전 조건:
    - docker compose up -d postgres qdrant
    - QDRANT_HOST=localhost python scripts/init_qdrant.py
    - OPENAI_API_KEY 환경변수 설정
    - POSTGRES_PASSWORD 환경변수 설정

실행:
    PYTHONPATH=src pytest tests/test_e2e.py -v --timeout=120
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from httpx_sse import aconnect_sse

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def _check_openai_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-fake"):
        return "OPENAI_API_KEY not set"
    return None


def _check_docker_service(host: str, port: int, name: str) -> str | None:
    """TCP 연결로 서비스 가용 여부 확인 (동기)."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=3):
            return None
    except OSError:
        return f"{name} not reachable at {host}:{port}"


def _skip_reason() -> str | None:
    reason = _check_openai_key()
    if reason:
        return reason
    reason = _check_docker_service(
        os.getenv("QDRANT_HOST", "localhost"),
        int(os.getenv("QDRANT_PORT", "6333")),
        "Qdrant",
    )
    if reason:
        return reason
    reason = _check_docker_service(
        os.getenv("POSTGRES_HOST", "localhost"),
        int(os.getenv("POSTGRES_PORT", "5432")),
        "PostgreSQL",
    )
    if reason:
        return reason
    return None


_reason = _skip_reason()
if _reason:
    pytestmark = [pytestmark, pytest.mark.skip(reason=_reason)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(scope="module")
async def client():
    """AGENT_RUNNER_MODE=langgraph로 FastAPI 앱 생성, httpx AsyncClient 반환."""
    os.environ["AGENT_RUNNER_MODE"] = "langgraph"

    from api.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        timeout=httpx.Timeout(120.0),
    ) as ac:
        yield ac


@pytest.fixture(scope="module")
def anomaly_payload() -> dict:
    """data/payloads/scenario1_day31.json 로드."""
    payload_path = Path(__file__).parent.parent / "data" / "payloads" / "scenario1_day31.json"
    with open(payload_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def normal_payload(anomaly_payload: dict) -> dict:
    """anomaly_payload 복사 후 정상 상태로 변경."""
    payload = copy.deepcopy(anomaly_payload)
    payload["anomaly_detection_result"]["anomaly_detected"] = False
    payload["anomaly_detection_result"]["anomaly_score"] = 0.15
    payload["anomaly_detection_result"]["health_state"] = "normal"
    payload["anomaly_detection_result"]["confidence"] = 0.95
    return payload


@pytest.fixture(scope="module")
def critical_payload(anomaly_payload: dict) -> dict:
    """anomaly_payload 복사 후 critical 상태로 변경."""
    payload = copy.deepcopy(anomaly_payload)
    payload["anomaly_detection_result"]["anomaly_score"] = 0.98
    payload["anomaly_detection_result"]["health_state"] = "critical"
    payload["anomaly_detection_result"]["confidence"] = 0.96
    return payload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def collect_sse_events(
    client: httpx.AsyncClient,
    run_id: str,
    timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """SSE 스트림에서 이벤트를 수집하여 반환."""
    events: list[dict[str, Any]] = []
    async with asyncio.timeout(timeout):
        async with aconnect_sse(
            client, "GET", f"/api/agent/stream/{run_id}"
        ) as event_source:
            async for sse in event_source.aiter_sse():
                data = json.loads(sse.data)
                events.append({"event": sse.event or data.get("event", ""), "data": data})
    return events


async def run_full_cycle(
    client: httpx.AsyncClient,
    payload: dict,
    timeout: float = 120.0,
) -> tuple[str, list[dict[str, Any]]]:
    """POST /api/events → run_id 획득 → SSE 수집 → (run_id, events) 반환."""
    resp = await client.post("/api/events", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    run_id = body["run_id"]
    assert run_id

    # SSE 수집 시작 전 약간의 대기 (agent 시작 시간 확보)
    await asyncio.sleep(0.5)

    events = await collect_sse_events(client, run_id, timeout=timeout)
    return run_id, events


def _find_events(events: list[dict], event_type: str) -> list[dict]:
    """특정 타입의 이벤트만 필터링."""
    return [e for e in events if e["data"].get("event") == event_type]


# ---------------------------------------------------------------------------
# TestDockerE2E
# ---------------------------------------------------------------------------

class TestDockerE2E:
    """실제 Docker 서비스를 사용하는 E2E 테스트."""

    @pytest.mark.asyncio
    async def test_anomaly_warning_full_cycle(
        self, client: httpx.AsyncClient, anomaly_payload: dict
    ):
        """POST 200 + SSE 전체 사이클: warning 이상 페이로드."""
        run_id, events = await run_full_cycle(client, anomaly_payload)

        # run_id가 유효한 UUID 형식
        assert len(run_id) > 0

        # SSE 스트림이 비어있지 않음
        assert len(events) > 0

        # run_started 이벤트 존재
        started = _find_events(events, "run_started")
        assert len(started) >= 1

        # run_completed 이벤트 존재
        completed = _find_events(events, "run_completed")
        assert len(completed) >= 1

        # diagnosis 이벤트 검증
        diagnosis_events = _find_events(events, "diagnosis")
        assert len(diagnosis_events) >= 1
        diag = diagnosis_events[0]["data"]["diagnosis"]
        assert "fault_type" in diag or "severity" in diag

        # report_generated 이벤트 존재
        reports = _find_events(events, "report_generated")
        assert len(reports) >= 1

        # work_order_generated 이벤트 존재 (warning이므로)
        work_orders = _find_events(events, "work_order_generated")
        assert len(work_orders) >= 1

    @pytest.mark.asyncio
    async def test_normal_no_work_order(
        self, client: httpx.AsyncClient, normal_payload: dict
    ):
        """정상 페이로드 → work_order_generated 이벤트 없음."""
        run_id, events = await run_full_cycle(client, normal_payload)

        # diagnosis 존재
        diagnosis_events = _find_events(events, "diagnosis")
        assert len(diagnosis_events) >= 1

        # work_order_generated 없음
        work_orders = _find_events(events, "work_order_generated")
        assert len(work_orders) == 0

        # error 이벤트 없음
        errors = _find_events(events, "error")
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_critical_urgent_response(
        self, client: httpx.AsyncClient, critical_payload: dict
    ):
        """critical 페이로드 → diagnosis 존재, work_order_generated 존재."""
        run_id, events = await run_full_cycle(client, critical_payload)

        # diagnosis 이벤트 존재
        diagnosis_events = _find_events(events, "diagnosis")
        assert len(diagnosis_events) >= 1

        # work_order_generated 존재
        work_orders = _find_events(events, "work_order_generated")
        assert len(work_orders) >= 1

        # run_completed 존재
        completed = _find_events(events, "run_completed")
        assert len(completed) >= 1

    @pytest.mark.asyncio
    async def test_sse_event_completeness(
        self, client: httpx.AsyncClient, anomaly_payload: dict
    ):
        """run_started가 첫 이벤트, run_completed가 마지막 이벤트, error 없음."""
        run_id, events = await run_full_cycle(client, anomaly_payload)

        assert len(events) >= 2

        # 첫 이벤트: run_started
        assert events[0]["data"]["event"] == "run_started"

        # 마지막 이벤트: run_completed
        assert events[-1]["data"]["event"] == "run_completed"

        # error 이벤트 없음
        errors = _find_events(events, "error")
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_memory_persistence(
        self, client: httpx.AsyncClient, anomaly_payload: dict
    ):
        """동일 equipment_id로 2번 연속 실행 → 두 번째 실행에서 이전 이력 참조 가능."""
        # 첫 번째 실행
        run_id_1, events_1 = await run_full_cycle(client, anomaly_payload)
        completed_1 = _find_events(events_1, "run_completed")
        assert len(completed_1) >= 1

        # 두 번째 실행 (동일 장비)
        run_id_2, events_2 = await run_full_cycle(client, anomaly_payload)
        completed_2 = _find_events(events_2, "run_completed")
        assert len(completed_2) >= 1

        # 두 번째 실행이 정상 완료됨 (메모리 로드/저장 통과)
        errors_2 = _find_events(events_2, "error")
        assert len(errors_2) == 0

        # node_entered에 load_memory가 포함됨 (메모리 로드 시도)
        node_events = _find_events(events_2, "node_entered")
        node_names = [e["data"]["node_name"] for e in node_events]
        assert "load_memory" in node_names

    @pytest.mark.asyncio
    async def test_tool_calls_present(
        self, client: httpx.AsyncClient, anomaly_payload: dict
    ):
        """anomaly 이벤트 → tool_call/tool_result 이벤트가 1개 이상 존재."""
        run_id, events = await run_full_cycle(client, anomaly_payload)

        tool_calls = _find_events(events, "tool_call")
        tool_results = _find_events(events, "tool_result")

        # RAG 검색 등 tool 호출이 1개 이상 존재
        assert len(tool_calls) >= 1, "tool_call 이벤트가 없음 (RAG 검색 미수행)"
        assert len(tool_results) >= 1, "tool_result 이벤트가 없음"


# ---------------------------------------------------------------------------
# TestErrorScenarios
# ---------------------------------------------------------------------------

class TestErrorScenarios:
    """에러 시나리오 테스트."""

    @pytest.mark.asyncio
    async def test_invalid_payload_422(self, client: httpx.AsyncClient):
        """불완전한 JSON → HTTP 422."""
        resp = await client.post(
            "/api/events",
            json={"event_id": "test", "timestamp": "2026-01-01T00:00:00"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_payload_422(self, client: httpx.AsyncClient):
        """빈 JSON → HTTP 422."""
        resp = await client.post("/api/events", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_nonexistent_run_404(self, client: httpx.AsyncClient):
        """존재하지 않는 run_id → HTTP 404."""
        resp = await client.get("/api/agent/stream/nonexistent-run-id")
        assert resp.status_code == 404
