"""FastAPI HTTP/SSE 클라이언트."""

from __future__ import annotations

import json
from typing import Generator

import requests


def submit_event(api_url: str, payload: dict) -> str:
    """POST /api/events 로 이벤트 페이로드 제출.

    Returns:
        run_id.
    """
    resp = requests.post(
        f"{api_url}/api/events",
        json=payload,
        timeout=(5, 30),
    )
    resp.raise_for_status()
    return resp.json()["run_id"]


def stream_events(api_url: str, run_id: str) -> Generator[dict, None, None]:
    """GET /api/agent/stream/{run_id} SSE 스트림 파싱.

    Yields:
        파싱된 SSE 이벤트 dict (event, data fields).
    """
    resp = requests.get(
        f"{api_url}/api/agent/stream/{run_id}",
        stream=True,
        timeout=(5, 300),
    )
    resp.raise_for_status()

    event_type = None
    data_lines: list[str] = []

    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue

        if line == "":
            # 빈 줄 = 이벤트 구분자
            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    parsed = json.loads(raw_data)
                except json.JSONDecodeError:
                    parsed = {"raw": raw_data}
                if event_type:
                    parsed["event"] = event_type
                yield parsed
            event_type = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    # 마지막 이벤트 처리 (빈 줄 없이 스트림 종료된 경우)
    if data_lines:
        raw_data = "\n".join(data_lines)
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError:
            parsed = {"raw": raw_data}
        if event_type:
            parsed["event"] = event_type
        yield parsed
