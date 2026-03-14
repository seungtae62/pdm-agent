#!/bin/bash
# POST 이벤트 → run_id 추출 → SSE 스트림 자동 구독
# 사용법: bash scripts/test_event.sh [payload_file] [port]

PAYLOAD=${1:-data/payloads/scenario1_day31.json}
PORT=${2:-8000}
BASE="http://localhost:${PORT}"

echo ">>> POST ${BASE}/api/events (payload: ${PAYLOAD})"
RESPONSE=$(curl -s -X POST "${BASE}/api/events" \
  -H "Content-Type: application/json" \
  -d @"${PAYLOAD}")

echo "${RESPONSE}"

RUN_ID=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])" 2>/dev/null)

if [ -z "${RUN_ID}" ]; then
  echo "ERROR: run_id 추출 실패"
  exit 1
fi

echo ""
echo ">>> GET ${BASE}/api/agent/stream/${RUN_ID}"
echo "--- SSE 스트림 시작 ---"
curl -N -s "${BASE}/api/agent/stream/${RUN_ID}"
echo ""
echo "--- SSE 스트림 종료 ---"
