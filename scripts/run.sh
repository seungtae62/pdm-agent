#!/usr/bin/env bash
# PdM Agent 실행 스크립트
# Usage: ./scripts/run.sh [agent|ui|all]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source venv/bin/activate
fi

# Load .env
set -a
source .env
set +a

# src 내부 모듈이 api.*, agent.* 등 상대경로로 import하므로 PYTHONPATH에 추가
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
    agent)
        echo "Starting PdM Agent API server on :8000 ..."
        exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    ui)
        echo "Starting Streamlit UI on :8501 ..."
        exec streamlit run ui/app.py --server.port 8501
        ;;
    all)
        echo "Starting Agent API (:8000) and Streamlit UI (:8501) ..."
        trap 'kill 0' EXIT
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
        streamlit run ui/app.py --server.port 8501 &
        wait
        ;;
    *)
        echo "Usage: $0 {agent|ui|all}"
        exit 1
        ;;
esac