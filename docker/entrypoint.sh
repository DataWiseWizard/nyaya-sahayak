#!/bin/bash
set -euo pipefail

MODE="${NYAYA_MODE:-streamlit}"

case "$MODE" in
    streamlit)
    exec streamlit run app.py \
        --server.address=0.0.0.0 \
        --server.port=8501 \
        --server.headless=true \
        --browser.gatherUsageStats=false
    ;;
    api)
    exec uvicorn backend.api:app --host 0.0.0.0 --port 8000
    ;;
  *)
    echo "ERROR: Unknown NYAYA_MODE='$MODE' (expected 'streamlit' or 'api')" >&2
    exit 1
    ;;
esac