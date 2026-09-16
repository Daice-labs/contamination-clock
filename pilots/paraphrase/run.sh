#!/usr/bin/env bash
set -uo pipefail
KEY="${1:-${OPENAI_API_KEY:-}}"
[ -z "$KEY" ] && { echo "usage: ./run.sh sk-ant-YOUR-KEY"; exit 1; }
export OPENAI_API_KEY="$KEY"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.anthropic.com/v1/}"
export RPT05_MODEL="${RPT05_MODEL:-claude-haiku-4-5-20251001}"
PY="${PYTHON:-python3}"
"$PY" -c "from evalplus.data import get_human_eval_plus; import openai, numpy" || { echo ""; echo "dependency import failed (traceback above). Try: pip install evalplus openai numpy; if datasets errors: pip install \"datasets<4\""; exit 1; }
"$PY" paraphrase_detect.py
