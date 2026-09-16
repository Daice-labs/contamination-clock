#!/usr/bin/env bash
# T05 Arm M launcher.  Usage:  ./run.sh sk-YOUR-KEY [results_dir]
# Anthropic:  OPENAI_BASE_URL=https://api.anthropic.com/v1/ RPT05_MODEL=<dated-claude-model> ./run.sh sk-ant-KEY
set -uo pipefail
find_python() {
  for cand in ${PYTHON:-} python3 python /usr/bin/python3; do
    [ -z "$cand" ] && continue
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" - <<'PYEOF' >/dev/null 2>&1
import importlib
for m in ["nbconvert","nbformat","numpy","pandas","scipy","matplotlib","openai","evalplus"]:
    importlib.import_module(m)
PYEOF
    then echo "$cand"; return 0; fi
  done
  return 1
}
PY="$(find_python)" || {
  echo "ERROR: no Python here has the required packages."
  echo "Fix:  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
}
KEY="${1:-${OPENAI_API_KEY:-}}"
[ -z "$KEY" ] && { echo "usage: ./run.sh sk-YOUR-KEY [results_dir]"; exit 1; }
RESULTS="${2:-$(pwd)/rpt05_results}"
export RPT05_RESULTS="$RESULTS" RPT05_RUN_MODE=api RPT05_CONFIRM=1
export OPENAI_API_KEY="$KEY"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export RPT05_MODEL="${RPT05_MODEL:-gpt-4o-mini-2024-07-18}"
export MPLBACKEND=Agg
mkdir -p "$RESULTS"
echo "python    -> $PY ($($PY -V 2>&1))"
echo "model     -> $RPT05_MODEL  (via $OPENAI_BASE_URL)"
echo "results   -> $RESULTS"
echo "expect    -> ~850 calls, well under \$0.50, roughly 25 to 45 min"
echo "note      -> the sim track (Arm S validation) runs first, ~2 min, no API calls"
echo "monitor   -> $PY progress_meter.py \"$RESULTS\" --watch"
echo
"$PY" -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
  --output run_done.ipynb RP-T05-contamination-clock-study.ipynb 2>&1 | tee run.log
STATUS=${PIPESTATUS[0]}
echo
if [ "$STATUS" -eq 0 ]; then
  echo "DONE. Arm M outputs:"; ls -1 "$RESULTS" | sed 's/^/   /'
else
  echo "exited $STATUS. Completed arms are checkpointed: rerun the same command to resume."
fi
