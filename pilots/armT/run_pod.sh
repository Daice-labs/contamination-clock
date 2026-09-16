#!/usr/bin/env bash
set -uo pipefail
python3 -m pip install -q -r requirements-gpu.txt
export RPT05_T_MODEL="${RPT05_T_MODEL:-deepseek-ai/deepseek-coder-1.3b-base}"
python3 -u RP-T05-armT.py 2>&1 | tee armT_run.log
