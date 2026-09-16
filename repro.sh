#!/bin/sh
# One-command reproduction of the simulation arm (Arm S), then verification.
# The simulation is free (no API calls) and deterministic under the recorded seeds.
# Usage: ./repro.sh [results_dir]   (default: ./results_repro)
set -e
OUT="${1:-$PWD/results_repro}"
python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate
pip -q install -r code/requirements.txt
RPT05_RESULTS="$OUT" jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output executed.ipynb code/RP-T05-contamination-clock-study.ipynb
cp expected_outputs.json verify_results.py "$OUT/.." 2>/dev/null || true
echo "Run complete. Verifying shipped records:"
python3 verify_results.py
echo "To verify the fresh run instead, point BASE in verify_results.py at $OUT."
