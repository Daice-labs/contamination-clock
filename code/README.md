# RP-T05 Arm M: the real-model memory agent

Runs the Contamination Clock's Arm M against a real hosted model: a frozen model with a retrieval
store, leakage injected into the store, every insertion hash-chained on the update ledger.
**~850 calls, well under $0.50, roughly 25 to 45 minutes.** The Arm S simulation (already fully
validated) reruns first in ~2 minutes as a free integrity check: expect `GATE G0: PASS` before any
API call.

## Commands

```bash
unzip rpt05-run.zip && cd rpt05-run
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
caffeinate -i ./run.sh sk-YOUR-KEY
```

Reusing the T03/T02A venv also works: run from this folder with that venv active.

**On the Anthropic key** (e.g. while T02A occupies OpenAI):

```bash
OPENAI_BASE_URL=https://api.anthropic.com/v1/ RPT05_MODEL=<your-dated-claude-model> \
caffeinate -i ./run.sh sk-ant-YOUR-KEY
```

Uses Anthropic's OpenAI-compatible endpoint. If the first arm errors immediately, the compat layer
is the suspect: fall back to OpenAI in the morning after T02A finishes.

Monitor: `python progress_meter.py ./rpt05_results --watch` from a second terminal.

## What it runs
Hard-subset targets (headroom lesson from T03), drawn from the ledger-clean pool: in bring-up the
sealed-slice audit refused targets the simulation had already leaked, which is the ledger doing its
job. Arms: clean baselines on H and the sealed slice (exchangeability + floor), C-M exact-leak dose,
C-G laundering (the model paraphrases leaked solutions, originals scrubbed), C-S selection
(winner's curse: reported vs fresh-eval persistent). Per-item paired uplift on leaked vs unleaked
items is the primary evidence channel: it has far more power than the aggregate at this scale.

## Interruptions
Every arm checkpoints (`arms/*.json`, config-signature guarded). Rerun the same command to resume;
verified: all six arms resume and the run completes.

## Knobs (env vars)
`RPT05_API_NH=36 RPT05_API_NF=24 RPT05_API_R=3 RPT05_API_DOSE=14` are the defaults. Halving NH/NF
halves cost and widens the floor.

## Send back
The whole results dir, or minimally: `armM_summary.json`, `adjudication_armM.csv`,
`progress_log.jsonl` (plus `summary.json` from the sim track).

## Honest caveats
One model, one benchmark family, dose at one level: this is the first real-model confirmation cell,
not the full Arm M program. The aggregate inflation may sit near the floor at these sizes; the
per-item concentration profile is the load-bearing readout. Scorer timeouts are counted and capped
at 10s per candidate (non-halting code scores 0).
