# RP-T05 Contamination Clock: Arm M results (real model, claude-haiku-4-5)

First real-model confirmation cell, run on the Anthropic OpenAI-compat endpoint. Clean run: 649
calls, zero rate limits, zero API errors, 10 scorer timeouts (1.5 percent, notably below
gpt-4o-mini's 4 to 6 percent), all six arms checkpointed. Model recorded:
claude-haiku-4-5-20251001. Hard-subset targets (36 holdout, 24 sealed, dose 14) drawn from the
ledger-clean pool.

## One correction applied in analysis, disclosed
The bundle computed its floor from replicate resampling only; at temperature 0.2 Haiku is
per-task deterministic (measured replicate variance 0.0000), so that floor collapsed to zero and
one verdict printed as CHECK. The correct floor at this design is task-level bootstrap:
**+/-0.197** at n=36/24. All verdicts below use it. (Bundle fix queued: task-level bootstrap, and
R is wasted at temp 0.2; future cells should spend budget on tasks, not replicates.)

## Findings

**The memory channel is confirmed by concentration, not by the aggregate.**
Aggregate DiD inflation +0.167 sits under the +/-0.197 task-level floor, exactly as the
calculator predicts for n=36: small holdouts cannot certify aggregate inflation of this size.
The per-item paired evidence is decisive instead: **4 of 14 leaked items improved, 0 of 22
unleaked items changed (Fisher exact p = 0.017)**, and of the 6 leaked items failing at baseline,
retrieval converted **67 percent to passing**, with zero spillover. One retrieval hit of the
exact solution flips a failing task; nothing else moves.

**Laundering persists at the item level.** After the model paraphrased the leaked solutions and
originals were scrubbed, 3 of the 4 exact-leak conversions survived (item persistence 75 percent;
aggregate ratio 0.75, matching the simulation's injected-truth band). Model-written derivatives
retain most of the contaminating utility.

**The winner's curse replicates on a real model.** Selection over 14 prompt variants measured on
5 tasks: reported gain +0.200; persistent gain on fresh evaluation -0.040. The reported number is
almost entirely curse, consistent with the Arm S law.

**Exchangeability holds.** Baseline gap -0.153 is within the task-level floor (M4 resolves PASS
with the corrected floor; the DiD estimator subtracts this gap by construction either way).

## Verdicts (corrected floor)
- M1 aggregate inflation vs floor: **BELOW FLOOR at n=36 (as budgeted); concentration carries it**
- M1b concentration on leaked items: **PASS (Fisher p=0.017)**
- M2 laundering persistence: **PASS (3/4 item-level, 75 percent)**
- M3 reported > persistent under selection: **PASS**
- M4 exchangeability: **PASS (corrected floor)**

## What this buys the paper and the TTCL short
Arm S validated the instrument against injected truth; Arm M now shows the same signatures on a
real frozen model with a real retrieval store: concentrated item-level inflation, laundering
persistence, and the curse. The certification-budget message is itself demonstrated by the data:
the aggregate at n=36 is honestly unresolvable while item-level pairing resolves cleanly, which
is the calculator's argument in one table. Remaining for the full paper: Arm T (LoRA, parametric
channel) and a larger-n Arm M cell if an aggregate certification is wanted.
