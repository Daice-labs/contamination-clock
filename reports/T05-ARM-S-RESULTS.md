# RP-T05 Contamination Clock: Arm S results (simulation, constructed ground truth)

Full pipeline executed end to end: 16-cell notebook, all experiments, zero errors, 451 hash-chained
ledger events, all artifacts exported. **Gate G0: PASS. All six simulation-scoped claims: PASS.**

## Scope, stated plainly
Every effect here is **injected with known magnitude** into a simulated learner over real HumanEval+
task text. These results validate the instrument and exhibit the mechanisms against constructed
truth. They are the Gate G0 deliverable and the core of a TTCL workshop short. They are **not**
real-model findings: Arm M (memory agent, your API key) and Arm T (LoRA, one GPU) carry those.

## What was established

**G0, the instrument.** Sealed-slice differencing (difference-in-differences against the measured
pre-exposure gap) recovers injected inflation within intervals on every channel (e.g. C-M exact:
truth +0.0656, estimate +0.0737; C-P exact: +0.0422 vs +0.0449). The zero-leak control reads -0.0007
against a +/-0.048 floor. The legitimate-training control is the sharpest validation: raw holdout
score rises 8 points from genuine generalisation while measured inflation reads +0.0005: the
estimator counts contamination, not improvement. The ledger's smuggle test flags a planted
sealed-slice item loudly and the clean slice audits sealed.

**S-H1 channel potency.** At matched exposure (one event per item), the memory channel inflates
2.79x the parametric channel per exact leaked item. One retrieval hit is worth roughly three epochs.

**S-H2 proximity.** Monotone decay exact > paraphrase > solution-only > adjacent on both channels,
with adjacent-only data indistinguishable from zero. Dose-response curves in inflation_curves.csv.

**S-H3 the selection speed limit.** In Proposition 2's stated regime (equal means), measured
selection inflation matches the **exact** expected-maximum law within 20 percent at every k >= 10.
Bring-up caught a spec error worth keeping: the asymptotic sqrt(2 ln k) band implies a fitted
exponent of ~0.5, but the exact Gaussian maximum over k in [3, 3000] grows with local exponent
~0.73; the pre-registered test was recalibrated to the exact law, which is stronger. In the
realistic-loop arm, reported inflation reaches +0.22 by k=3000 while persistent inflation (what
survives fresh evaluation) stays near zero: the winner's curse is almost the entire reported number.

**S-H4 detector asymmetry, on real text.** A 13-gram overlap detector at deployed settings (training
corpus only): recall 100% on parametric-exact, **0% on memory, selection, and laundered channels**.
The constructive counterpart: an auditor told to inspect the memory store recovers 100%. The
asymmetry is structural: detectors look where parametric contamination lives.

**S-H5 laundering.** After scrubbing originals, 74% of inflation persists through paraphrase
derivatives (injected truth 65%, within tolerance). Provenance closure caught 100% of derivatives at
the configured lineage-retention rate; coverage is measured, not assumed.

**S-H6 the discipline.** Undisciplined reporting (best score on the selected-against holdout):
+0.166 of pure winner's curse at k=1000. Disciplined certification from sealed slices: -0.012,
at the floor, at 10 certifications per 1000 queries. Rotation resets holdout overfit by construction.

**The calculator's brutal arithmetic, now validated:** at n=50, k_max(delta=0.05) is about 1.3:
a holdout this size supports roughly one adaptive decision per sealed slice at that tolerance.

## Artifacts
`RP-T05-contamination-clock-study.ipynb` (source), `RP-T05-executed.ipynb` (with outputs),
`rpt05_results/`: summary.json, inflation_curves.csv, selection_law.csv, detection_recall.csv,
laundering.json, discipline.json, prereg.json (hashed), ledger.jsonl (451 events),
adjudication_sim.csv, figures/figA_clock.png.

## What Arms M and T will need from you (future runs)
Arm M: your OpenAI or Anthropic key, ~1-2k calls (a frozen model with a real retrieval store,
channels injected into the store). Arm T: one 24GB GPU for LoRA on a small open-weight coder.
The notebook's channel and ledger machinery is arm-agnostic; the estimator is now validated.
