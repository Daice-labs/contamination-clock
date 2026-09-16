# RP-T05 Arm T Results: The Parametric Channel on Real Weights

**Run:** deepseek-coder-1.3b-base, LoRA r16, RunPod GPU, 2026-07-25. Verified: ledger chain intact
(42 events), trained-task hashes match independent reconstruction exactly for leak and control
arms, every summary statistic and randomization p reproduced to machine precision from raw arrays.

## Design
60 hardest ledger-clean tasks, 36/24 split (seed 7+1700). Leakage randomized WITH stratification
on baseline outcome (12 failing + 2 passing of 14, mirroring the 30/36 holdout fail rate; the
improvement motivated by the memory arm's chance imbalance). Zero-leak control: 14 clean tasks
outside H and F. Doses: 1 and 3 epochs. Baseline H 0.167, F 0.125. Scorer timeouts: 7 (capped).

## Results
| Arm | DiD | Leaked uplift | Unleaked uplift | rand p |
|---|---|---|---|---|
| leak e1 | +0.056 | +0.071 | +0.045 | 0.42 |
| leak e3 | -0.028 | +0.286 | -0.091 | 0.66 |
| control e3 | +0.097 | +0.214 | +0.091 | 0.28 |

Verdicts: T1 PASS (e3 leaked > unleaked), T2 PASS (dose monotone), **T3 FAIL** (control |DiD|
exceeds leak |DiD|).

## The headline: T3's failure is the finding
Fine-tuning a small base model on ANY code moves scores broadly and unevenly across the realized
partition: the zero-leak control produced a larger aggregate |DiD| (+0.097) than the leak arm
(-0.028). This is the single-partition false-certification mode the paper's misspecification
stress test prices (15-19% tail), realized on real weights. The aggregate estimator is not
interpretable in this regime; the layered analyses survive:

- **Concentration (headroom-balanced by stratification):** failing-restricted conversions 5/12
  leaked vs 2/18 unleaked, Fisher p = 0.068 (exploratory). Full-sample 5/14 vs 2/22, p = 0.064.
- **Placebo-adjusted concentration:** differencing the control arm's per-task response out of the
  leak arm's leaves +0.253 leaked-vs-unleaked (permutation p = 0.13; raw concentrations +0.377
  leak vs +0.123 placebo).
- **Dose:** leaked uplift 0.071 -> 0.286 from 1 to 3 epochs, monotone.

## Potency vs the memory channel (matched dose 14)
Memory (frozen haiku, single insertion): leaked uplift +0.286. Parametric single exposure
(1 epoch): +0.071. Ratio ~4x memory:parametric, directionally consistent with the simulator's
configured asymmetry (measured 2.79 / injected 3.6). Cross-model, exploratory. Parametric reaches
+0.286 only at 3 epochs.

## Assumption (iv) contrast
No shared store -> no retrieval interference by design; sealed-slice movement (F +0.083 at e3) is
genuine generalization, the g-term, not iota. The two deployed experiments thus exhibit the two
distinct threats the identification section names: interference (memory arm) and heterogeneous
legitimate response (parametric arm).

## Paper integration (done)
Design paragraph in Section 4.2; results block + Table 5 in Section 6; potency pointer in 5.2;
Limitations rescoped ("simulation-only" retired); abstract and conclusion clauses. All numbers in
the verification canon.
