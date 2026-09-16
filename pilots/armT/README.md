# RP-T05 Arm T: the parametric channel on real weights (LoRA)

Fine-tunes LoRA adapters on the leaked solutions and measures inflation with the same
sealed-slice instrument as the rest of the study. Stratified randomized leakage (baseline-outcome
balanced), a zero-leak training control, 1-epoch and 3-epoch doses, per-item paired analysis,
randomization inference, and a fresh ledger chain seeded from the bundled study ledger so Arm T
never evaluates on previously contaminated tasks.

## RunPod (24GB GPU, under an hour, roughly a dollar)
1. Launch any PyTorch 2.x template with a 24GB card (RTX 3090, 4090, or A5000 class).
2. Upload this folder (or git clone your copy), then:

       cd rpt05-armT-run
       open RP-T05-armT.ipynb in Jupyter and Run All (or: bash run_pod.sh)

3. Default model: deepseek-ai/deepseek-coder-1.3b-base (public, no token). To use another:
   set RPT05_T_MODEL to any causal LM id before running.
4. Send back the whole `rpt05_armT_results/` directory (armT_summary.json is the core).

Interruptions are cheap: adapters and eval files checkpoint per arm; rerun the same command to
resume. Scorer timeouts are counted and capped at 10 seconds per candidate.

## What the verdicts mean
T1: leaked-task uplift exceeds unleaked (the parametric channel is real on weights).
T2: three epochs move leaked tasks at least as much as one (dose response).
T3: the control adapter, trained on clean tasks, produces no comparable inflation
(the estimator does not mistake fine-tuning itself for contamination).
This is the experiment that turns the paper's simulation-calibrated potency ratio into a
real-weights comparison against the deployed memory result.

## Verified before shipping
The complete pipeline (pool selection from the ledger, stratified leakage, LoRA training,
generation, scoring, checkpoint resume, analysis, export) was executed end to end on CPU with a
tiny random-initialized model, all three verdicts behaving as constructed. The pretrained path is
identical code with the model swapped.

## Verification status
This bundle passed end-to-end CPU verification in tiny mode (RPT05_T_TINY=1): a locally built
tokenizer plus random-init 2-layer model memorizes the leaked solutions verbatim, the EvalPlus
scorer passes them, all three verdicts return PASS with real (non-vacuous) signal, and the
resume path was exercised. The GPU path shares every line of that pipeline except model loading
and LoRA attachment.
