"""RP-T05 Arm T: the parametric channel on real weights (LoRA fine-tuning).
Modes: default = pretrained model on GPU (RunPod); RPT05_T_TINY=1 = CPU mechanism verification
with a locally built tokenizer and a random-init 2-layer model (no downloads).
Design: stratified randomized leakage (the review-driven improvement), zero-leak training control,
1-epoch and 3-epoch dose, per-item paired analysis, randomization inference, fresh ledger chain.
"""
import os, sys, json, time, hashlib, re, math, random
import numpy as np

TINY = os.environ.get("RPT05_T_TINY") == "1"
OUT = os.path.abspath(os.environ.get("RPT05_T_RESULTS", "./rpt05_armT_results"))
os.makedirs(os.path.join(OUT, "adapters"), exist_ok=True)
SEED = 7
def sha(o): return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()

# ---------------- task universe + ledger-clean pool (sim AND deployed-memory contamination excluded) ----------------
from evalplus.data import get_human_eval_plus
DATA = get_human_eval_plus()
rng = np.random.default_rng(SEED); random.seed(SEED)
ITEMS = [{"id": k, "text": v["prompt"], "solution": v["canonical_solution"],
          "hash": sha(v["prompt"] + v["canonical_solution"])} for k, v in DATA.items()]
rng.shuffle(ITEMS)
events = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl"))]
closure = set()
for e in events:
    closure.add(e["payload"]); closure.update(e["provenance"])
clean = [t for t in ITEMS if t["hash"] not in closure]
hard = sorted(clean, key=lambda t: (len(t["text"]) + len(t["solution"]) if TINY else -len(t["solution"])))
if TINY:
    N_H, N_F, DOSE = 8, 6, 4
else:
    N_H, N_F, DOSE = int(os.environ.get("RPT05_T_NH", "36")), int(os.environ.get("RPT05_T_NF", "24")), 14
sel = hard[: N_H + N_F]
hr = np.random.default_rng(SEED + 1700); hr.shuffle(sel)
H_T, F_T = sel[:N_H], sel[N_H:]
CONTROL_POOL = hard[N_H + N_F : N_H + N_F + DOSE]          # clean tasks outside H and F
print(f"clean pool {len(clean)} | H {len(H_T)} F {len(F_T)} | dose {DOSE} | control pool {len(CONTROL_POOL)}")

LEDGER_PATH = os.path.join(OUT, "armT_ledger.jsonl")
_prev = ["genesis-armT"]
def ledger(etype, payload_hash, prov=()):
    ev = {"type": etype, "payload": payload_hash, "provenance": list(prov), "prev": _prev[0], "t": time.time()}
    ev["hash"] = sha(ev); _prev[0] = ev["hash"]
    with open(LEDGER_PATH, "a") as f: f.write(json.dumps(ev) + "\n")

# ---------------- model + tokenizer ----------------
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEV == "cuda" else torch.float32
print(f"device {DEV} dtype {DTYPE} | tiny={TINY}")

if TINY:
    from tokenizers import ByteLevelBPETokenizer
    tokp = os.path.join(OUT, "tiny_tok")
    os.makedirs(tokp, exist_ok=True)
    corpus_file = os.path.join(OUT, "corpus.txt")
    with open(corpus_file, "w") as f:
        for t in ITEMS: f.write(t["text"] + "\n" + t["solution"] + "\n")
    tk = ByteLevelBPETokenizer()
    tk.train([corpus_file], vocab_size=800, min_frequency=1, special_tokens=["<pad>", "<eos>"])
    EOS_ID = 1
    def enc(s): return tk.encode(s).ids
    def dec(ids): return tk.decode([i for i in ids if i != EOS_ID])
    VOCAB = tk.get_vocab_size()
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=768, n_embd=128, n_layer=2, n_head=4,
                     bos_token_id=1, eos_token_id=1)
    def fresh_model(): return GPT2LMHeadModel(cfg).to(DEV)
    MAXNEW, EPOCH_SCALE, LR = 220, 80, 1e-3
    MODEL_NAME = "tiny-random-gpt2(2L,128d)"
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    MODEL_NAME = os.environ.get("RPT05_T_MODEL", "deepseek-ai/deepseek-coder-1.3b-base")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    EOS_ID = tok.eos_token_id
    def enc(s): return tok(s, return_tensors=None)["input_ids"]
    def dec(ids): return tok.decode(ids, skip_special_tokens=True)
    def fresh_model():
        return AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEV)
    MAXNEW, EPOCH_SCALE, LR = 384, 1, 2e-4
print(f"model {MODEL_NAME}")

from peft import LoraConfig, get_peft_model
def lora_wrap(m):
    if TINY:
        # random-init base: adapters alone cannot memorize, so tiny mode widens LoRA and fully
        # trains the embeddings (modules_to_save); the peft attach/save/load path stays exercised
        lc = LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0.0,
                        target_modules=["c_attn", "c_fc", "c_proj"],
                        modules_to_save=["wte", "wpe"])
    else:
        lc = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(m, lc)

def train_lora(train_tasks, epochs, tag):
    """Fine-tune on prompt+canonical (the C-P exact channel). LoRA when pretrained; full FT in
    tiny verification mode (adapters cannot steer a random-init base). Returns adapter dir."""
    adir = os.path.join(OUT, "adapters", tag)
    if os.path.exists(os.path.join(adir, "done.json")):
        print(f"  [{tag}] RESUMED (adapter exists)"); return adir
    m = fresh_model() if TINY else lora_wrap(fresh_model())
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
    # encode prompt and solution SEPARATELY, then concatenate: generation encodes the prompt alone,
    # and BPE merges across the prompt/solution junction would otherwise make the eval-time prefix
    # differ from the memorized prefix (the failure mode this fixes)
    seqs = [torch.tensor((enc(DATA[t["id"]]["prompt"]) + enc(DATA[t["id"]]["canonical_solution"]))[:(700 if TINY else 1000)]
                         + ([EOS_ID] if EOS_ID is not None else []),
                         device=DEV).unsqueeze(0) for t in train_tasks]
    for t in train_tasks:
        ledger("train_batch", t["hash"], [t["hash"]])
    E = epochs * EPOCH_SCALE
    t0, last = time.time(), None
    for ep in range(E):
        random.shuffle(seqs)
        for s in seqs:
            out = m(input_ids=s, labels=s)
            out.loss.backward(); opt.step(); opt.zero_grad()
            last = float(out.loss.detach())
        if ep % max(E // 5, 1) == 0:
            print(f"  [{tag}] epoch {ep+1}/{E} loss {last:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if TINY:
        os.makedirs(adir, exist_ok=True)
        torch.save(m.state_dict(), os.path.join(adir, "full_ft.pt"))
    else:
        m.save_pretrained(adir)
    json.dump({"loss_final": last, "epochs": E}, open(os.path.join(adir, "done.json"), "w"))
    print(f"  [{tag}] trained: final loss {last:.3f}")
    del m
    if DEV == "cuda": torch.cuda.empty_cache()
    return adir

# ---------------- hardened scorer (EvalPlus, execution-timeout capped) ----------------
import signal as _sg
from contextlib import contextmanager as _cm
class _CT(BaseException): pass
@_cm
def _tl(sec):
    if not hasattr(_sg, "SIGALRM"): yield; return
    old = _sg.signal(_sg.SIGALRM, lambda s, f: (_ for _ in ()).throw(_CT()))
    _sg.setitimer(_sg.ITIMER_REAL, sec)
    try: yield
    finally: _sg.setitimer(_sg.ITIMER_REAL, 0); _sg.signal(_sg.SIGALRM, old)
_GC, TIMEOUTS = {}, {"gold": 0, "cand": 0}
def _gold(tid):
    if tid in _GC: return _GC[tid]
    ep = DATA[tid]; ns = {"__name__": "__g__"}; cases = []
    try: exec(ep["prompt"] + ep["canonical_solution"], ns)
    except Exception: _GC[tid] = []; return []
    g = ns.get(ep["entry_point"])
    if g:
        import copy as _cp
        inp = (list(ep.get("base_input") or []) + list(ep.get("plus_input") or []))[:80]
        try:
            with _tl(30.0):
                for a in inp:
                    try: cases.append((a, g(*_cp.deepcopy(a))))
                    except Exception: continue
        except _CT: TIMEOUTS["gold"] += 1
    _GC[tid] = cases; return cases
def score(code, tid):
    import copy as _cp
    ep = DATA[tid]; pre = ep["prompt"]
    try:
        with _tl(10.0):
            ns = {"__name__": "__c__"}; exec(pre + code, ns)
    except _CT: TIMEOUTS["cand"] += 1; return 0
    except Exception: return 0
    fn = ns.get(ep["entry_point"])
    if fn is None: return 0
    cases = _gold(tid)
    if not cases: return 0
    def eq(a, b):
        try:
            if isinstance(a, float) or isinstance(b, float): return abs(a - b) < 1e-6
            return a == b
        except Exception: return False
    try:
        with _tl(10.0):
            for a, w in cases:
                try: got = fn(*_cp.deepcopy(a))
                except _CT: raise
                except Exception: return 0
                if not eq(got, w): return 0
    except _CT: TIMEOUTS["cand"] += 1; return 0
    return 1

STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert"]
CTX_CAP = 768 if TINY else 1024
def gen_completion(model, tid):
    prompt = DATA[tid]["prompt"]
    keep = CTX_CAP - MAXNEW - 8
    ids = torch.tensor(enc(prompt)[-keep:], device=DEV).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False,
                             eos_token_id=EOS_ID,
                             pad_token_id=0 if TINY else (tok.pad_token_id or tok.eos_token_id))
    text = dec(out[0][ids.shape[1]:].tolist())
    cut = len(text)
    for s in STOPS:
        i = text.find(s)
        if i > 0: cut = min(cut, i)
    return text[:cut]

def eval_state(model, tag):
    p = os.path.join(OUT, f"eval_{tag}.json")
    if os.path.exists(p):
        d = json.load(open(p)); print(f"  [eval {tag}] RESUMED")
        return np.array(d["H"]), np.array(d["F"])
    model.eval()
    t0 = time.time()
    Hs = [score(gen_completion(model, t["id"]), t["id"]) for t in H_T]
    Fs = [score(gen_completion(model, t["id"]), t["id"]) for t in F_T]
    json.dump({"H": Hs, "F": Fs}, open(p, "w"))
    print(f"  [eval {tag}] H {np.mean(Hs):.3f} F {np.mean(Fs):.3f} "
          f"({time.time()-t0:.0f}s, timeouts {TIMEOUTS['cand']})", flush=True)
    return np.array(Hs), np.array(Fs)

# ---------------- run ----------------
print("\n== T0 baseline ==")
base_model = fresh_model()
h0, f0 = eval_state(base_model, "t0_base")
gap0 = h0.mean() - f0.mean()
del base_model
if DEV == "cuda": torch.cuda.empty_cache()

# stratified randomized leakage on baseline outcome (the review-driven design improvement)
sr = np.random.default_rng(SEED + 41)
fail_idx = [i for i in range(N_H) if h0[i] == 0]; pass_idx = [i for i in range(N_H) if h0[i] == 1]
n_fail = round(DOSE * len(fail_idx) / N_H)
leak_idx = sorted(list(sr.choice(fail_idx, size=min(n_fail, len(fail_idx)), replace=False)) +
                  list(sr.choice(pass_idx, size=DOSE - min(n_fail, len(fail_idx)), replace=False)))
LEAKED = [H_T[i] for i in leak_idx]
unleak_idx = [i for i in range(N_H) if i not in leak_idx]
print(f"stratified leak: {len(leak_idx)} tasks ({sum(h0[i]==0 for i in leak_idx)} failing, "
      f"{sum(h0[i]==1 for i in leak_idx)} passing) mirroring baseline fail rate {len(fail_idx)}/{N_H}")

results = {}
for tag, tasks, ep in [("t1_leak_e1", LEAKED, 1), ("t1_leak_e3", LEAKED, 3),
                        ("tc_control_e3", CONTROL_POOL, 3)]:
    print(f"\n== {tag} ==")
    adir = train_lora(tasks, ep, tag)
    if TINY:
        m = fresh_model()
        m.load_state_dict(torch.load(os.path.join(adir, "full_ft.pt"), map_location=DEV))
        base = None
    else:
        from peft import PeftModel
        base = fresh_model()
        m = PeftModel.from_pretrained(base, adir).to(DEV)
    h1, f1 = eval_state(m, tag)
    did = (h1.mean() - f1.mean()) - gap0
    dH = h1 - h0
    results[tag] = {"H": float(h1.mean()), "F": float(f1.mean()), "DiD": float(did),
                    "uplift_leaked": float(dH[leak_idx].mean()),
                    "uplift_unleaked": float(dH[unleak_idx].mean()),
                    "dH": dH.tolist(), "dF": (f1 - f0).tolist()}
    print(f"  DiD {did:+.3f} | leaked uplift {dH[leak_idx].mean():+.3f} vs unleaked {dH[unleak_idx].mean():+.3f}")
    del m
    if base is not None: del base
    if DEV == "cuda": torch.cuda.empty_cache()

# ---------------- analysis: randomization p, bootstrap CI, verdicts ----------------
def rand_p(dH, dF, reps=20000):
    allD = np.concatenate([dH, dF]); obs = dH.mean() - dF.mean()
    r = np.random.default_rng(3); cnt = 0
    for _ in range(reps):
        pm = r.permutation(len(allD))
        if allD[pm[:len(dH)]].mean() - allD[pm[len(dH):]].mean() >= obs: cnt += 1
    return cnt / reps
summary = {"mode": "tiny-verification" if TINY else "pretrained", "model": MODEL_NAME,
           "config": {"NH": N_H, "NF": N_F, "dose": DOSE, "seed": SEED},
           "baseline": {"H": float(h0.mean()), "F": float(f0.mean()), "gap": float(gap0)},
           "leak_stratification": {"failing": int(sum(h0[i]==0 for i in leak_idx)),
                                    "passing": int(sum(h0[i]==1 for i in leak_idx))},
           "arms": results, "scorer_timeouts": TIMEOUTS}
for tag in ("t1_leak_e1", "t1_leak_e3", "tc_control_e3"):
    dH = np.array(results[tag]["dH"]); dF = np.array(results[tag]["dF"])
    summary["arms"][tag]["randomization_p"] = rand_p(dH, dF)
summary["verdicts"] = {
  "T1 parametric inflation present (e3 leaked uplift > unleaked)":
      results["t1_leak_e3"]["uplift_leaked"] > results["t1_leak_e3"]["uplift_unleaked"],
  "T2 dose monotone (e3 >= e1 leaked uplift)":
      results["t1_leak_e3"]["uplift_leaked"] >= results["t1_leak_e1"]["uplift_leaked"] - 1e-9,
  "T3 control near zero (|DiD| < leak e3 DiD)":
      abs(results["tc_control_e3"]["DiD"]) < abs(results["t1_leak_e3"]["DiD"]) + 1e-9,
}
json.dump(summary, open(os.path.join(OUT, "armT_summary.json"), "w"), indent=1)
print("\n== verdicts ==")
for k, v in summary["verdicts"].items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"\nwrote {OUT}/armT_summary.json  <- send the whole results dir back")
