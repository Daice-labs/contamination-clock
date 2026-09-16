"""RP-T05 paraphrase-detection experiment.
Regenerates the deployed run's model-written paraphrases of the 14 leaked solutions
(the texts were not persisted; the ledger holds their hashes), then tests the standard
13-gram corpus-overlap detector against them. ~14 API calls, then local string matching.
"""
import os, sys, json, time, hashlib, re
import numpy as np

def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()

# ---------- reconstruct the 14 leaked tasks from the shipped ledger (validated method) ----------
from evalplus.data import get_human_eval_plus
DATA = get_human_eval_plus()
rng = np.random.default_rng(7)
ITEMS = [{"id": k, "text": v["prompt"], "solution": v["canonical_solution"],
          "hash": sha(v["prompt"] + v["canonical_solution"])} for k, v in DATA.items()]
rng.shuffle(ITEMS)
hash2task = {t["hash"]: t for t in ITEMS}
events = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "ledger.jsonl"))]
mem_ins = [e for e in events if e["type"] == "memory_insert"]
leak_hashes = [e["provenance"][0] for e in mem_ins[-28:][:14]]
LEAKED = [hash2task[h] for h in leak_hashes]
assert len(LEAKED) == 14, "leaked-task reconstruction failed"
print(f"reconstructed {len(LEAKED)} leaked tasks from ledger ({len(events)} events)")

# ---------- generate model paraphrases (same prompt as the deployed run) ----------
from openai import OpenAI
MODEL = os.environ.get("RPT05_MODEL", "claude-haiku-4-5-20251001")
BASE = os.environ.get("OPENAI_BASE_URL", "https://api.anthropic.com/v1/")
KEY = os.environ.get("OPENAI_API_KEY", "")
if not KEY:
    sys.exit("set OPENAI_API_KEY first (your sk-ant key for the Anthropic endpoint)")
client = OpenAI(base_url=BASE, api_key=KEY, max_retries=2, timeout=120.0)

def extract(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m[0] if m else text

derivs, t0 = [], time.time()
for i, t in enumerate(LEAKED):
    r = client.chat.completions.create(model=MODEL, temperature=0.7, max_tokens=512,
        messages=[{"role": "system", "content": "You are a coding assistant."},
                  {"role": "user", "content":
                   "Rewrite this Python solution with different variable names and structure, same "
                   "behavior. Reply with one ```python block.\n```python\n"
                   + DATA[t["id"]]["canonical_solution"] + "\n```"}])
    code = extract(r.choices[0].message.content or "")
    derivs.append({"task": t["id"], "text": DATA[t["id"]]["prompt"] + code})
    print(f"  [{i+1:2d}/14] {t['id']}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- the detector: 13-gram overlap, exactly the deployed configuration ----------
def ngrams(text, n=13):
    tok = text.split()
    return {" ".join(tok[j:j+n]) for j in range(max(len(tok)-n+1, 1))}

corpus = set()
for d in derivs:
    corpus |= ngrams(d["text"])
per_task = []
for t in LEAKED:
    probe = ngrams(t["text"] + " " + t["solution"])
    hits = probe & corpus
    longest = 0
    tokens = (t["text"] + " " + t["solution"]).split()
    for start in range(len(tokens)):
        for L in range(longest + 1, len(tokens) - start + 1):
            if " ".join(tokens[start:start+L]) in " ".join(d["text"] for d in derivs):
                longest = L
            else:
                break
    per_task.append({"task": t["id"], "flagged": bool(hits),
                     "matching_13grams": len(hits), "longest_shared_run_tokens": longest})
recall = sum(p["flagged"] for p in per_task) / 14
out = {"model": MODEL, "n_derivatives": 14, "detector": "13-gram overlap vs derivative corpus",
       "recall_on_model_paraphrases": recall,
       "median_longest_shared_run": float(np.median([p["longest_shared_run_tokens"] for p in per_task])),
       "per_task": per_task, "derivative_texts": [d["text"] for d in derivs]}
json.dump(out, open("paraphrase_detection_results.json", "w"), indent=1)
print(f"\nRESULT: 13-gram detector recall on model-written paraphrases: {recall:.0%}")
print(f"median longest shared token run: {out['median_longest_shared_run']:.0f}")
print("(simulation's word-scramble derivatives gave 0 percent; this is the realistic version)")
print("wrote paraphrase_detection_results.json  <- send this back")
