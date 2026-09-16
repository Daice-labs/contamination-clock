#!/usr/bin/env python3
"""Verify the shipped T05 records against the expected-output manifest.
Checks (1) headline statistics read from the record files, (2) internal
consistency of the adjudication table, and (3) ledger hash-chain linkage.
Exit code 0 = all checks pass."""
import json, csv, sys

BASE = "results"
exp = json.load(open("expected_outputs.json"))
fail = []
def check(name, got, want, tol=1e-6):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(("  OK   " if ok else "  FAIL ") + f"{name}: {got} (expected {want})")
    if not ok: fail.append(name)

s = json.load(open(f"{BASE}/summary.json"))
check("prereg_hash_prefix", s["prereg_hash"][:16], exp["prereg_hash_prefix"])
check("gate_G0_verdict", s["gate_G0"]["verdict"], "PASS")
check("noise_floor", float(s["floor"]), exp["noise_floor"], 1e-4)
check("SH1_channel_ratio", float(s["SH1_ratio"]), exp["SH1_channel_ratio"], 0.01)
check("undisciplined_reported_inflation", float(s["discipline"]["undisciplined_reported_inflation"]), exp["undisciplined_reported_inflation"], 1e-4)
check("laundering_persistence", float(s["laundering"]["persistence"]), exp["laundering_persistence"], 1e-3)

rows = list(csv.DictReader(open(f"{BASE}/adjudication_sim.csv")))
check("adjudication_rows", len(rows), exp["adjudication_rows"])
check("adjudication_failures", sum(r["verdict"] == "FAIL" for r in rows), 0)

sel = list(csv.DictReader(open(f"{BASE}/selection_law.csv")))
k1 = next(r for r in sel if r["k"] == "1")
check("selection_exact_law_k1", float(k1["exact_law"]), 0.0, 1e-9)
kmax = max(sel, key=lambda r: int(r["k"]))
check("selection_reported_below_envelope_kmax", float(kmax["reported"]) < float(kmax["envelope"]), True)

det = {r["arm"]: r for r in csv.DictReader(open(f"{BASE}/detection_recall.csv"))}
check("detector_recall_exact", float(det[exp["detector_exact_arm"]]["recall_leaked"]), 1.0)
check("detector_false_flags_total", sum(int(r["false_flags"]) for r in det.values()), 0)

n, broken, prev = 0, 0, "genesis"
for line in open(f"{BASE}/ledger.jsonl"):
    e = json.loads(line)
    if e["prev"] != prev: broken += 1
    prev = e["hash"]; n += 1
check("ledger_entries_nonempty", n > 0, True)
check("ledger_chain_breaks", broken, 0)

print("\n" + ("ALL CHECKS PASSED" if not fail else f"{len(fail)} CHECK(S) FAILED"))
sys.exit(0 if not fail else 1)
