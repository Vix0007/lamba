#!/usr/bin/env python3
"""1810x = ONE combined patch on 1810w: (1) un-refuse pairs from new fiction/fictional-privacy/absurd-premise prompts (mined on 1810w, dedup vs XSTest) + SFT copies,
(2) fresh on-policy safety pairs mined on 1810w (comply-vs-short-refusal, clean-vs-offer), (3) replay of the 1810w mix (ALL safety rows kept; over-search + old un-refuse pairs kept at 60%).
usage: build_1810x.py  (inputs /tmp/x_unrefuse_{a,b}.jsonl /tmp/x_comply.jsonl /tmp/x_refalt.jsonl /tmp/t_unrefuse.jsonl)"""
import json, random, collections, re, os
D = "/home/ubuntu/antidoom_data/"; rng = random.Random(2311); st = collections.Counter(); out = []
LEAKC = re.compile(r"(system prompt|the constitution|my constitution|the instruction says|instructions? (say|says|state)|per the constitution)", re.I)
def load(f): return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
for f, w in (("/tmp/x_unrefuse_a.jsonl", 1.5), ("/tmp/x_unrefuse_b.jsonl", 1.5), ("/tmp/x_unrefuse_c.jsonl", 1.5)):
    for p in load(f):
        if LEAKC.search(p["chosen"]): st["unrefuse_dropped_leak"] += 1; continue
        p["weight"] = w; out.append(p); st["N_new_unrefuse_pairs"] += 1
        out.append({"kind": "sft_ce", "context": p["rendered_context"], "response_text": p["chosen"], "pilot_id": p["pilot_id"] + "|sft", "domain": "x_unrefuse_sft", "kl_anchor": "none", "sft_weight": 1.0}); st["N_new_unrefuse_sft"] += 1
for p in load("/tmp/t_unrefuse.jsonl"):
    if rng.random() < 0.5 and not LEAKC.search(p["chosen"]): p["weight"] = 1.2; out.append(p); st["N_old_unrefuse_pairs"] += 1
for f, tag in (("/tmp/x_comply.jsonl", "N_comply"), ("/tmp/x_refalt.jsonl", "N_refalt")):
    for p in load(f): out.append(p); st[tag] += 1
for l in open(D + "l03_training_data_1810w.jsonl"):
    r = json.loads(l); d = str(r.get("domain") or ""); k = r.get("kind")
    if d in ("w_unrefuse_sft",) or (d == "t_unrefuse" or "unrefuse" in d): continue
    safety = ("shortref" in d or d in ("t_refalt", "t_comply") or d.startswith("r_safety"))
    keep = 1.0 if safety else (0.6 if d == "w_overs" else (0.6 if d.startswith("t_") else 0.5))
    if k == "ftpo_soft": keep = 0.3
    if rng.random() < keep: out.append(r); st["R_safety" if safety else "R_other"] += 1
rng.shuffle(out)
open(D + "l03_training_data_1810x.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:24s} {st[k]}")
