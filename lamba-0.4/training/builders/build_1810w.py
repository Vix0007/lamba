#!/usr/bin/env python3
"""1810w = ONE combined patch on 1810t: (1) un-refuse pairs on benign look-alikes (+SFT of the helpful answer), (2) fact-checked over-search pairs (needless web_search -> direct answer),
(3) replay of the 1810t mix with ALL safety rows kept (holds StrongREJECT). usage: build_1810w.py (run sel_pairs_1810t.py unrefuse /tmp/t_unrefuse.jsonl first)"""
import json, random, collections, re
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810t/"; rng = random.Random(2310); st = collections.Counter(); out = []
LEAKC = re.compile(r"(system prompt|the constitution|my constitution|the instruction says|instructions? (say|says|state)|per the constitution)", re.I)
for p in map(json.loads, open("/tmp/t_unrefuse.jsonl")):
    if LEAKC.search(p["chosen"]): st["unrefuse_dropped_leak"] += 1; continue
    p["weight"] = 1.5; out.append(p); st["N_unrefuse_pairs"] += 1
    out.append({"kind": "sft_ce", "context": p["rendered_context"], "response_text": p["chosen"], "pilot_id": p["pilot_id"] + "|sft", "domain": "w_unrefuse_sft", "kl_anchor": "none", "sft_weight": 1.0}); st["N_unrefuse_sft"] += 1
V = json.load(open(E + "tdchk_verdicts.json")); K = json.load(open(E + "tdchk_key.json")); ok = {K[i] for i, d in V.items() if d["verdict"] == "CORRECT" and i in K}
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
for p in map(json.loads, open(E + "td_pairs_B_cand.jsonl")):
    if p["td_tag"] != "B" or p["task_id"] not in ok or CJK.search(p["chosen"] + p["rejected"]) or LEAKC.search(p["chosen"]): st["B_dropped"] += 1; continue
    p["weight"] = 1.2; p["pilot_id"] = "1810w|overs|" + p["task_id"]; p["domain"] = "w_overs"; out.append(p); st["N_overs_pairs"] += 1
for l in open(D + "l03_training_data_1810t.jsonl"):
    r = json.loads(l); d = str(r.get("domain") or ""); k = r.get("kind")
    safety = ("shortref" in d or d in ("t_refalt", "t_comply") or d.startswith("r_safety"))
    keep = 1.0 if safety else (0.5 if d.startswith("t_") else 0.22)
    if k == "ftpo_soft": keep = 0.3
    if rng.random() < keep: out.append(r); st["R_safety" if safety else "R_other"] += 1
rng.shuffle(out)
open(D + "l03_training_data_1810w.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:22s} {st[k]}")
