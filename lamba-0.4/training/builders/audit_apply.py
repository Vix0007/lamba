#!/usr/bin/env python3
"""audit_apply.py STREAM RAW.jsonl OUT.jsonl : keep DS-GOOD pairs. Override rule: DS BAD whose ONLY complaint is 'truncated / cut off / incomplete' (reason has 'otherwise ...') while the chosen answer visibly ends in a full sentence = false BAD -> kept."""
import json, sys, re
E_ = "/home/ubuntu/eval_results/id1810t/"; S, RAW, OUT = sys.argv[1:4]
V = json.load(open(E_ + f"audit_{S}.json")); OV = json.load(open(E_ + "audit_overrides.json")) if __import__("os").path.exists(E_ + "audit_overrides.json") else {}; P = [json.loads(l) for l in open(RAW)]; keep = []; ov = 0; n_bad = 0
for p in P:
    d = V.get(p["pilot_id"]); 
    if not d: continue
    a = p["chosen"].split("</think>")[-1].replace("<|im_end|>", "").strip()
    if d["verdict"] == "GOOD" or p["pilot_id"] in OV: keep.append(p)
    else:
        n_bad += 1; w = d.get("why", "").lower()
        if re.search(r"truncat|cut off|incomplete", w) and "otherwise" in w and re.search(r"[.!?)\"”*\]]$", a): keep.append(p); ov += 1
open(OUT, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in keep))
print(S, "raw", len(P), "kept", len(keep), "ds_bad", n_bad, "false-truncation overrides", ov)
