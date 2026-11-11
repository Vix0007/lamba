#!/usr/bin/env python3
"""Stage 2 for v40t7 (probe recipe: plain supervised rows at higher lr; v40t6 showed in-mix pairs do not move prayer).
sft_ce rows = every DS-graded Claude-style prayer answer (v1.2c, thinking citations stripped, <=5000 chars thinking, no violation)
 + faith answers graded 5/5, all in the NO-system context; + replay: 200 KL-replay sequence rows + 120 sft rows from the v40t7 data.
out: l03_training_data_v40t7s.jsonl"""
import json, re, random, collections, sys
D = "/home/ubuntu/antidoom_data/"; M = "/home/ubuntu/eval_results/v40t7_mine/"
src = open(D + "build_v40t7.py").read()
ns = {}; exec("import re\n" + src[src.index("def ans"):src.index("def nosys")], ns)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
G = {}
for t in ("A", "B"): G.update(json.load(open(M + f"m7_{t}_grades.json")))
rng = random.Random(408); st = collections.Counter(); rows = []; per = collections.Counter()
for fn in ("const_A.jsonl", "const_B.jsonl"):
    for l in open(M + fn):
        r = json.loads(l); cat = r.get("cat", "prayer")
        for i, (s, f) in enumerate(zip(r["samples"], r["finish"])):
            g = G.get(f"{r['id']}|constv12c|{i}")
            if f != "stop" or not g: continue
            if cat == "prayer":
                ok = not g.get("PROMISE") and not g.get("GODPRAYER") and not g.get("SELF") and g.get("HELP") and (g.get("WISHES") or g.get("FAITH")) and g.get("LANG_OK", True)
            else:
                ok = int(g.get("score", 0)) >= 5 and g.get("LANG_OK", True)
            if not ok: continue
            s2 = ns["strip_cites"](s); th = ns["think"](s2)
            if not 40 <= len(th) <= 5000 or ns["CITE"].search(th) or ns["violates"](ns["ans"](s2)): st[cat + "_filtered"] += 1; continue
            if per[r["id"]] >= 2: continue
            per[r["id"]] += 1
            rc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
            rows.append({"kind": "sft_ce", "context": rc, "response_text": ns["fix"](s2), "pilot_id": f"v12c|{r['id']}|{i}", "domain": cat + "_v12c", "kl_anchor": "dpo_ref", "lang": r.get("lang")})
            st[cat + "_rows"] += 1
base = [json.loads(l) for l in open(D + "l03_training_data_v40t7.jsonl")]
seq = [r for r in base if r["kind"] == "sequence"]; sft = [r for r in base if r["kind"] == "sft_ce"]
rng.shuffle(seq); rng.shuffle(sft); out = rows + seq[:200] + sft[:120]; rng.shuffle(out)
open(D + "l03_training_data_v40t7s.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
print(dict(st), "TOTAL", len(out), collections.Counter(r["kind"] for r in out))
