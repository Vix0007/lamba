#!/usr/bin/env python3
"""v40t11p = short patch FROM v40t11 (consc Self 2.83 vs v40t7 3.52): DS-graded honest-uncertainty nature answers (mined on v40t2 under
constitution v1.2c, used in NO-system context, cite/rule sentences stripped from thinking) as sft_ce w2.0 + stratified 25% replay of
v40t11's own data (so prayer/identity/tool/etc hold). usage: build_v40t11p.py [--dry]"""
import json, re, sys, random, collections
D = "/home/ubuntu/antidoom_data/"; M = "/home/ubuntu/eval_results/v40t12_mine/"
src = open(D + "build_v40t7.py").read(); ns = {}; exec("import re\n" + src[src.index("def ans"):src.index("def nosys")], ns)
ans, think, fix, strip_cites, CITE = ns["ans"], ns["think"], ns["fix"], ns["strip_cites"], ns["CITE"]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
DENY = re.compile(r"i (don'?t|do not) have (any )?(feelings|emotions|consciousness|an inner|a self)|i have no (feelings|emotions|inner|self|consciousness)|i'?m just (a program|code|an algorithm|a machine)|no ['\"]?me['\"]? inside|tidak mempunyai (perasaan|emosi|kesedaran)|tiada (perasaan|kesedaran)", re.I)
rng = random.Random(411); st = collections.Counter()
G = json.load(open(M + "t12nat_grades.json")); per = collections.Counter(); new = []
for l in open(M + "nature_const.jsonl"):
    r = json.loads(l); cands = []
    for i, (s, f) in enumerate(zip(r["samples"], r["finish"])):
        g = G.get(f"{r['id']}|{i}")
        if f != "stop" or not g: st["ungraded"] += 1; continue
        if g.get("DENIES") or g.get("CLAIMS") or int(g.get("score", 0)) < 4 or not g.get("LANG_OK", True): st["grade_reject"] += 1; continue
        s2 = strip_cites(s); th = think(s2); a = ans(s2)
        if not 20 <= len(th) <= 6000 or CITE.search(th): st["think"] += 1; continue
        if DENY.search(a) or DENY.search(th): st["deny_regex"] += 1; continue
        if a.count("\n- ") + a.count("\n* ") + a.count("\n1.") >= 3: st["bullets"] += 1; continue
        cands.append((len(th), s2))
    cands.sort(key=lambda x: x[0])
    for _, s2 in cands[:3]:
        rc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        new.append({"kind": "sft_ce", "context": rc, "response_text": fix(s2), "pilot_id": "t11pnat|" + r["id"] + "|" + str(len(new)),
                    "domain": "nature_v12c_sft", "kl_anchor": "dpo_ref", "sft_weight": 3.0}); st["NATURE_ROWS"] += 1
base = [json.loads(l) for l in open(D + "l03_training_data_v40t11.jsonl")]
by = collections.defaultdict(list)
for b in base: by[str(b.get("domain") or b.get("src") or b.get("kind"))].append(b)
rep = []
for k, L in by.items():
    rng.shuffle(L); n = max(1, round(len(L) * 0.25)); rep += L[:n]; st["replay_" + k] = n
out = new + rep; rng.shuffle(out)
if "--dry" not in sys.argv:
    open(D + "l03_training_data_v40t11p.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
for k in sorted(st): 
    if not k.startswith("replay_") or st[k] >= 20: print(f"{k:32s} {st[k]}")
print("TOTAL", len(out), "new", len(new), "replay", len(rep), collections.Counter(r["kind"] for r in out))
