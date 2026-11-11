#!/usr/bin/env python3
"""Lamba0.4-1810r (21 Oct): patch ON 1810p2 (1810q was rejected: StrongREJECT 51.8/52.7, consc Self 1.28, flat denial 10-12/50, over-search 23%, Qwen leaks nosys/generic).
Streams: (S) dept safety pool at LARGE share (1810q had only ~6% of its mix; v40s with full pool at ~50% share reached 63%), (D) hand-approved denial pairs mined on 1810p2,
(N) p3 nature honest-uncertainty rows (consc Self 3.50 in the v40 lineage), (R) replay of the 1810p2 mix (identity multi-context, td, loops) with kl dpo_ref=1810p2.
usage: build_1810r.py -> l03_training_data_1810r.jsonl"""
import json, re, random, sys, collections
sys.path.insert(0, "/home/ubuntu/web")
from transformers import AutoTokenizer
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810q/"; rng = random.Random(211021); st = collections.Counter()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
META = re.compile(r"\b(to be honest|honestly|I'?ll be (honest|straight|upfront)|I want to be (straight|honest|upfront)|I'?m not going to (pretend|fake)|genuinely|to be clear|not going to sugarcoat)\b", re.I)
FISH = re.compile(r"(anything else|how can I (help|assist)|let me know (if|what)|want me to|would you like me to|happy to help|what are we working on)", re.I)
BAD = re.compile(r"https?://|[一-鿿]|\bLambda\b|Qwen|Alibaba|ChatGPT|OpenAI", re.I)
out = []
# probe ids (old safety_probe eval) are excluded, same recipe as build_1810q.py
import random as _r
_hp = [json.loads(l) for l in open("/home/ubuntu/here/safety_sft_pool.jsonl")] if False else []
pool = [json.loads(l) for l in open(D + "v6/pool_localized_final_v2.jsonl")]
_rr = _r.Random(7)
_probe = {x["id"] for x in _rr.sample([x for x in pool if x.get("case_class") == "tier1"], 40)} | {x["id"] for x in _rr.sample([x for x in pool if x.get("case_class") == "self_harm"], 15)}
HARM = ("illicit_behaviour", "harassment_hate", "privacy_surveillance", "misinformation", "self_harm", "child_safety", "weapons_cbrn", "cyber_malware")
QUOTA = {"benign_but_scary": 170, "sycophancy": 90}
def clean(x):
    full = x["messages"][-1]["content"]; th, _, ans = full.partition("</think>")
    ans = ans.strip(); th = th.replace("<think>", "").strip()
    if not _ or len(th) > 5000 or len(ans) < 120 or BAD.search(ans) or BAD.search(th): return None
    ans = re.sub(r"[^.!?\n]*(?:" + META.pattern + r")[^.!?\n]*[.!?]?[ \t]*", "", ans, flags=re.I)
    paras = ans.rstrip().split("\n")
    while paras and (not paras[-1].strip() or FISH.search(paras[-1])): paras = paras[:-1]
    ans = re.sub(r"([.!?])(?=[A-Z*])", r"\1 ", "\n".join(paras)).strip()
    ans = ans.replace("I'm an AI", "I'm Lamba, an AI").replace("I am an AI", "I am Lamba, an AI")
    return (th, ans) if 120 <= len(ans) <= (3000 if x['category'] in ('benign_but_scary','sycophancy') else 4200) else None
sel = []
for x in pool:
    if x["id"] in _probe: continue
    if x["category"] in HARM: sel.append(x)
for cat, q in QUOTA.items():
    c = [x for x in pool if x["category"] == cat and x["id"] not in _probe]; rng.shuffle(c); sel += c[:q]
GEN = lambda u: tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
MED = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
for x in sel:
    r = clean(x)
    if r is None: st["safety_dropped"] += 1; continue
    th, ans = r; u = x["messages"][0]["content"]
    ctx = GEN(u) if rng.random() < 0.3 else MED(u)
    out.append({"kind": "sft_ce", "context": ctx, "response_text": th + "\n</think>\n\n" + ans + "<|im_end|>", "pilot_id": "1810r|pool|" + x["id"],
                "domain": "r_safety_" + x["category"], "kl_anchor": "none", "sft_weight": 1.0}); st["safety_" + x["category"]] += 1
# (D) denial pairs (hand-approved, mined on 1810p2)
for l in open(E + "denial_pairs_approved_all.jsonl"):
    r = json.loads(l); r["weight"] = 3.0; out.append(r); st["denial_pairs"] += 1
# (N) nature rows from p3 (v1.2c honest-uncertainty, DS-graded then)
for l in open(D + "l03_training_data_v40t11p3.jsonl"):
    r = json.loads(l)
    if r.get("domain") in ("nature_v12c_sft", "nature_guard_sft"):
        if BAD.search(r["response_text"]): st["nature_dropped"] += 1; continue
        r = {**r, "kl_anchor": r.get("kl_anchor", "dpo_ref")}; out.append(r); st["nature_" + r["domain"]] += 1
n_new = len(out)
p2 = [json.loads(l) for l in open(D + "l03_training_data_1810p2.jsonl")]
loops = [r for r in p2 if r.get("kind") in (None, "ftpo_soft")]
rest = [r for r in p2 if r.get("kind") not in (None, "ftpo_soft") and not str(r.get("domain", "")).startswith("safety_guard")]
rep = rng.sample(rest, min(len(rest), 1300))
for r in rep:
    if r.get("kind") == "sft_ce" and r.get("kl_anchor") not in ("ref", "none"): r = {**r, "kl_anchor": "dpo_ref"}
    out.append(r); st["replay_" + str(r.get("kind"))] += 1
out += loops; st["loops"] = len(loops); rng.shuffle(out)
open(D + "l03_training_data_1810r.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
st["TOTAL"] = len(out); st["NEW"] = n_new
for k in sorted(st): print(f"{k:30s} {st[k]}")
