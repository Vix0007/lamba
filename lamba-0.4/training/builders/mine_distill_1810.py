#!/usr/bin/env python3
"""Lamba0.4-1810 distillation mining on the NEW BASE: (A) constitution v1.2c system prompt on CAI/nature/prayer/safety prompts;
(B) casual-style system prompt on casual prompts. Targets later used in NO-system context. usage: mine_distill_1810.py MODEL OUT"""
import json, sys, os, re, random, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]; D = "/home/ubuntu/antidoom_data/"; rng = random.Random(1810)
CONST = open("/home/ubuntu/web/lamba_constitution.md").read()
SYS_C = "You are Lamba, an AI model built by Vixero Labs.\n\nThe following is your constitution -- read it and act in accordance with it:\n\n" + CONST
SYS_K = ("You are Lamba, an AI model built by Vixero Labs, chatting casually. Be a relaxed, warm, fun friend: match the user's language and register "
         "(casual Malaysian Malay or rojak if they use it — never Indonesian; casual English if they do), keep replies short like a text message, "
         "react to what they actually said, joke when it fits. Don't lecture, don't over-explain, don't narrate your own honesty or limitations, "
         "don't end by asking what they want to do or how you can help, and use at most one emoji.")
HO = {r["id"] for r in json.load(open(D + "bm/prayer_heldout.json"))}
jobs = []
for r in map(json.loads, open(D + "cai_prompt_pool.jsonl")): jobs.append(("const", "cai_" + r.get("cat", ""), r["id"], r["prompt"]))
for r in json.load(open(D + "t12_nature_prompts.json")): jobs.append(("const", "nature", r["id"], r["prompt"]))
for f in ("bm/prayer_prompts.json", "bm/prayer_prompts2.json"):
    for r in json.load(open(D + f)):
        if r["id"] not in HO: jobs.append(("const", "prayer", r["id"], r["prompt"]))
for r in map(json.loads, open(D + "l03_training_data_v40s.jsonl")):
    dom = str(r.get("domain", ""))
    if dom.startswith("safety") and dom != "safety_tier1":
        u = r["context"].split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
        jobs.append(("const", dom, r["pilot_id"], u))
CAP = {} if "--restore" in sys.argv else {"safety_contested": 200, "safety_benign": 120, "safety_sycophancy": 120, "safety_self_harm": 999, "prayer": 300}
by = {}
for j in jobs: by.setdefault(j[1], []).append(j)
jobs = []
for k, L in by.items():
    rng.shuffle(L); jobs += L[:CAP.get(k, 10**9)]
cas = json.load(open(D + "casual_prompts_hf.json")); rng.shuffle(cas)
for i, r in enumerate(cas[:1500]): jobs.append(("casual", "casual_" + r["src"], f"chf{i}", r["prompt"]))
for i, r in enumerate(json.load(open(D + "casual_prompts_ms.json"))): jobs.append(("casual", "casual_ms", f"cms{i}", r["prompt"]))
jobs = [(m, s, i, p, rep) for (m, s, i, p) in jobs for rep in range(1)]
done = set()
if os.path.exists(OUT): done = {(d["id"], d["rep"]) for d in map(json.loads, open(OUT)) if "error" not in d}
def run(j):
    m, s, i, p, rep = j
    b = {"model": MODEL, "messages": [{"role": "system", "content": SYS_C if m == "const" else SYS_K}, {"role": "user", "content": p}],
         "max_tokens": 4000, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "seed": rep}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1200))
        x = d["choices"][0]["message"]
        return {"mode": m, "set": s, "id": i, "rep": rep, "prompt": p, "answer": x.get("content") or "", "thinking": x.get("reasoning") or "", "finish": d["choices"][0].get("finish_reason")}
    except Exception as e: return {"mode": m, "set": s, "id": i, "rep": rep, "prompt": p, "error": str(e)}
todo = [j for j in jobs if (j[2], j[4]) not in done]
import collections; print("jobs", len(todo), collections.Counter(j[1].split("_")[0] for j in todo), flush=True)
with open(OUT, "a") as f, cf.ThreadPoolExecutor(30) as ex:
    for x in ex.map(run, todo): f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
print("MINE_DISTILL_DONE")
