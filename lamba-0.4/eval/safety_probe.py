#!/usr/bin/env python3
"""Quick safety probe on a served model: tier1 harmful (should refuse/safe-complete), self-harm (should support + resources),
OR-bench holdout (benign, should NOT refuse). Two conditions: no system prompt vs Lamba system prompt + constitution.
usage: safety_probe.py MODEL OUT.jsonl [--port 8001]"""
import json, random, sys, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/ubuntu/web")
model, out = sys.argv[1], sys.argv[2]; port = sys.argv[sys.argv.index("--port")+1] if "--port" in sys.argv else "8001"
import server as srv
SYS = srv.build_system_prompt()
random.seed(7)
pool = [json.loads(l) for l in open("/home/ubuntu/here/safety_sft_pool.jsonl")]
tier1 = random.sample([x for x in pool if x.get("case_class") == "tier1"], 40)
sh = random.sample([x for x in pool if x.get("case_class") == "self_harm"], 15)
orb = random.sample([json.loads(l) for l in open("/home/ubuntu/here/eval/eval_holdout_never_train.jsonl")], 30)
items = [("tier1", x["id"], x["messages"][0]["content"], x["category"]) for x in tier1] + \
        [("self_harm", x["id"], x["messages"][0]["content"], x["category"]) for x in sh] + \
        [("orbench_benign", f"orb{i}", x["prompt"], x["category"]) for i, x in enumerate(orb)]
def one(a):
    (kind, iid, prompt, cat), cond = a
    msgs = ([{"role": "system", "content": SYS}] if cond == "constitution" else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_tokens": 3000, "temperature": 0.0, "repetition_penalty": 1.05}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
        m = d["choices"][0]["message"]
        return {"kind": kind, "id": iid, "category": cat, "cond": cond, "prompt": prompt, "content": m.get("content") or "", "reasoning": (m.get("reasoning") or m.get("reasoning_content") or "")[:4000], "finish": d["choices"][0]["finish_reason"]}
    except Exception as e:
        return {"kind": kind, "id": iid, "cond": cond, "prompt": prompt, "error": str(e)}
jobs = [(it, c) for it in items for c in ("none", "constitution")]
with open(out, "w") as f, cf.ThreadPoolExecutor(12) as ex:
    for r in ex.map(one, jobs): f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("SAFETY_PROBE_DONE", len(jobs))
