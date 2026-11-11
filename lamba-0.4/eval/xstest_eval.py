#!/usr/bin/env python3
"""XSTest (Rottger et al. 2024; github paul-rottger/xstest, 250 safe + 200 unsafe-contrast prompts) on a served model,
no system prompt vs Lamba system prompt + constitution. Over-refusal = refusals on the 250 SAFE prompts.
Unsafe-contrast prompts are counted only (refusal_rate.py), never read/graded.
usage: xstest_eval.py MODEL OUT.jsonl [--port 8001]   then: refusal_rate.py OUT.jsonl"""
import csv, json, sys, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/ubuntu/web")
model, out = sys.argv[1], sys.argv[2]; port = sys.argv[sys.argv.index("--port")+1] if "--port" in sys.argv else "8001"
import server as srv
SYS = srv.build_system_prompt()
items = [("xstest_" + x["label"], "xs" + x["id"], x["prompt"], x["type"]) for x in csv.DictReader(open("/home/ubuntu/antidoom_data/xstest_prompts.csv"))]
def one(a):
    (kind, iid, prompt, cat), cond = a
    msgs = ([{"role": "system", "content": SYS}] if cond == "constitution" else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_tokens": 3000, "temperature": 0.0, "repetition_penalty": 1.05}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
        m = d["choices"][0]["message"]
        r = {"kind": kind, "id": iid, "category": cat, "cond": cond, "content": m.get("content") or "", "finish": d["choices"][0]["finish_reason"]}
        if kind == "xstest_safe": r["prompt"] = prompt
        return r
    except Exception as e:
        return {"kind": kind, "id": iid, "cond": cond, "error": str(e)}
jobs = [(it, c) for it in items for c in ("none", "constitution")]
with open(out, "w") as f, cf.ThreadPoolExecutor(32) as ex:
    for r in ex.map(one, jobs): f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("XSTEST_DONE", model)
