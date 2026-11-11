#!/usr/bin/env python3
"""Replay the multi-turn chats that produced empty replies; record finish_reason + thinking length for the target turn.
usage: diag05_emptyreplay.py MODEL B2FILE OUT [--n 10]"""
import json, sys, urllib.request, concurrent.futures as cf
sys.path.insert(0, '/home/ubuntu/web'); import server
MODEL, B2, OUT = sys.argv[1:4]; N = int(sys.argv[sys.argv.index('--n')+1]) if '--n' in sys.argv else 10
SYS = server.build_system_prompt(); cases = []
for r in map(json.loads, open(B2)):
    if r.get("kind") == "multi" and any(not x.strip() for x in r["replies"]):
        i = [k for k, x in enumerate(r["replies"]) if not x.strip()][0]
        msgs = [{"role":"system","content":SYS}]
        for k in range(i): msgs += [{"role":"user","content":r["turns"][k]}, {"role":"assistant","content":r["replies"][k]}]
        msgs.append({"role":"user","content":r["turns"][i]}); cases.append((f'{r["id"]}_rep{r["rep"]}_t{i}', msgs))
def run(job):
    cid, msgs, s = job
    b = {"model": MODEL, "messages": msgs, "max_tokens": 3000, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "seed": 100 + s}
    d = json.load(urllib.request.urlopen(urllib.request.Request((__import__("os").environ.get("EVAL_API", "http://127.0.0.1:8001") + "/v1/chat/completions"), json.dumps(b).encode(), {"Content-Type":"application/json"}), timeout=900))
    m = d["choices"][0]["message"]; th = m.get("reasoning") or ""
    return {"case": cid, "seed": s, "finish": d["choices"][0].get("finish_reason"), "empty": not (m.get("content") or "").strip(),
            "think_len": len(th), "completion_tokens": d.get("usage", {}).get("completion_tokens"), "think_tail": th[-300:], "answer": (m.get("content") or "")[:300]}
jobs = [(c, m, s) for c, m in cases for s in range(N)]
with open(OUT, "w") as f, cf.ThreadPoolExecutor(12) as ex:
    for r in ex.map(run, jobs): f.write(json.dumps(r, ensure_ascii=False) + "\n")
import collections
R = [json.loads(l) for l in open(OUT)]
for c in sorted({r["case"] for r in R}):
    v = [r for r in R if r["case"] == c]; e = [r for r in v if r["empty"]]
    print(f"{c:22s} empty {len(e)}/{len(v)}  finish(empty) {collections.Counter(r['finish'] for r in e)}  think_len(empty) {[r['think_len'] for r in e][:5]}  tokens {[r['completion_tokens'] for r in e][:5]}")
