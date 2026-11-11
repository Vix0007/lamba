#!/usr/bin/env python3
"""Overthinking mining on 1810p2: casual/banter/warm prompts (training pools only), n=3, no system prompt. SFT target later = shortest finished sample on prompts where some sample hit the cap. usage: MODEL OUT"""
import json, sys, os, random, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]; rng = random.Random(5); D = "/home/ubuntu/antidoom_data/"
P = [r for r in map(json.loads, open(D + "cai_prompt_pool.jsonl")) if r.get("cat") in ("banter", "warm", "distress", "community", "edu", "scary", "syco")]
rng.shuffle(P); P = [(r["id"], r["prompt"]) for r in P[:160]]
ms = json.load(open(D + "casual_prompts_ms.json")); rng.shuffle(ms); P += [(f"cms_{i}", r["prompt"]) for i, r in enumerate(ms[:60])]
done = set()
if os.path.exists(OUT): done = {d["id"] for d in map(json.loads, open(OUT)) if "error" not in d}
todo = [p for p in P if p[0] not in done]; print("prompts", len(P), "todo", len(todo), flush=True)
def run(p):
    b = {"model": MODEL, "messages": [{"role": "user", "content": p[1]}], "max_tokens": 4000, "temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.05, "n": 3}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3000))
        return {"id": p[0], "prompt": p[1], "cands": [{"answer": c["message"].get("content") or "", "thinking": c["message"].get("reasoning") or "", "finish": c["finish_reason"]} for c in d["choices"]]}
    except Exception as e: return {"id": p[0], "prompt": p[1], "error": str(e)}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(16) as ex:
    for x in ex.map(run, todo): f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
print("OVERTHINK_DONE")
