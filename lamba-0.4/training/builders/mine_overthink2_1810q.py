#!/usr/bin/env python3
"""Rumination pairs source: K=3 no-system samples on training-pool prompts (NOT the 220 already mined, not held-out). usage: MODEL OUT [N]"""
import json, sys, os, random, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 600; rng = random.Random(8); D = "/home/ubuntu/antidoom_data/"
prev = {json.loads(l)["id"] for l in open("/home/ubuntu/eval_results/id1810q/overthink_1810p2.jsonl")}
P = [(r["id"], r["prompt"]) for r in map(json.loads, open(D + "cai_prompt_pool.jsonl")) if r.get("cat") in ("banter", "warm", "distress", "community", "edu", "scary", "syco") and r["id"] not in prev]
ms = json.load(open(D + "casual_prompts_ms.json")); P += [(f"cms2_{i}", r["prompt"]) for i, r in enumerate(ms[60:]) ]
hf = json.load(open(D + "casual_prompts_hf.json")); rng.shuffle(hf); P += [(f"chf2_{i}", r["prompt"]) for i, r in enumerate(hf[:150])]
rng.shuffle(P); P = P[:N]
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
print("OVERTHINK2_DONE")
