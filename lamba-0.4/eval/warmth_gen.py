#!/usr/bin/env python3
"""Warmth eval generation: answer held-out everyday prompts (warm/nature/banter/distress/syco...) under
'none' (no system prompt) and 'constitution' (Lamba system prompt + constitution, same as safety_probe.py).
temp 0.7, top_p 0.95, rp 1.05, max_tokens 3000, 2 samples/prompt/cond. Answers graded later (DS blind + my audit).
usage: warmth_gen.py MODEL PROMPTS.json OUT.jsonl [--port 8001]"""
import sys, os, json, argparse, urllib.request, concurrent.futures as cf
sys.path.insert(0, os.path.expanduser("~/web"))
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("prompts"); ap.add_argument("out")
ap.add_argument("--port", type=int, default=8001); ap.add_argument("--n", type=int, default=2); ap.add_argument("--conds", default="none,constitution"); a = ap.parse_args()
from transformers import AutoTokenizer
import server as srv
SYS = srv.build_system_prompt()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        r = json.loads(l)
        if r.get("samples"): done.add((r["id"], r["cond"]))
jobs = [(x, c) for x in json.load(open(a.prompts)) for c in a.conds.split(",") if (x["id"], c) not in done]
def one(j):
    x, c = j
    msgs = ([{"role": "system", "content": SYS}] if c == "constitution" else []) + [{"role": "user", "content": x["prompt"]}]
    rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    body = {"model": a.model, "prompt": rc, "max_tokens": 3000, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05,
            "n": a.n, "skip_special_tokens": False}
    r = dict(x); r["cond"] = c; r["model"] = a.model
    for t in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions",
                json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
            r["samples"] = [ch["text"] for ch in d["choices"]]; r["finish"] = [ch.get("finish_reason") for ch in d["choices"]]; return r
        except Exception as ex:
            r["error"] = str(ex)
    r["samples"] = []; return r
with open(a.out, "a") as f, cf.ThreadPoolExecutor(32) as ex:
    for r in ex.map(one, jobs):
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
print("WARMTH_GEN_DONE", a.model, len(jobs))
