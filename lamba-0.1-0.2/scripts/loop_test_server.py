#!/usr/bin/env python3
"""g150-style loop test through the LIVE vLLM server (/v1/completions with the exact rendered prompt, greedy, 4000 tok),
so decode-time knobs (repetition_penalty, presence_penalty) can be compared without reloading an offline engine.
Row schema = loop_test_g.py (+ presence_penalty), so tail_stats_cmp.py works (use ALLOW_RP_MISMATCH=1 when rp differs).
usage: loop_test_server.py <served_model> <out.jsonl> --rp 1.0 --pp 0.5 [--limit N] [--prompts loop_testset150.jsonl]"""
import argparse, json, os, random, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--rp", type=float, required=True); ap.add_argument("--pp", type=float, default=0.0)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-tokens", type=int, default=4000)
ap.add_argument("--prompts", default="/home/ubuntu/antidoom_data/loop_testset150.jsonl"); ap.add_argument("--conc", type=int, default=8)
a = ap.parse_args()
TOK = AutoTokenizer.from_pretrained("/home/ubuntu/models/lmh_ruminconfab_dpo_v2", trust_remote_code=True)
items = [json.loads(l) for l in open(a.prompts)]
if a.limit: items = items[:a.limit]
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)["id"])
        except Exception: pass
todo = [b for b in items if b["id"] not in done]
print(f"{len(items)} prompts, {len(done)} done, {len(todo)} todo; rp={a.rp} pp={a.pp}", flush=True)
lock = threading.Lock(); n = {"k": 0}
def one(b):
    ctx = TOK.apply_chat_template([{"role": "user", "content": b["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
    body = {"model": a.model, "prompt": ctx, "max_tokens": a.max_tokens, "temperature": 0.0, "repetition_penalty": a.rp,
            "presence_penalty": a.pp, "skip_special_tokens": False}
    t0 = time.time()
    try:
        r = urllib.request.Request("http://127.0.0.1:8001/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(r, timeout=3600)); c = resp["choices"][0]
        row = {"id": b["id"], "domain": b.get("domain"), "n_pairs": b.get("n_pairs"), "finish_reason": c["finish_reason"],
               "n_tokens": resp["usage"]["completion_tokens"], "elapsed_s": round(time.time() - t0, 1),
               "repetition_penalty": a.rp, "presence_penalty": a.pp, "text": c["text"]}
    except Exception as e:
        row = {"id": b["id"], "error": str(e)[:200]}
    with lock:
        if "error" not in row:
            open(a.out, "a").write(json.dumps(row, ensure_ascii=False) + "\n")
        n["k"] += 1
        if n["k"] % 10 == 0: print(f"[{n['k']}/{len(todo)}]", flush=True)
with ThreadPoolExecutor(a.conc) as ex: list(ex.map(one, todo))
print("LOOP_TEST_SERVER_DONE", flush=True)
