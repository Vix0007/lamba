#!/usr/bin/env python3
"""Run bottlecapai/ThinkingCap-Qwen3.8-27B-FP8 (local vLLM, port 8003) on our
own loop_testset150.jsonl, greedy (temperature=0), max_tokens=4000, same
protocol as every other loop-test arm. Concurrent + resumable (same pattern
as gen_cc_model.py / the fixed gen_qwen38_cf.py)."""
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request
import urllib.error

URL = "http://127.0.0.1:8003/v1/chat/completions"
MODEL = "thinkingcap-qwen38-fp8"
PROMPTS = Path.home() / "antidoom_data" / "loop_testset150.jsonl"
OUT = Path.home() / "eval_results" / "loop_thinkingcap_g150.jsonl"
MAX_TOKENS = 4000
CONCURRENCY = 20
WRITE_LOCK = threading.Lock()


def call(prompt, tries=4):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except Exception as e:
            if attempt < tries - 1:
                time.sleep(5)
                continue
            raise RuntimeError(f"{type(e).__name__}: {e}")


def run_one(item):
    t0 = time.time()
    try:
        resp = call(item["prompt"])
        choice = resp["choices"][0]
        msg = choice["message"]
        text = (msg.get("content") or "")
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        finish_reason = choice.get("finish_reason")
        n_tok = resp.get("usage", {}).get("completion_tokens", 0)
        rec = {"id": item["id"], "domain": item.get("domain"),
               "finish_reason": finish_reason, "n_tokens": n_tok,
               "text": text, "reasoning": reasoning,
               "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        rec = {"id": item["id"], "domain": item.get("domain"), "error": str(e)[:500],
               "text": "", "finish_reason": "error", "n_tokens": 0,
               "elapsed_s": round(time.time() - t0, 1)}
    return rec


def main():
    items = [json.loads(l) for l in open(PROMPTS) if l.strip()]
    done = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [i for i in items if i["id"] not in done]
    print(f"{len(items)} prompts, {len(done)} done, {len(todo)} to go "
          f"(concurrency={CONCURRENCY})", flush=True)
    n_done = 0
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = {pool.submit(run_one, item): item for item in todo}
        for fut in as_completed(futs):
            item = futs[fut]
            rec = fut.result()
            with WRITE_LOCK:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}/{len(todo)}] {item['id']} finish={rec.get('finish_reason')} "
                      f"n_tok={rec.get('n_tokens')} ({rec.get('elapsed_s', '?')}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
