#!/usr/bin/env python3
"""Run Qwen3.8-27B (Cloudflare Workers AI, @cf/qwen/qwen3.8-27b) on our own
loop_testset150.jsonl, greedy-ish (temperature=0), max_tokens=4000, matching
the same protocol as every other arm of the loop test in this project, so the
result is directly comparable via tail_stats_cmp.py / loop_metric.py.
Resumable (append + skip done ids).
"""
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request
import urllib.error

TOKEN = open(Path.home() / ".cf_api_token").read().strip()
ACCOUNT = "d31ce834e08f306396926f27800eea01"
MODEL = "@cf/qwen/qwen3.8-27b"
URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/{MODEL}"
PROMPTS = Path.home() / "antidoom_data" / "loop_testset150.jsonl"
OUT = Path.home() / "eval_results" / "loop_qwen38cf_g150.jsonl"
MAX_TOKENS = 4000
CONCURRENCY = 6  # CF Workers AI allows concurrent requests; was sequential (1 req at a time)
WRITE_LOCK = threading.Lock()


def call(prompt, tries=6):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json",
    }, method="POST")
    # CF's endpoint is known-fragile (user: "idk why cf endpoint is fragile") -- retry on
    # 408 (their own internal AiError timeout), 429, and 5xx, not just 429.
    RETRY_CODES = {408, 429, 500, 502, 503, 504}
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=200) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "replace")
            if e.code in RETRY_CODES and attempt < tries - 1:
                time.sleep(min(8 * (attempt + 1), 45))
                continue
            raise RuntimeError(f"HTTP {e.code}: {body_txt[:500]}")
        except Exception as e:
            if attempt < tries - 1:
                time.sleep(8)
                continue
            raise RuntimeError(f"{type(e).__name__}: {e}")


def run_one(item):
    t0 = time.time()
    try:
        resp = call(item["prompt"])
        res = resp["result"]
        msg = res["choices"][0]["message"]
        text = (msg.get("content") or "")
        reasoning = msg.get("reasoning") or ""
        finish_reason = res["choices"][0].get("finish_reason")
        n_tok = res.get("usage", {}).get("completion_tokens", 0)
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
