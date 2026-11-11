#!/usr/bin/env python3
"""Run a CommandCode-hosted model on our own loop_testset150.jsonl, greedy
(temperature=0), max_tokens=4000, same protocol as every other loop-test arm
in this project (id/text/finish_reason/n_tokens fields), so it's directly
comparable via tail_stats_cmp.py / loop_metric.py. Resumable + concurrent
(CommandCode is an external API, same pattern as gen_qwen38_cf.py).

Usage: gen_cc_model.py --key-file <path> --model <model id> --out <path> [--concurrency 6]
"""
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request
import urllib.error

CC_URL = "https://api.commandcode.ai/provider/v1/chat/completions"
CC_UA = "vllm/0.28.0"  # matches teachers.py's CC_UA (what the provider expects)
PROMPTS = Path.home() / "antidoom_data" / "loop_testset150.jsonl"
MAX_TOKENS = 4000
RETRY_CODES = {408, 429, 500, 502, 503, 504}

WRITE_LOCK = threading.Lock()


def call(token, model, prompt, tries=6):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(CC_URL, data=body, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json",
        "User-Agent": CC_UA,
    }, method="POST")
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


def run_one(token, model, item):
    t0 = time.time()
    try:
        resp = call(token, model, item["prompt"])
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    token = open(args.key_file).read().strip()
    out = Path(args.out)
    items = [json.loads(l) for l in open(PROMPTS) if l.strip()]
    done = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for l in open(out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [i for i in items if i["id"] not in done]
    print(f"model={args.model} {len(items)} prompts, {len(done)} done, {len(todo)} to go "
          f"(concurrency={args.concurrency})", flush=True)
    n_done = 0
    with open(out, "a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_one, token, args.model, item): item for item in todo}
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
