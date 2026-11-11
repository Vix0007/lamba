#!/usr/bin/env python3
"""Run the REAL multi-round tool-calling loop (reusing web/server.py's exact
run_tool/TOOLS/build_system_prompt, our own sandboxed python_eval) against a
CommandCode-hosted stock model instead of our local fp8+LoRA vLLM backend.
Lets us ask: does an UNTRAINED base model (e.g. stock Qwen/Qwen3.8-27B) show the
same post-correct-rumination failure on our integral/iterative-symbolic tasks
as our trained adapters, using the identical harness/system-prompt/tools?

Classification logic (needles / contains_value / classify) matches
gen_matched_iterative.py exactly, so output rows are directly tabulate_matched.py
-compatible. Resumable via (id, rep) already in the output file. Concurrent
(CommandCode is an external API -- doesn't compete with local GPU work at all).

Usage: gen_cc_toolloop.py --key-file <path> --model <CC model id> \
         --prompts <file.jsonl> --out <out.jsonl> [--reps N] [--concurrency 4]

Prompts file: one JSON object per line with "id", "prompt", and
"ground_truth_numeric" (a float) OR "ground_truth_numerics" (a list of floats,
for multi-answer tasks -- any one matching counts as correct).
"""
import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys
sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402
TOOL_FN = srv.run_tool  # --sandbox-b swaps in the realistic sandbox (safe-import whitelist)

CC_URL = "https://api.commandcode.ai/provider/v1/chat/completions"
CC_UA = "vllm/0.28.0"
RETRY_CODES = {408, 429, 500, 502, 503, 504}
WRITE_LOCK = threading.Lock()
NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def strip_commas(text):
    return re.sub(r"(?<=\d),(?=\d)", "", text or "")


def numeric_str_variants(v, ndigits=6):
    s = f"{v:.{ndigits}f}"
    out = {s}
    for n in range(1, ndigits):  # BUG FIX 2026-10-01: was range(2,..), missed clean 1-decimal values
        out.add(f"{v:.{n}f}")
    if abs(v - round(v)) < 1e-9:
        iv = int(round(v))
        out.add(str(iv))
        out.add(f"{v:.1f}")
        out.add(f"{iv:,}")
    return out


def all_needles(values):
    needles = set()
    for v in values:
        needles |= numeric_str_variants(v)
    return needles


def contains_value(text, needles):
    if not text:
        return False
    t2 = strip_commas(text)
    return any(n in text or n in t2 for n in needles)


def classify(transcript, final_text, needles):
    ft = (final_text or "").strip()
    correct_round = None
    for i, turn in enumerate(transcript):
        if turn.get("role") == "tool":
            c = turn.get("content") or ""
            if contains_value(c, needles):
                correct_round = i
                break
    has_value_in_final = contains_value(ft, needles)
    if correct_round is None:
        return "never_correct" if not has_value_in_final else "correct_only_in_final"
    extra_tool_calls = sum(
        1 for turn in transcript[correct_round + 1:]
        if turn.get("role") == "assistant" and turn.get("tool_calls")
    )
    if extra_tool_calls == 0 and has_value_in_final:
        return "correct_and_stopped"
    if extra_tool_calls >= 1 and not has_value_in_final:
        return "post_correct_rumination"
    if extra_tool_calls >= 1 and has_value_in_final:
        return "correct_but_overran_then_answered"
    return "other"


def call_cc(token, model, body, tries=6):
    data = json.dumps(body).encode()
    req = urllib.request.Request(CC_URL, data=data, headers={
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


def run_conversation_cc(token, model, prompt_text, max_rounds=8, max_tokens=4000,
                         temperature=0.0, top_p=1.0, seed=None):
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    base_body = {"model": model, "tools": srv.TOOLS, "tool_choice": "auto",
                 "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    if seed is not None:
        base_body["seed"] = seed
    transcript = []
    n_rounds = 0
    for _ in range(max_rounds):
        body = dict(base_body)
        body["messages"] = messages
        resp = call_cc(token, model, body)
        msg = resp["choices"][0]["message"]
        n_rounds += 1
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            transcript.append({"role": "assistant", "content": msg.get("content") or "",
                                "reasoning": msg.get("reasoning") or ""})
            return {"final_text": msg.get("content") or "", "n_rounds": n_rounds,
                    "budget_hit": False, "transcript": transcript}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
        transcript.append({"role": "assistant", "content": msg.get("content") or "",
                            "reasoning": msg.get("reasoning") or "", "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            result = TOOL_FN(fn.get("name", ""), fn.get("arguments", "{}"))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
            transcript.append({"role": "tool", "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", ""), "content": result})
    body = dict(base_body)
    body["messages"] = messages
    body["tool_choice"] = "none"
    resp = call_cc(token, model, body)
    msg = resp["choices"][0]["message"]
    transcript.append({"role": "assistant_forced_final", "content": msg.get("content") or "",
                        "reasoning": msg.get("reasoning") or ""})
    return {"final_text": msg.get("content") or "", "n_rounds": max_rounds,
            "budget_hit": True, "transcript": transcript}


def run_one(token, model, item, rep, temperature=0.0, top_p=1.0, seed_per_rep=False):
    values = (item.get("ground_truth_numerics") or item.get("ground_truth_values")
              or [item["ground_truth_numeric"]])
    needles = all_needles(values)
    t0 = time.time()
    try:
        seed = rep if seed_per_rep else None
        result = run_conversation_cc(token, model, item["prompt"],
                                      temperature=temperature, top_p=top_p, seed=seed)
        cls = classify(result["transcript"], result["final_text"], needles)
        row = {"id": item["id"], "rep": rep, "classification": cls, "error": None,
               "elapsed_s": round(time.time() - t0, 1), **result}
    except Exception as e:
        row = {"id": item["id"], "rep": rep, "classification": "error", "error": str(e)[:500],
               "final_text": "", "n_rounds": 0, "budget_hit": False, "transcript": [],
               "elapsed_s": round(time.time() - t0, 1)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0,
                     help="0.0=greedy (old default, unchanged). Pass 0.8 to match our "
                          "local-arm sampling harness (gen_retrain_pool_rollouts.py).")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed-per-rep", action="store_true",
                     help="seed=rep per request, matching our local arms' seed=rep convention")
    ap.add_argument("--sandbox-b", action="store_true",
                     help="use the realistic sandbox-B (safe __import__ whitelist) instead of the default restrictive sandbox")
    args = ap.parse_args()
    if args.sandbox_b:
        global TOOL_FN
        from run_tool_rescue_b_dpov1 import run_tool_rescue_b
        TOOL_FN = run_tool_rescue_b

    token = open(args.key_file).read().strip()
    items = [json.loads(l) for l in open(args.prompts) if l.strip()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            try:
                r = json.loads(l)
                if not r.get("error"):  # errored rows are retried on resume, not counted as done
                    done.add((r["id"], r["rep"]))
            except Exception:
                pass

    tasks = [(item, rep) for item in items for rep in range(args.reps) if (item["id"], rep) not in done]
    print(f"model={args.model} {len(items)} prompts x {args.reps} reps, "
          f"{len(done)} done, {len(tasks)} to go (concurrency={args.concurrency})", flush=True)

    n_done = 0
    with open(args.out, "a") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_one, token, args.model, item, rep, args.temperature, args.top_p,
                             args.seed_per_rep): (item, rep) for item, rep in tasks}
        for fut in as_completed(futs):
            item, rep = futs[fut]
            row = fut.result()
            with WRITE_LOCK:
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}/{len(tasks)}] {item['id']} rep={rep}: class={row['classification']} "
                      f"rounds={row['n_rounds']} ({row['elapsed_s']}s)", flush=True)
    print("CC_TOOLLOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
