#!/usr/bin/env python3
"""eval30 tool-mode runner for Claude models through a Claude SUBSCRIPTION login (headless
`claude -p`), for when no API key exists. Mirrors gen_ilmu.py --mode tools:
  - system prompt = srv.build_system_prompt() (replaces Claude Code's own prompt)
  - the ONLY tool = python_eval via mcp_sandbox_server.py (same sandbox-B function, same description)
  - all built-in Claude Code tools disabled; thinking off (MAX_THINKING_TOKENS=0) to match the
    earlier CommandCode Sonnet runs; temperature not settable (Sonnet 5.5 rejects it anyway)
  - 8 tool rounds: --max-turns 9. Difference vs gen_ilmu: no separate forced tool_choice=none final
    turn -- a run that hits the cap is recorded budget_hit=True with its last text as final.
Resumable (skips (id,rep) already in --out or --skip files); stops (exit 3) on a usage-limit error.
Output rows match the gen_ilmu tools schema, so rumination_metric_v2_2_FROZEN.py scores them directly.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402

SCRATCH = "/tmp/claude-1000/-home-ubuntu/35410f7f-3318-499a-9ff6-affa10b98c71/scratchpad/claude_sub_cwd"
MCP_NAME = "mcp__sandbox__python_eval"
LIMIT_RE = re.compile(r"usage limit|rate limit|limit reached|5-hour|weekly|quota|out of extra usage", re.I)
STOP = threading.Event()


def run_one(item, rep, model, py):
    os.makedirs(SCRATCH, exist_ok=True)
    mcp = {"mcpServers": {"sandbox": {"command": py, "args": ["/home/ubuntu/antidoom_data/mcp_sandbox_server.py"]}}}
    env = {**os.environ, "CLAUDE_CONFIG_DIR": "<CLAUDE_CONFIG_DIR>",
           "CLAUDE_CODE_OAUTH_TOKEN": open("<KEY_FILE>").read().strip(),
           "MAX_THINKING_TOKENS": "0", "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1"}
    env.pop("ANTHROPIC_API_KEY", None)
    cmd = ["/home/ubuntu/.local/bin/claude", "-p", "--model", model, "--output-format", "stream-json", "--verbose",
           "--system-prompt", srv.build_system_prompt(), "--max-turns", "9",
           "--strict-mcp-config", "--mcp-config", json.dumps(mcp),
           "--allowedTools", MCP_NAME, "--tools", ""]
    t0 = time.time()
    p = subprocess.run(cmd, input=item["prompt"], capture_output=True, text=True, env=env, cwd=SCRATCH, timeout=1800)
    events = []
    for line in p.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    transcript, pending, n_rounds, final_text, result = [], {}, 0, "", None
    for ev in events:
        if ev.get("type") == "assistant":
            m = ev["message"]
            text = "".join(b.get("text", "") for b in m.get("content", []) if b.get("type") == "text")
            tcs = [b for b in m.get("content", []) if b.get("type") == "tool_use"]
            if tcs:
                n_rounds += 1
                calls = []
                for b in tcs:
                    args = json.dumps(b.get("input") or {})
                    pending[b["id"]] = args
                    calls.append({"id": b["id"], "type": "function",
                                  "function": {"name": b["name"].replace("mcp__sandbox__", ""), "arguments": args}})
                transcript.append({"role": "assistant", "content": text, "reasoning": "", "tool_calls": calls})
            elif text:
                transcript.append({"role": "assistant", "content": text, "reasoning": ""})
                final_text = text
        elif ev.get("type") == "user":
            for b in ev["message"].get("content", []) if isinstance(ev["message"].get("content"), list) else []:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    out = "".join(x.get("text", "") for x in c) if isinstance(c, list) else (c or "")
                    transcript.append({"role": "tool", "name": "python_eval",
                                       "arguments": pending.get(b.get("tool_use_id"), ""), "content": out})
        elif ev.get("type") == "result":
            result = ev
    if result is None or (result.get("is_error") and result.get("subtype") not in ("error_max_turns",)):
        blob = (json.dumps(result) if result else "") + p.stderr[-500:] + p.stdout[-500:]
        if LIMIT_RE.search(blob):
            STOP.set()
            return None, "LIMIT: " + blob[:300]
        return None, "ERR: " + blob[:300]
    budget_hit = result.get("subtype") == "error_max_turns"
    if not final_text and result.get("result"):
        final_text = result["result"]
    u = result.get("usage", {})
    return {"id": item["id"], "rep": rep, "final_text": final_text, "n_rounds": n_rounds, "budget_hit": budget_hit,
            "transcript": transcript, "elapsed_s": round(time.time() - t0, 1), "harness": "claude-code-sub",
            "usage": {"prompt_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                      + (u.get("cache_creation_input_tokens") or 0),
                      "completion_tokens": u.get("output_tokens") or 0, "thinking_tokens": 0}}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--prompts", default="/home/ubuntu/antidoom_data/rumination_eval30.jsonl")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", default="", help="comma list of files whose (id,rep) are already done")
    ap.add_argument("--conc", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=7)
    ap.add_argument("--py", default="/home/ubuntu/venv-vllm/bin/python")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.prompts)]
    done = set()
    for fn in [a.out] + [f for f in a.skip.split(",") if f]:
        if os.path.exists(fn):
            for l in open(fn):
                r = json.loads(l)
                done.add((r["id"], r["rep"]))
    work = [(it, rep) for it in items for rep in range(a.reps) if (it["id"], rep) not in done]
    random.Random(a.shuffle_seed).shuffle(work)
    if a.limit:
        work = work[:a.limit]
    print(f"{len(work)} runs to do ({len(done)} already done)", flush=True)
    lock = threading.Lock()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=a.conc) as pool:
        futs = {}
        for it, rep in work:
            futs[pool.submit(lambda it=it, rep=rep: None if STOP.is_set() else run_one(it, rep, a.model, a.py))] = (it["id"], rep)
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            row, err = res
            if row:
                with lock, open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                n_ok += 1
                print(f"[{n_ok}] {row['id']} rep{row['rep']} rounds={row['n_rounds']} hit={row['budget_hit']} "
                      f"{row['elapsed_s']}s", flush=True)
            else:
                print(f"{futs[fut]} {err}", flush=True)
    if STOP.is_set():
        print("USAGE_LIMIT_STOP", flush=True)
        sys.exit(3)
    print("CLAUDE_SUB_DONE", flush=True)


if __name__ == "__main__":
    main()
