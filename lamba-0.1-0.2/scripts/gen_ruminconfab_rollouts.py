#!/usr/bin/env python3
"""Resumable rollout generator for the tool-call-rumination / confabulation
failure mode (see ~/web/FINDINGS.md). Drives its OWN full multi-round tool
loop directly against the backend (127.0.0.1:8001), reusing the real
web_search/python_eval implementations from ~/web/server.py, so we capture
the ENTIRE transcript (every round's content/reasoning/tool calls) -- not
just the final answer -- which is what's needed to find where rumination
actually starts.

Appends one JSON object per prompt to the output file; skips ids already
present, so it's safe to stop/restart. Flags candidates heuristically
(long reasoning, many rounds, no final content, degrading tool-call length)
for later DS blind-labelling -- flags are NOT a trained/validated detector,
just a cheap pre-filter to sort what a human/DS reviewer looks at first
(same caveat as every other heuristic in this project's history: don't
trust it, it's a triage aid only).
"""
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path.home() / "web"))
import server as srv  # reuse run_tool, TOOLS, build_system_prompt

BACKEND = "http://127.0.0.1:8001"
MODEL_NAME = "letterfix2-c10"
MAX_ROUNDS = 8  # DS 98 protocol: max_tool_rounds=8
MAX_TOKENS_PER_ROUND = 4096  # DS 98 protocol: non-tool max_tokens=4096
TEMPERATURE = 0.7  # DS 98 protocol
TOP_P = 0.9  # DS 98 protocol
OUT = Path.home() / "eval_results" / "ruminconfab_round2_rollouts.jsonl"
PROMPTS = Path.home() / "antidoom_data" / "ruminconfab_round2.jsonl"
CONCURRENCY = 6  # vLLM batches concurrent requests internally (max-num-seqs=8 on lamba-infer);
                 # was strictly sequential (1 multi-round transcript at a time), wasting the
                 # server's batching capacity.
WRITE_LOCK = threading.Lock()


def seed_for(item_id: str) -> int:
    """Fixed seed per prompt (DS 98 protocol: 'fixed seed per prompt')."""
    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)


def post_backend(body, timeout=600):
    req = urllib.request.Request(
        f"{BACKEND}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_one(item):
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": item["prompt"]},
    ]
    tools_mode = item.get("tools", "auto")  # "required" | "disabled" | "auto"
    seed = seed_for(item["id"])
    rounds = []
    for round_i in range(MAX_ROUNDS):
        body = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "max_tokens": MAX_TOKENS_PER_ROUND,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": seed,
            "chat_template_kwargs": srv.DEFAULT_CHAT_TEMPLATE_KWARGS,
        }
        if tools_mode != "disabled":
            # "required" (DS 98 sense) means tools are AVAILABLE for the task, not that
            # the API must force a tool call every round -- that would prevent ever
            # reaching a final answer. tool_choice stays "auto" either way.
            body["tools"] = srv.TOOLS
            body["tool_choice"] = "auto"
        try:
            resp = post_backend(body)
        except urllib.error.HTTPError as e:
            rounds.append({"round": round_i, "error": e.read().decode("utf-8", "replace")[:1000]})
            break
        except Exception as e:
            rounds.append({"round": round_i, "error": str(e)})
            break
        msg = resp["choices"][0]["message"]
        finish = resp["choices"][0].get("finish_reason")
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        tool_calls = msg.get("tool_calls") or []
        rounds.append({
            "round": round_i, "finish_reason": finish,
            "content_len": len(content), "reasoning_len": len(reasoning),
            "content": content, "reasoning": reasoning,
            "tool_calls": [{"name": tc.get("function", {}).get("name"),
                             "arguments": tc.get("function", {}).get("arguments")} for tc in tool_calls],
            "usage": resp.get("usage"),
        })
        if not tool_calls:
            break
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc.get("function", {})
            result = srv.run_tool(fn.get("name", ""), fn.get("arguments", "{}"))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
    return rounds


def flag(rounds):
    """Cheap triage heuristics -- NOT a validated detector. Just sorts what to look at first."""
    flags = []
    if not rounds:
        return ["no_rounds"]
    last = rounds[-1]
    if "error" in last:
        return ["error"]
    final_content = last.get("content", "")
    if len(rounds) >= MAX_ROUNDS - 1:
        flags.append("hit_round_cap")
    if not final_content.strip() and any(r.get("reasoning_len", 0) > 500 for r in rounds):
        flags.append("no_final_content_but_reasoned")
    max_reasoning = max((r.get("reasoning_len", 0) for r in rounds), default=0)
    if max_reasoning > 6000:
        flags.append("very_long_reasoning")
    tool_rounds = [r for r in rounds if r.get("tool_calls")]
    if len(tool_rounds) >= 4:
        flags.append("many_tool_rounds")
    # crude degrading-code proxy: code length growing a lot round over round
    code_lens = [len((tc.get("arguments") or "")) for r in rounds for tc in r.get("tool_calls", [])
                 if tc.get("name") == "python_eval"]
    if len(code_lens) >= 3 and code_lens[-1] > 2 * (sum(code_lens[:2]) / 2 + 1):
        flags.append("growing_code_len")
    return flags


def process_one(item):
    t0 = time.time()
    try:
        rounds = run_one(item)
    except Exception as e:
        rounds = [{"round": 0, "error": f"top-level: {e}"}]
    flags = flag(rounds)
    rec = {"id": item["id"], "domain": item["domain"], "tools": item.get("tools"),
           "prompt": item["prompt"], "ground_truth": item.get("ground_truth"),
           "n_rounds": len(rounds), "flags": flags, "rounds": rounds,
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
        futs = {pool.submit(process_one, item): item for item in todo}
        for fut in as_completed(futs):
            item = futs[fut]
            rec = fut.result()
            with WRITE_LOCK:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}/{len(todo)}] {item['id']} rounds={rec['n_rounds']} "
                      f"flags={rec['flags']} ({rec['elapsed_s']}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
