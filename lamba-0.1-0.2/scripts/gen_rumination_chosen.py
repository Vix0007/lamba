#!/usr/bin/env python3
"""Rumination DPO, "chosen" side. For every sandbox-B rollout that reached the DONE point
(rumination_metric_v2: exact value + a successful independent check) and then made >=1
redundant tool call, rebuild the EXACT chat prefix up to and including the done-point tool
result, and generate the model's own final answer from that same prefix with tool calls
disabled (tool_choice="none"). No nudge/guard text is injected into the context -- the only
difference from the real rollout is the decoding constraint.

  rejected = the rollout's actual next turn after the done point (its first redundant call)
  chosen   = this script's answer from the identical prefix (correctness checked downstream)

Parity with generation time: same system prompt, same tools schema, same chat_template_kwargs
(the old nudge-completion script omitted chat_template_kwargs -- fixed here), same temp/top_p.
Usage: gen_rumination_chosen.py <served_model> <rollouts.jsonl> <prompts.jsonl[,more.jsonl]> <out.jsonl> [concurrency]
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402
from gen_a_fresh_toolloop import post_backend  # noqa: E402
from rumination_metric_v2 import analyze  # noqa: E402


def attach_tool_call_ids(transcript):
    """Raw transcripts don't store tool_call_id on tool turns; ids are 1:1 in order with the
    preceding assistant turn's tool_calls (same reconstruction verified for the nudge data)."""
    pending = []
    for turn in transcript:
        if turn.get("role") == "assistant" and turn.get("tool_calls"):
            pending = [tc.get("id", "") for tc in turn["tool_calls"]]
        elif turn.get("role") == "tool" and "tool_call_id" not in turn:
            turn["tool_call_id"] = pending.pop(0) if pending else ""
    return transcript


def prefix_messages(prompt, transcript, done_idx):
    msgs = [{"role": "system", "content": srv.build_system_prompt()},
            {"role": "user", "content": prompt}]
    for turn in transcript[:done_idx + 1]:
        if turn.get("role") == "assistant" and turn.get("tool_calls"):
            msgs.append({"role": "assistant", "content": turn.get("content") or "",
                         "tool_calls": turn["tool_calls"]})
        elif turn.get("role") == "tool":
            msgs.append({"role": "tool", "tool_call_id": turn.get("tool_call_id", ""),
                         "content": turn.get("content", "")})
    return msgs


def main():
    model, src, prompt_files, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    conc = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    tasks = {}
    for pf in prompt_files.split(","):
        for l in open(pf):
            t = json.loads(l)
            tasks[t["id"]] = t
    done_keys = set()
    try:
        for l in open(out):
            r = json.loads(l)
            done_keys.add((r["id"], r["rep"]))
    except FileNotFoundError:
        pass

    todo = []
    for l in open(src):
        r = json.loads(l)
        t = tasks.get(r["id"])
        if t is None or (r["id"], r["rep"]) in done_keys:
            continue
        a = analyze(r, t["ground_truth_numeric"], t.get("family", "integral"))
        if a["done"] and a["redundant"] >= 1:
            todo.append((r, t, a["done_idx"]))
    print(f"{len(todo)} ruminating rollouts to generate chosen for", flush=True)
    lock = threading.Lock()

    def run_one(r, t, done_idx):
        tr = attach_tool_call_ids(r["transcript"])
        msgs = prefix_messages(t["prompt"], tr, done_idx)
        body = {"model": model, "messages": msgs, "tools": srv.TOOLS, "tool_choice": "none",
                "temperature": 0.8, "top_p": 0.95, "seed": r["rep"], "max_tokens": 4000,
                "chat_template_kwargs": srv.DEFAULT_CHAT_TEMPLATE_KWARGS}
        t0 = time.time()
        try:
            msg = post_backend(body)["choices"][0]["message"]
            row = {"id": r["id"], "rep": r["rep"], "done_idx": done_idx, "family": t.get("family"),
                   "chosen_content": msg.get("content") or "", "chosen_reasoning": msg.get("reasoning") or "",
                   "error": None, "elapsed_s": round(time.time() - t0, 1)}
        except Exception as e:
            row = {"id": r["id"], "rep": r["rep"], "error": str(e)[:300]}
        with lock, open(out, "a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(run_one, *x) for x in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)
    print("GEN_RUMINATION_CHOSEN_DONE", flush=True)


if __name__ == "__main__":
    main()
