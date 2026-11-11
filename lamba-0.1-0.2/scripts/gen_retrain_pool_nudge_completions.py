#!/usr/bin/env python3
"""For every row in retrain_pool_s1337.jsonl with guard_fired=True, replay its REAL
transcript up to the fire point (genuine tool_call ids/arguments, exactly as generated),
inject the real nudge, and make ONE real forced-final call. This produces the (shadow,
nudge) pair data needed for DPO mining -- same validated pattern as
stress_test_v6_ablation.py, applied systematically to all naturally-fired rollouts
instead of 2 hand-picked ones. Resumable via (id, rep) already in the output.
"""
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/web")
sys.path.insert(0, "/home/ubuntu/antidoom_data")
import server as srv  # noqa: E402
from rumination_guard import nudge_intervention_messages  # noqa: E402

BACKEND = "http://127.0.0.1:8001"
WRITE_LOCK = threading.Lock()


def post_backend(body, timeout=120):
    req = urllib.request.Request(f"{BACKEND}/v1/chat/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def build_prefix_to_fire_point(prompt_text, transcript, fire_round):
    """Reuses the REAL tool_call ids/arguments from the transcript, truncated to the
    tool result at fire_round (0-indexed among tool rounds), matching GuardState's own
    round-counting convention exactly."""
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    tool_round_seen = -1
    for turn in transcript:
        role = turn.get("role")
        if role == "assistant" and turn.get("tool_calls"):
            messages.append({"role": "assistant", "content": turn.get("content") or "",
                              "tool_calls": turn["tool_calls"]})
        elif role == "tool":
            messages.append({"role": "tool", "tool_call_id": turn.get("tool_call_id", ""),
                              "content": turn.get("content", "")})
            tool_round_seen += 1
            if tool_round_seen == fire_round:
                break
    return messages


def main():
    served_model_name = sys.argv[1]
    src_path = sys.argv[2]
    prompts_path = sys.argv[3]
    out_path = sys.argv[4]

    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(prompts_path)}
    done = set()
    try:
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["id"], r["rep"]))
    except FileNotFoundError:
        pass

    rows = [json.loads(l) for l in open(src_path)]
    fired_rows = [r for r in rows if r.get("guard_fired") and (r["id"], r["rep"]) not in done]
    concurrency = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    print(f"{len(fired_rows)} fired rows to nudge-complete (of {len(rows)} total, {len(done)} "
          f"already done, concurrency={concurrency})", flush=True)

    def run_one(r):
        prompt_text = prompts.get(r["id"])
        if prompt_text is None:
            return None
        # need tool_call_id on the original transcript's "tool" turns -- the real
        # generator (gen_retrain_pool_rollouts.py) stored them under a different key
        # name via srv.run_tool's caller; reconstruct via the assistant tool_calls'
        # own ids, which are already 1:1 with the tool turns in order.
        transcript = r["transcript"]
        last_ids = []
        for turn in transcript:
            if turn.get("role") == "assistant" and turn.get("tool_calls"):
                last_ids = [tc["id"] for tc in turn["tool_calls"]]
            elif turn.get("role") == "tool" and "tool_call_id" not in turn:
                turn["tool_call_id"] = last_ids.pop(0) if last_ids else ""

        messages = build_prefix_to_fire_point(prompt_text, transcript, r["guard_fire_round"])
        messages.extend(nudge_intervention_messages())
        body = {"model": served_model_name, "tools": srv.TOOLS, "tool_choice": "none",
                "temperature": 0.0, "max_tokens": 4000, "messages": messages}
        t0 = time.time()
        try:
            resp = post_backend(body)
            msg = resp["choices"][0]["message"]
            row = {"id": r["id"], "rep": r["rep"], "gt": r.get("ground_truth_numeric"),
                   "integrand_class": r.get("integrand_class"),
                   "fire_round": r["guard_fire_round"], "fire_reason": r["guard_fire_reason"],
                   "shadow_final_text": r.get("final_text", ""),
                   "shadow_classification": r.get("classification"),
                   "nudge_final_text": msg.get("content") or "",
                   "nudge_reasoning_len": len(msg.get("reasoning") or ""),
                   "error": None, "elapsed_s": round(time.time() - t0, 1)}
        except Exception as e:
            row = {"id": r["id"], "rep": r["rep"], "error": str(e)[:300],
                   "nudge_final_text": "", "elapsed_s": round(time.time() - t0, 1)}
        return row

    n_done = 0
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(run_one, r): r for r in fired_rows}
        for fut in as_completed(futs):
            r = futs[fut]
            row = fut.result()
            if row is None:
                continue
            with WRITE_LOCK:
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}/{len(fired_rows)}] {r['id']} rep={r['rep']}: "
                      f"nudge_empty={not row.get('nudge_final_text','').strip()} "
                      f"({row['elapsed_s']}s)", flush=True)
    print("NUDGE_COMPLETIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
