#!/usr/bin/env python3
"""DS #198 Condition A: restricted sandbox (unchanged), round cap 16 instead of 8,
same prompts/seeds, re-running ONLY the 43 original never_correct (id,rep,arm) cases
for tight pairing. Per-round max_tokens stays 4000 (confirmed independent of round
count, not a shared total budget -- resolves DS's stated confound concern).
Usage: gen_rescue_condition_a.py <served_model_name_for_this_arm> <arm_filter> <out.jsonl>
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402
from gen_a_fresh_toolloop import post_backend  # noqa: E402
from gen_retrain_pool_rollouts import numeric_str_variants  # noqa: E402

ROUND_CAP_A = 16


def run_conversation_round_cap(prompt_text, served_model_name, temperature, top_p, seed, round_cap):
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    base_body = {
        "model": served_model_name, "tools": srv.TOOLS, "tool_choice": "auto",
        "temperature": temperature, "top_p": top_p, "seed": seed,
        "max_tokens": 4000, "chat_template_kwargs": srv.DEFAULT_CHAT_TEMPLATE_KWARGS,
    }
    transcript = []
    n_rounds = 0
    for _ in range(round_cap):
        body = dict(base_body)
        body["messages"] = messages
        body["stream"] = False
        resp = post_backend(body)
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
            result = srv.run_tool(fn.get("name", ""), fn.get("arguments", "{}"))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
            transcript.append({"role": "tool", "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", ""), "content": result})
    body = dict(base_body)
    body["messages"] = messages
    body["stream"] = False
    body["tool_choice"] = "none"
    resp = post_backend(body)
    msg = resp["choices"][0]["message"]
    transcript.append({"role": "assistant", "content": msg.get("content") or "",
                        "reasoning": msg.get("reasoning") or ""})
    return {"final_text": msg.get("content") or "", "n_rounds": round_cap,
            "budget_hit": True, "transcript": transcript}


def main():
    served_model_name = sys.argv[1]
    arm_filter = sys.argv[2]  # "adapter" or "c10"
    out_path = sys.argv[3]
    concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    cases = json.load(open("/home/ubuntu/antidoom_data/never_correct_cases.json"))
    cases = [c for c in cases if c["arm"] == arm_filter]

    prompts = {}
    for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
        r = json.loads(l)
        prompts[r["id"]] = r["prompt"]

    done = set()
    try:
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["id"], r["rep"]))
    except FileNotFoundError:
        pass

    todo = [c for c in cases if (c["id"], c["rep"]) not in done]
    print(f"[{arm_filter}] {len(todo)}/{len(cases)} to do (round_cap={ROUND_CAP_A}), {len(done)} already done", flush=True)

    lock = threading.Lock()

    def run_one(case):
        tid, rep = case["id"], case["rep"]
        result = run_conversation_round_cap(
            prompts[tid], served_model_name, temperature=0.8, top_p=0.95, seed=rep,
            round_cap=ROUND_CAP_A,
        )
        row = {"id": tid, "rep": rep, "arm": arm_filter, "round_cap": ROUND_CAP_A, **result}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return tid, rep

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(run_one, c): c for c in todo}
        n = 0
        for fut in as_completed(futs):
            tid, rep = fut.result()
            n += 1
            print(f"[{n}/{len(todo)}] {tid} rep={rep} done", flush=True)

    print(f"CONDITION_A_{arm_filter.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
