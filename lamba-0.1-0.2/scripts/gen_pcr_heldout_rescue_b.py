#!/usr/bin/env python3
"""Quick combined check: does the sandbox-B fix also rescue the ADAPTER's own 7
never_correct cases? Reuses the same pattern as gen_rescue_condition_b.py but
targets whichever arm's never_correct cases are requested.
Usage: gen_pcr_heldout_rescue_b.py <served_model_name> <out.jsonl> <reps> <concurrency>
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402
from gen_a_fresh_toolloop import post_backend  # noqa: E402
from run_tool_rescue_b_dpov1 import run_tool_rescue_b  # noqa: E402

ROUND_CAP = 8


def run_conversation_rescue_b(prompt_text, served_model_name, temperature, top_p, seed):
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
    for _ in range(ROUND_CAP):
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
            result = run_tool_rescue_b(fn.get("name", ""), fn.get("arguments", "{}"))
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
    return {"final_text": msg.get("content") or "", "n_rounds": ROUND_CAP,
            "budget_hit": True, "transcript": transcript}


def main():
    served_model_name = sys.argv[1]
    out_path = sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    cases = json.load(open("/home/ubuntu/antidoom_data/never_correct_cases.json"))
    cases = [c for c in cases if c["arm"] == "adapter"]  # only the adapter's own 7

    prompts = {}
    for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
        r = json.loads(l)
        prompts[r["id"]] = r["prompt"]

    todo = [(c, rep) for c in cases for rep in range(reps)]
    print(f"running {len(todo)} draws (7 adapter never_correct cases x {reps} reps)", flush=True)

    lock = threading.Lock()

    def run_one(case, rep):
        tid = case["id"]
        result = run_conversation_rescue_b(prompts[tid], served_model_name, temperature=0.8, top_p=0.95, seed=rep)
        row = {"id": tid, "rep": rep, "arm": "adapter", "condition": "B", **result}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return tid, rep

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(run_one, c, rep) for c, rep in todo]
        n = 0
        for fut in as_completed(futs):
            tid, rep = fut.result()
            n += 1
            print(f"[{n}/{len(todo)}] {tid} rep={rep} done", flush=True)

    print("GEN_PCR_RESCUE_B_DONE", flush=True)


if __name__ == "__main__":
    main()
