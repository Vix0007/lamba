#!/usr/bin/env python3
"""Real (organic, not synthetic) rollout generation on the 52-task expanded integral
pool, on s1337. Reuses the exact real tool-loop (srv.run_tool/TOOLS/build_system_prompt)
and the classify()/needle logic already validated tonight (matches
gen_matched_iterative.py exactly). This mines a real, diverse pool of fire events toward
DS's >=50-task/>=300-pair retrain sizing. Resumable via (id, rep) already in the output.
Usage: gen_retrain_pool_rollouts.py <served_model_name> <out.jsonl> <n_reps>
"""
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402
from gen_a_fresh_toolloop import post_backend, MAX_TOOL_ROUNDS  # noqa: E402
from rumination_guard import GuardState  # noqa: E402


def run_conversation_sampled(prompt_text, served_model_name, temperature, top_p, seed):
    """Same loop as gen_a_fresh_toolloop.run_conversation, but with sampling params
    exposed -- per DS #168, mining rollouts need temp=0.8/top_p=0.95 for rep diversity,
    not greedy (which would make every rep identical and waste the reps)."""
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
    for _ in range(MAX_TOOL_ROUNDS):
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
    transcript.append({"role": "assistant_forced_final", "content": msg.get("content") or "",
                        "reasoning": msg.get("reasoning") or ""})
    return {"final_text": msg.get("content") or "", "n_rounds": MAX_TOOL_ROUNDS,
            "budget_hit": True, "transcript": transcript}

WRITE_LOCK = threading.Lock()
NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def strip_commas(t):
    return re.sub(r"(?<=\d),(?=\d)", "", t or "")


def numeric_str_variants(v, ndigits=6):
    s = f"{v:.{ndigits}f}"
    out = {s}
    # BUG FIX (2026-10-01, found via HO_p1 false-negative audit): the 1-decimal
    # representation (e.g. "2.5") was previously only added when v was near an
    # INTEGER, so any clean 1-decimal ground truth (2.5, 7.5, 0.5, ...) never got
    # its natural short form in the needle set -- a model correctly writing "2.5"
    # (no trailing-zero padding) was invisible to every classifier built on this
    # function. Now always emit every n-decimal truncation from 1..ndigits-1,
    # not gated on near-integer-ness.
    for n in range(1, ndigits):
        out.add(f"{v:.{n}f}")
    if abs(v - round(v)) < 1e-9:
        iv = int(round(v))
        out.add(str(iv))
        out.add(f"{iv:,}")
    return out


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


def run_one(served_model_name, item, rep):
    values = [item["ground_truth_numeric"]]
    needles = numeric_str_variants(values[0])
    t0 = time.time()
    try:
        result = run_conversation_sampled(item["prompt"], served_model_name,
                                           temperature=0.8, top_p=0.95, seed=rep)
        cls = classify(result["transcript"], result["final_text"], needles)
        # also replay the guard offline on this real transcript, to tag which fire
        # pathway (if any) would have triggered, for downstream mining
        gs = GuardState(item["prompt"])
        tool_round = -1
        for turn in result["transcript"]:
            if turn.get("role") == "tool":
                tool_round += 1
                gs.observe_tool_round(turn.get("content") or "", tool_round)
        row = {"id": item["id"], "rep": rep, "classification": cls, "error": None,
               "integrand_class": item.get("integrand_class"),
               "ground_truth_numeric": item["ground_truth_numeric"],
               "guard_fired": gs.fired, "guard_fire_round": gs.fire_round,
               "guard_fire_reason": gs.fire_reason,
               "elapsed_s": round(time.time() - t0, 1), **result}
    except Exception as e:
        row = {"id": item["id"], "rep": rep, "classification": "error", "error": str(e)[:500],
               "final_text": "", "n_rounds": 0, "budget_hit": False, "transcript": [],
               "elapsed_s": round(time.time() - t0, 1)}
    return row


def main():
    served_model_name = sys.argv[1]
    out_path = sys.argv[2]
    n_reps = int(sys.argv[3])
    concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    prompts_path = sys.argv[5] if len(sys.argv) > 5 else "/home/ubuntu/antidoom_data/retrain_pool_integrals.jsonl"

    items = [json.loads(l) for l in open(prompts_path)]
    done = set()
    try:
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["id"], r["rep"]))
    except FileNotFoundError:
        pass

    tasks = [(item, rep) for item in items for rep in range(n_reps) if (item["id"], rep) not in done]
    print(f"model={served_model_name} {len(items)} tasks x {n_reps} reps, "
          f"{len(done)} done, {len(tasks)} to go (concurrency={concurrency})", flush=True)

    n_done = 0
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {pool.submit(run_one, served_model_name, item, rep): (item, rep) for item, rep in tasks}
        for fut in as_completed(futs):
            item, rep = futs[fut]
            row = fut.result()
            with WRITE_LOCK:
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}/{len(tasks)}] {item['id']} rep={rep}: class={row['classification']} "
                      f"guard_fired={row.get('guard_fired')} rounds={row['n_rounds']} "
                      f"({row['elapsed_s']}s)", flush=True)
    print("RETRAIN_POOL_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
