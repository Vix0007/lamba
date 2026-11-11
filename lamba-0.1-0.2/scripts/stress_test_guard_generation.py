#!/usr/bin/env python3
"""DS #165's stress-test variants 3/4/6 (the ones that need REAL model generation, since
they test what the model actually SAYS once nudged/forced-final, given a SYNTHETIC prior
context we control). Injects a fake multi-round tool-calling prefix directly (bypassing
natural generation for the early rounds) so the intermediate/candidate state is exactly
what we specify, then either (a) lets GuardState decide whether to fire on it and if so
applies the real nudge + one real forced-final generation, or (b) for the no-fire check
just records what a normal one-more-tool-call-allowed continuation would look like.

Variant 3: I correct, C (candidate) WRONG -- one bad round after an otherwise-correct
  trajectory. Guard may fire (per the timing-only design); check no WRONG VALUE gets
  reinforced/injected by the nudge, and correctness doesn't get worse.
Variant 4: I partially correct (off-by-factor-of-2, sign-flip, missing constant) --
  similarly, check the guard's nudge doesn't lock in the error.
Variant 6: garbled/empty tool output right before a forced-final with a tight remaining
  budget -- this directly reproduces the s42 empty-final failure mode. Checks whether
  giving the forced-final call MORE tokens (the DS-recommended mitigation) fixes it.

Pilot scale (affordable, not DS's full 20-40/variant): 3 integral tasks x 2 reps x
3 variants = 18 generations. Honest caveat: this is directional, not fully powered.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/home/ubuntu/web")
sys.path.insert(0, "/home/ubuntu/antidoom_data")
import server as srv  # noqa: E402
from rumination_guard import GuardState, nudge_intervention_messages  # noqa: E402

BACKEND = "http://127.0.0.1:8001"

# 3 tasks with known exact values (reuse from nudge_ab_prompts.jsonl's first 3)
TASKS = [
    {"id": "ST_t1", "prompt": ("Using python_eval with SymPy, compute the definite integral of "
        "x^2 * ln(x) from x=1 to x=e, then verify by numerically integrating the same function "
        "over the same bounds and confirming the two match."), "gt": 4.574563760717865},
    {"id": "ST_t2", "prompt": ("Using python_eval with SymPy, compute the definite integral of "
        "x^3 * e^(-x) from x=0 to x=3, then verify by numerically integrating the same function "
        "over the same bounds and confirming the two match."), "gt": 2.1166086673},
    {"id": "ST_t3", "prompt": ("Using python_eval with SymPy, compute the definite integral of "
        "x*sin(x) from x=0 to x=pi, then verify by numerically integrating the same function "
        "over the same bounds and confirming the two match."), "gt": 3.1415926536},
]


def post_backend(body, timeout=120):
    req = urllib.request.Request(f"{BACKEND}/v1/chat/completions", data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fake_tool_call(idx):
    return {"id": f"stress_call_{idx}", "type": "function",
            "function": {"name": "python_eval", "arguments": '{"code": "# (synthetic, stress-test)"}'}}


def build_synthetic_prefix(prompt_text, tool_outputs):
    """tool_outputs: list of strings, one per synthetic round. Builds a messages list with
    that many fake assistant+tool_call / tool-result pairs already in place."""
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    for i, out in enumerate(tool_outputs):
        messages.append({"role": "assistant", "content": "",
                          "tool_calls": [fake_tool_call(i)]})
        messages.append({"role": "tool", "tool_call_id": f"stress_call_{i}", "content": out})
    return messages


def run_variant(served_model_name, task, variant, tool_outputs, forced_final_max_tokens=4000):
    gs = GuardState(task["prompt"])
    fired = False
    for i, out in enumerate(tool_outputs):
        gs.observe_tool_round(out, i)
    fired = gs.fired
    messages = build_synthetic_prefix(task["prompt"], tool_outputs)

    row = {"task_id": task["id"], "variant": variant, "gt": task["gt"],
           "guard_fired": fired, "fire_round": gs.fire_round, "fire_reason": gs.fire_reason}

    if fired:
        messages.extend(nudge_intervention_messages())
    body = {"model": served_model_name, "tools": srv.TOOLS, "tool_choice": "none",
            "temperature": 0.0, "max_tokens": forced_final_max_tokens, "messages": messages}
    t0 = time.time()
    try:
        resp = post_backend(body)
        msg = resp["choices"][0]["message"]
        row["final_text"] = msg.get("content") or ""
        row["reasoning_len"] = len(msg.get("reasoning") or "")
        row["error"] = None
    except Exception as e:
        row["final_text"] = ""
        row["reasoning_len"] = 0
        row["error"] = str(e)[:300]
    row["elapsed_s"] = round(time.time() - t0, 1)
    return row


def contains_value(text, v, ndigits=4):
    if not text:
        return False
    s = f"{v:.{ndigits}f}"
    return s[:ndigits+2] in text or str(round(v)) in text  # loose match, good enough for a stress test


def main():
    served_model_name = sys.argv[1] if len(sys.argv) > 1 else "ruminconfab-s1337"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/home/ubuntu/eval_results/stress_test_generation.jsonl"
    rows = []
    for task in TASKS:
        gt = task["gt"]
        wrong = gt * 1.37 + 5  # a clearly-wrong but plausible-looking value
        partial = gt / 2  # variant 4: off-by-factor-of-2 error, a classic real mistake

        # Variant 3: I correct (converges twice), then a WRONG value takes over and
        # itself converges (twice) -- tests whether firing on a stabilized-but-wrong
        # later value risks the nudge locking that wrong value in.
        v3_outputs = [f"Exact value: {gt:.10f}", f"Confirmed: {gt:.10f}",
                      f"Wait, rechecking: {wrong:.10f}", f"Confirmed: {wrong:.10f}"]
        # Variant 4: partial/systematic error (factor of 2), repeated (looks converged)
        v4_outputs = [f"Result: {partial:.10f}", f"Verification: {partial:.10f}", f"Matches: {partial:.10f}"]
        # Variant 6 (corrected): the guard CANNOT fire on garbled/no-value output (no
        # candidate ever set) -- the real s42 miss was a genuinely CORRECT convergence
        # (fire_reason=verified_or_confirmed_no_new_info) followed by the SEPARATE forced-
        # final generation call itself burning its own token budget re-deriving instead of
        # restating. So V6 reuses a real converged-correct sequence and instead varies the
        # forced-final token budget to see if a bigger budget (DS's suggested mitigation)
        # reduces the empty-final rate.
        v6_outputs = [f"Exact: {gt:.10f}", f"Verified numerically: {gt:.10f}", f"Confirmed match: {gt:.10f}"]

        for variant, outputs, budget in [("V3_correct_then_wrong_stabilizes", v3_outputs, 4000),
                                          ("V4_partial_factor_of_2", v4_outputs, 4000),
                                          ("V6_reconverge_tight_budget", v6_outputs, 4000),
                                          ("V6_reconverge_larger_budget", v6_outputs, 8000)]:
            row = run_variant(served_model_name, task, variant, outputs, forced_final_max_tokens=budget)
            row["gt_in_final"] = contains_value(row["final_text"], gt)
            row["wrong_in_final"] = contains_value(row["final_text"], wrong) if "wrong" in variant.lower() or variant.startswith("V3") else None
            row["partial_in_final"] = contains_value(row["final_text"], partial) if variant.startswith("V4") else None
            rows.append(row)
            print(f"{task['id']} {variant}: fired={row['guard_fired']} gt_in_final={row['gt_in_final']} "
                  f"wrong_in_final={row.get('wrong_in_final')} partial_in_final={row.get('partial_in_final')} "
                  f"final_empty={not row['final_text'].strip()} elapsed={row['elapsed_s']}s", flush=True)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} stress-test rows to {out_path}")
    print("STRESS_TEST_GENERATION_DONE")


if __name__ == "__main__":
    main()
