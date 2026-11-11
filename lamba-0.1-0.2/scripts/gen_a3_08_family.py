#!/usr/bin/env python3
"""A3_08 family check (DS consult #153's targeted follow-up): the original Aexp_A3_08
integral-reconciliation prompt run to n=25 (20 more reps added to the 5 already collected),
plus 6 sympy-verified paraphrased variants of the same task shape x5 reps each, per arm.
Endpoint (DS's exact definition): post-correct rumination = the correct numeric value appears
in a tool-call result, then >=1 additional tool call happens, and the final answer is empty or
does not contain the value.
"""
import json
import re
import sys
import time

sys.path.insert(0, "/home/ubuntu/antidoom_data")
from gen_a_fresh_toolloop import run_conversation  # noqa: E402

ORIG_ID = "Aexp_A3_08"
ORIG_PROMPT = ("Using python_eval with SymPy, compute the definite integral of x^2 * ln(x) "
               "from x=1 to x=e, then verify by numerically integrating the same function "
               "over the same bounds (e.g. via a fine Riemann sum or sympy's numerical "
               "integration) and confirming the two match.")
ORIG_GT = "4.574563760717865"
ORIG_EXTRA_REPS = 20  # rep indices 5..24, on top of the 5 (0..4) already in a_repro_*.jsonl


def numeric_str_variants(v, ndigits=6):
    """A handful of plausible printed forms of a float, to search tool output for."""
    s = f"{v:.{ndigits}f}"
    # also allow shorter truncations (model/tool may print fewer digits)
    out = {s}
    for n in range(1, ndigits):  # BUG FIX 2026-10-01: was range(2,..), missed clean 1-decimal values
        out.add(f"{v:.{n}f}")
    return out


def contains_value(text, needles):
    if not text:
        return False
    return any(n in text for n in needles)


def classify(result, needles):
    t = result.get("transcript", [])
    ft = (result.get("final_text") or "").strip()
    correct_round = None
    for i, turn in enumerate(t):
        if turn.get("role") == "tool":
            c = turn.get("content") or ""
            if contains_value(c, needles):
                correct_round = i
                break
    has_value_in_final = contains_value(ft, needles)
    if correct_round is None:
        return "never_correct" if not has_value_in_final else "correct_only_in_final"
    extra_tool_calls = sum(
        1 for turn in t[correct_round + 1:]
        if turn.get("role") == "assistant" and turn.get("tool_calls")
    )
    if extra_tool_calls == 0 and has_value_in_final:
        return "correct_and_stopped"
    if extra_tool_calls >= 1 and not has_value_in_final:
        return "post_correct_rumination"
    if extra_tool_calls >= 1 and has_value_in_final:
        return "correct_but_overran_then_answered"
    return "other"


def main():
    if len(sys.argv) != 3:
        print("usage: gen_a3_08_family.py <served_model_name> <out.jsonl>")
        sys.exit(1)
    served_model_name, out_path = sys.argv[1], sys.argv[2]

    variants = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/a3_08_variants.jsonl")]

    tasks = []  # (id, prompt, needles, rep)
    orig_needles = numeric_str_variants(float(ORIG_GT))
    for rep in range(5, 5 + ORIG_EXTRA_REPS):
        tasks.append((ORIG_ID, ORIG_PROMPT, orig_needles, rep))
    for v in variants:
        needles = numeric_str_variants(v["ground_truth_numeric"])
        for rep in range(5):
            tasks.append((v["id"], v["prompt"], needles, rep))

    done = set()
    try:
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["id"], r["rep"]))
    except FileNotFoundError:
        pass

    with open(out_path, "a") as f:
        for pid, prompt, needles, rep in tasks:
            if (pid, rep) in done:
                continue
            t0 = time.time()
            try:
                result = run_conversation(prompt, served_model_name)
                result["error"] = None
            except Exception as e:
                result = {"final_text": "", "n_rounds": 0, "budget_hit": False,
                          "transcript": [], "error": str(e)}
            collapsed = len((result.get("final_text") or "").strip()) < 20
            cls = classify(result, needles) if not result.get("error") else "error"
            row = {"id": pid, "rep": rep, "elapsed_s": round(time.time() - t0, 1),
                   "collapsed": collapsed, "classification": cls, **result}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{pid} rep={rep}: collapsed={collapsed} class={cls} "
                  f"rounds={row['n_rounds']} elapsed={row['elapsed_s']}s", flush=True)
    print("A3_08_FAMILY_DONE")


if __name__ == "__main__":
    main()
