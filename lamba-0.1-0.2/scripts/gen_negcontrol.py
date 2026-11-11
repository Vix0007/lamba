#!/usr/bin/env python3
"""Pin down whether the #154 regression is specific to INTEGRALS or the broader
'compute exact value, then explicitly verify that SAME value via an independent method'
pattern. 8 new tasks (limit, matrix inverse, factoring, trig identity, sum-of-squares,
probability, geometric mean, quadratic solve) x5 reps per arm. Includes the comma-formatting
fix found while analyzing the previous (broad_verify) check.
"""
import json
import re
import sys
import time

sys.path.insert(0, "/home/ubuntu/antidoom_data")
from gen_a_fresh_toolloop import run_conversation  # noqa: E402

N_REPS = 5


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


def strip_commas(text):
    return re.sub(r"(?<=\d),(?=\d)", "", text or "")


def contains_value(text, needles):
    if not text:
        return False
    t2 = strip_commas(text)
    return any(n in text or n in t2 for n in needles)


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
        print("usage: gen_negcontrol.py <served_model_name> <out.jsonl>")
        sys.exit(1)
    served_model_name, out_path = sys.argv[1], sys.argv[2]

    prompts = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/negcontrol_prompts.jsonl")]

    done = set()
    try:
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["id"], r["rep"]))
    except FileNotFoundError:
        pass

    with open(out_path, "a") as f:
        for p in prompts:
            needles = all_needles(p["ground_truth_values"])
            for rep in range(N_REPS):
                if (p["id"], rep) in done:
                    continue
                t0 = time.time()
                try:
                    result = run_conversation(p["prompt"], served_model_name)
                    result["error"] = None
                except Exception as e:
                    result = {"final_text": "", "n_rounds": 0, "budget_hit": False,
                              "transcript": [], "error": str(e)}
                collapsed = len((result.get("final_text") or "").strip()) < 20
                cls = classify(result, needles) if not result.get("error") else "error"
                row = {"id": p["id"], "rep": rep, "elapsed_s": round(time.time() - t0, 1),
                       "collapsed": collapsed, "classification": cls, **result}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{p['id']} rep={rep}: collapsed={collapsed} class={cls} "
                      f"rounds={row['n_rounds']} elapsed={row['elapsed_s']}s", flush=True)
    print("NEGCONTROL_DONE")


if __name__ == "__main__":
    main()
