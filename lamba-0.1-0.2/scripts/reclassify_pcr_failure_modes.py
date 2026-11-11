#!/usr/bin/env python3
"""DS #195 diagnostic #1: reclassify all 192 draws (both arms, 12 HO tasks x 8 reps) into
mutually exclusive failure-mode categories, since the sandbox is confirmed STATELESS
(diagnostic #0, confirmed via direct code inspection of web/server.py's run_python_sandboxed
-- every python_eval call is a brand-new subprocess, no persisted namespace).

Categories (DS's schema):
  success: final contains correct value
  true_rumination: correct value reached, then extra tool calls that are NOT sandbox-state
    errors (i.e. the model re-verifies/re-derives without an external cause), final missing
    value
  sandbox_state_failure: correct value reached, then at least one subsequent tool result
    contains a NameError/ImportError/"is not defined"/"__import__ not found" pattern
    (model assumed persisted state that the stateless sandbox doesn't provide), final
    missing value
  formatting_empty_final: correct value reached, ZERO extra tool calls, but final text is
    empty/whitespace-only or doesn't contain the value (a pure output-formatting miss, not
    a reasoning continuation issue)
  never_correct: no tool result or final ever contains the correct value
  other: doesn't cleanly fit the above (e.g. correct value only in final, no tool trace)
"""
import json
import re
from collections import defaultdict

SANDBOX_ERR_RE = re.compile(
    r"NameError|ImportError|__import__ not found|is not defined|ModuleNotFoundError",
    re.I,
)


def strip_commas(s):
    return re.sub(r"(?<=\d),(?=\d)", "", s)


def contains_value(text, needles):
    if not text:
        return False
    t2 = strip_commas(text)
    return any(n in text or n in t2 for n in needles)


def reclassify(transcript, final_text, needles):
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
        return "never_correct" if not has_value_in_final else "other"

    post_turns = transcript[correct_round + 1:]
    extra_tool_calls = sum(
        1 for turn in post_turns if turn.get("role") == "assistant" and turn.get("tool_calls")
    )
    post_tool_results = [turn.get("content") or "" for turn in post_turns if turn.get("role") == "tool"]
    has_sandbox_err = any(SANDBOX_ERR_RE.search(r) for r in post_tool_results)

    if extra_tool_calls == 0:
        if has_value_in_final:
            return "success"
        else:
            return "formatting_empty_final"
    # extra_tool_calls >= 1
    if has_value_in_final:
        return "success_but_overran"  # not in DS's 6, but keep separate from true success
    if has_sandbox_err:
        return "sandbox_state_failure"
    return "true_rumination"


def load_needles_and_prompts():
    prompts = {}
    for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
        r = json.loads(l)
        prompts[r["id"]] = r["ground_truth_numeric"]
    return prompts


import sys
sys.path.insert(0, "/home/ubuntu/antidoom_data")
from gen_retrain_pool_rollouts import numeric_str_variants  # the REAL needle logic, not a guess


def needles_for(gt):
    return list(numeric_str_variants(gt))


def main():
    gts = load_needles_and_prompts()
    results = defaultdict(lambda: defaultdict(int))  # arm -> category -> count
    per_task = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # arm->task->cat->count

    for arm, fn in [("adapter", "/home/ubuntu/eval_results/pcr_heldout_dpov1.jsonl"),
                     ("c10", "/home/ubuntu/eval_results/pcr_heldout_c10.jsonl")]:
        for line in open(fn):
            r = json.loads(line)
            tid = r["id"]
            gt = gts[tid]
            needles = needles_for(gt)
            cat = reclassify(r["transcript"], r.get("final_text", ""), needles)
            results[arm][cat] += 1
            per_task[arm][tid][cat] += 1

    print("=== Pooled failure-mode distribution (96 draws/arm) ===")
    all_cats = ["success", "success_but_overran", "true_rumination", "sandbox_state_failure",
                "formatting_empty_final", "never_correct", "other"]
    for arm in ("adapter", "c10"):
        print(f"\n{arm}:")
        for cat in all_cats:
            n = results[arm][cat]
            print(f"  {cat:25s} {n:3d}/96 = {n/96:.1%}")

    print()
    print("=== THE KEY NUMBER: true_rumination rate (the actual intended target) ===")
    for arm in ("adapter", "c10"):
        n = results[arm]["true_rumination"]
        print(f"{arm}: {n}/96 = {n/96:.1%}")

    print()
    print("=== sandbox_state_failure rate (infrastructure artifact, not model rumination) ===")
    for arm in ("adapter", "c10"):
        n = results[arm]["sandbox_state_failure"]
        print(f"{arm}: {n}/96 = {n/96:.1%}")

    print()
    print("=== Per-task breakdown ===")
    for tid in sorted(per_task["adapter"].keys()):
        print(f"\n{tid}:")
        for arm in ("adapter", "c10"):
            cats = per_task[arm][tid]
            s = ", ".join(f"{c}={cats.get(c,0)}" for c in all_cats if cats.get(c, 0) > 0)
            print(f"  {arm}: {s}")


if __name__ == "__main__":
    main()
