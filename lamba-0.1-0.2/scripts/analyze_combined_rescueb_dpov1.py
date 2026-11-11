#!/usr/bin/env python3
"""Analyze the combined test: dpov1 adapter + sandbox-B fix, on the adapter's own 7
never_correct cases (56 draws = 7 cases x 8 reps). Reuses the real reclassify() logic
and real needle generation, same as every other condition in this investigation.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, "/home/ubuntu/antidoom_data")
from reclassify_pcr_failure_modes import reclassify, needles_for

# ground truth per id
gts = {}
for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
    r = json.loads(l)
    gts[r["id"]] = r["ground_truth_numeric"]

cats = defaultdict(int)
per_task = defaultdict(lambda: defaultdict(int))
rounds_by_outcome = defaultdict(list)

fn = "/home/ubuntu/eval_results/pcr_heldout_dpov1_rescueb.jsonl"
n = 0
for line in open(fn):
    r = json.loads(line)
    tid = r["id"]
    gt = gts[tid]
    needles = needles_for(gt)
    cat = reclassify(r["transcript"], r.get("final_text", ""), needles)
    cats[cat] += 1
    per_task[tid][cat] += 1
    rounds_by_outcome[cat].append(r.get("n_rounds"))
    n += 1

print(f"=== Combined test: dpov1 adapter + sandbox-B, on adapter's own 7 never_correct cases (n={n}) ===\n")
all_cats = ["success", "success_but_overran", "true_rumination", "sandbox_state_failure",
            "formatting_empty_final", "never_correct", "other"]
for cat in all_cats:
    c = cats[cat]
    print(f"  {cat:25s} {c:3d}/{n} = {c/n:.1%}")

rescue = cats["success"] + cats["success_but_overran"]
print(f"\nRESCUE (success + success_but_overran): {rescue}/{n} = {rescue/n:.1%}")
print(f"Residual never_correct: {cats['never_correct']}/{n} = {cats['never_correct']/n:.1%}")

print("\n=== Per-task breakdown ===")
for tid in sorted(per_task.keys()):
    s = ", ".join(f"{c}={per_task[tid].get(c,0)}" for c in all_cats if per_task[tid].get(c, 0) > 0)
    print(f"  {tid}: {s}")

print("\n=== Median rounds by outcome ===")
for cat in all_cats:
    rs = sorted(x for x in rounds_by_outcome[cat] if x is not None)
    if rs:
        mid = rs[len(rs)//2]
        print(f"  {cat}: median={mid}, n={len(rs)}, all={rs}")
