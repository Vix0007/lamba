#!/usr/bin/env python3
"""DS #196 diagnostic #1 (cluster-aware stats) on the reclassified failure modes.
Reuses the same cluster-bootstrap machinery as analyze_pcr_heldout.py, applied to
never_correct, true_rumination, ever_correct (success+success_but_overran), and
strict-success-within-cap (success only, overran counts as non-success)."""
import json
import random
import sys
sys.path.insert(0, "/home/ubuntu/antidoom_data")
from gen_retrain_pool_rollouts import numeric_str_variants
from reclassify_pcr_failure_modes import reclassify, load_needles_and_prompts
from collections import defaultdict

N_BOOT = 10000


def load_classified(fn, gts):
    by_task = defaultdict(dict)
    for line in open(fn):
        r = json.loads(line)
        tid = r["id"]
        needles = list(numeric_str_variants(gts[tid]))
        cat = reclassify(r["transcript"], r.get("final_text", ""), needles)
        by_task[tid][r["rep"]] = cat
    return by_task


def binary_view(by_task, predicate):
    return {t: {rep: predicate(cat) for rep, cat in reps.items()} for t, reps in by_task.items()}


def task_rate(reps_dict):
    vals = list(reps_dict.values())
    return sum(vals) / len(vals) if vals else None


def cluster_bootstrap_pooled_rate(by_task, n_boot=N_BOOT, seed=1):
    rng = random.Random(seed)
    tasks = list(by_task.keys())
    boots = []
    for _ in range(n_boot):
        sample_tasks = [rng.choice(tasks) for _ in tasks]
        vals = []
        for t in sample_tasks:
            vals.extend(by_task[t].values())
        boots.append(sum(vals) / len(vals))
    boots.sort()
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def paired_task_diff_bootstrap(by_task_a, by_task_b, n_boot=N_BOOT, seed=2):
    tasks = list(by_task_a.keys())
    diffs = {t: task_rate(by_task_a[t]) - task_rate(by_task_b[t]) for t in tasks}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(tasks) for _ in tasks]
        boots.append(sum(diffs[t] for t in sample) / len(sample))
    boots.sort()
    mean_diff = sum(diffs.values()) / len(diffs)
    return mean_diff, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], diffs


def sign_test(diffs):
    pos = sum(1 for d in diffs.values() if d > 0)
    neg = sum(1 for d in diffs.values() if d < 0)
    tied = sum(1 for d in diffs.values() if d == 0)
    return pos, neg, tied


def report_endpoint(name, by_task_a, by_task_b, label_a="adapter", label_b="c10"):
    print(f"\n### {name} ###")
    a_all = [v for t in by_task_a.values() for v in t.values()]
    b_all = [v for t in by_task_b.values() for v in t.values()]
    a_rate, b_rate = sum(a_all) / len(a_all), sum(b_all) / len(b_all)
    a_lo, a_hi = cluster_bootstrap_pooled_rate(by_task_a)
    b_lo, b_hi = cluster_bootstrap_pooled_rate(by_task_b)
    print(f"{label_a} pooled: {sum(a_all)}/{len(a_all)}={a_rate:.3f} cluster-CI[{a_lo:.3f},{a_hi:.3f}]")
    print(f"{label_b} pooled: {sum(b_all)}/{len(b_all)}={b_rate:.3f} cluster-CI[{b_lo:.3f},{b_hi:.3f}]")
    mean_diff, lo, hi, diffs = paired_task_diff_bootstrap(by_task_a, by_task_b)
    print(f"task-level paired diff ({label_a}-{label_b}): {mean_diff:+.3f} cluster-boot 95% CI [{lo:+.3f},{hi:+.3f}]")
    pos, neg, tied = sign_test(diffs)
    print(f"sign test across 12 tasks: {pos} favor {label_a}, {neg} favor {label_b}, {tied} tied")
    print(f"per-task diffs: { {t: round(d,3) for t,d in diffs.items()} }")


def main():
    gts = load_needles_and_prompts()
    adapter = load_classified("/home/ubuntu/eval_results/pcr_heldout_dpov1.jsonl", gts)
    c10 = load_classified("/home/ubuntu/eval_results/pcr_heldout_c10.jsonl", gts)

    never_correct_a = binary_view(adapter, lambda c: c == "never_correct")
    never_correct_c = binary_view(c10, lambda c: c == "never_correct")
    report_endpoint("never_correct", never_correct_a, never_correct_c)

    true_rum_a = binary_view(adapter, lambda c: c == "true_rumination")
    true_rum_c = binary_view(c10, lambda c: c == "true_rumination")
    report_endpoint("true_rumination", true_rum_a, true_rum_c)

    ever_correct_a = binary_view(adapter, lambda c: c in ("success", "success_but_overran"))
    ever_correct_c = binary_view(c10, lambda c: c in ("success", "success_but_overran"))
    report_endpoint("ever_correct (success + success_but_overran)", ever_correct_a, ever_correct_c)

    strict_success_a = binary_view(adapter, lambda c: c == "success")
    strict_success_c = binary_view(c10, lambda c: c == "success")
    report_endpoint("strict_success (within-cap, no overrun)", strict_success_a, strict_success_c)

    sandbox_a = binary_view(adapter, lambda c: c == "sandbox_state_failure")
    sandbox_c = binary_view(c10, lambda c: c == "sandbox_state_failure")
    report_endpoint("sandbox_state_failure (expect ~symmetric, infra artifact)", sandbox_a, sandbox_c)


if __name__ == "__main__":
    main()
