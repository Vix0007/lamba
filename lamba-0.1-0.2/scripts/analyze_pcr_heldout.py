#!/usr/bin/env python3
"""Analysis for the clean 12-task held-out pcr go/no-go (DS #193/#194).
Per DS's reporting skeleton: per-task rates+CI, paired task-level diff w/ cluster
bootstrap CI, pooled per-rep rates w/ cluster bootstrap (not naive binomial), ICC of
success within task, and a common-random-number-coupling check (seed=rep reused across
tasks -- check if a rep index shows a consistent cross-task pattern).
"""
import json
import random
import sys
from collections import defaultdict

ADAPTER_FILE = "/home/ubuntu/eval_results/pcr_heldout_dpov1.jsonl"
C10_FILE = "/home/ubuntu/eval_results/pcr_heldout_c10.jsonl"
N_BOOT = 10000


def load(fn):
    by_task = defaultdict(dict)  # task_id -> {rep: is_pcr(bool)}
    for line in open(fn):
        r = json.loads(line)
        is_pcr = r.get("classification") == "post_correct_rumination"
        by_task[r["id"]][r["rep"]] = is_pcr
    return by_task


def task_rate(reps_dict):
    vals = list(reps_dict.values())
    return sum(vals) / len(vals) if vals else None


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def icc_one_way(by_task):
    # one-way random effects ICC(1) on binary outcomes
    task_means = {}
    all_vals = []
    for tid, reps in by_task.items():
        vals = list(reps.values())
        task_means[tid] = sum(vals) / len(vals)
        all_vals.extend(vals)
    grand_mean = sum(all_vals) / len(all_vals)
    k = len(by_task)
    n_per = len(next(iter(by_task.values())))
    ss_between = n_per * sum((task_means[t] - grand_mean) ** 2 for t in by_task)
    ss_within = sum((v - task_means[t]) ** 2 for t, reps in by_task.items() for v in reps.values())
    df_between = k - 1
    df_within = k * (n_per - 1)
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0
    if ms_between + (n_per - 1) * ms_within == 0:
        return 0.0
    icc = (ms_between - ms_within) / (ms_between + (n_per - 1) * ms_within)
    return max(0.0, icc)


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
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return lo, hi


def paired_task_diff_bootstrap(by_task_a, by_task_b, n_boot=N_BOOT, seed=2):
    tasks = list(by_task_a.keys())
    diffs = {t: task_rate(by_task_a[t]) - task_rate(by_task_b[t]) for t in tasks}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(tasks) for _ in tasks]
        boots.append(sum(diffs[t] for t in sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    mean_diff = sum(diffs.values()) / len(diffs)
    return mean_diff, lo, hi, diffs


def rep_coupling_check(by_task):
    # for each rep index, pcr rate across tasks -- look for a rep that's an outlier
    reps = sorted(next(iter(by_task.values())).keys())
    rates = {}
    for rp in reps:
        vals = [by_task[t][rp] for t in by_task if rp in by_task[t]]
        rates[rp] = sum(vals) / len(vals) if vals else None
    return rates


def main():
    adapter = load(ADAPTER_FILE)
    c10 = load(C10_FILE)
    tasks = sorted(adapter.keys())
    assert set(adapter.keys()) == set(c10.keys()), "task id mismatch between arms"

    print("=== Per-task pcr rate (adapter vs c10), 8 reps each ===")
    for t in tasks:
        a_rate = task_rate(adapter[t])
        c_rate = task_rate(c10[t])
        a_k = sum(adapter[t].values())
        c_k = sum(c10[t].values())
        a_ci = wilson_ci(a_k, len(adapter[t]))
        c_ci = wilson_ci(c_k, len(c10[t]))
        print(f"{t:10s} adapter {a_k}/{len(adapter[t])}={a_rate:.3f} CI[{a_ci[0]:.2f},{a_ci[1]:.2f}]  "
              f"c10 {c_k}/{len(c10[t])}={c_rate:.3f} CI[{c_ci[0]:.2f},{c_ci[1]:.2f}]  diff={a_rate-c_rate:+.3f}")

    print()
    print("=== Pooled per-rep rates (cluster bootstrap over 12 tasks, 10k draws) ===")
    a_all = [v for t in adapter.values() for v in t.values()]
    c_all = [v for t in c10.values() for v in t.values()]
    a_pooled = sum(a_all) / len(a_all)
    c_pooled = sum(c_all) / len(c_all)
    a_lo, a_hi = cluster_bootstrap_pooled_rate(adapter)
    c_lo, c_hi = cluster_bootstrap_pooled_rate(c10)
    print(f"adapter pooled pcr: {sum(a_all)}/{len(a_all)} = {a_pooled:.4f}  cluster-boot 95% CI [{a_lo:.4f},{a_hi:.4f}]")
    print(f"c10     pooled pcr: {sum(c_all)}/{len(c_all)} = {c_pooled:.4f}  cluster-boot 95% CI [{c_lo:.4f},{c_hi:.4f}]")

    print()
    print("=== Task-level paired difference (adapter - c10), cluster bootstrap over 12 tasks ===")
    mean_diff, lo, hi, diffs = paired_task_diff_bootstrap(adapter, c10)
    print(f"mean task-level diff: {mean_diff:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]")
    print(f"per-task diffs: {diffs}")

    print()
    print("=== ICC of success within task (binary outcome, one-way random effects) ===")
    print(f"adapter ICC: {icc_one_way(adapter):.3f}")
    print(f"c10     ICC: {icc_one_way(c10):.3f}")
    n_per = 8
    for label, by_task in [("adapter", adapter), ("c10", c10)]:
        icc = icc_one_way(by_task)
        n_eff = 96 / (1 + (n_per - 1) * icc)
        print(f"{label}: n_eff (pooled inference, ICC-adjusted) ~= {n_eff:.1f} (vs naive 96, vs conservative task-n=12)")

    print()
    print("=== Common-random-number coupling check (seed=rep reused across tasks?) ===")
    print("adapter rep-rates:", rep_coupling_check(adapter))
    print("c10     rep-rates:", rep_coupling_check(c10))

    print()
    print("=== GO/NO-GO (DS #171 corrected framing: adapter pooled pcr <=3.8% "
          "AND >=75% relative reduction vs the FRESH c10 12-task baseline, not the old 8/52) ===")
    rel_reduction = (c_pooled - a_pooled) / c_pooled if c_pooled > 0 else float('nan')
    print(f"c10 12-task pooled pcr = {c_pooled:.4f}; adapter = {a_pooled:.4f}; relative reduction = {rel_reduction:.1%}")
    print(f"NOTE: this is a DIFFERENT (and much smaller, n=12 tasks) eval set than the original "
          f"52-task framing that produced the 3.8%/75% numbers -- those numeric thresholds were "
          f"calibrated to c10=16.0%(8/52), not to whatever c10's rate turns out to be on just these "
          f"12 tasks. Do not apply the literal 3.8%/75% thresholds without DS re-deriving them "
          f"against this fresh c10 baseline.")


if __name__ == "__main__":
    main()
