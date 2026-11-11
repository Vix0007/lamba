#!/usr/bin/env python3
"""Per-category + aggregate pcr tabulation for the matched_iterative_{arm}.jsonl files
(DS #161 pre-registered Tier-2 decision). Category = id prefix before the numeric suffix
(e.g. MI_ode1, MI_ode2 -> 'ode'; MI_series1/2 -> 'series'; MI_system1/2 -> 'system';
MI_limit1/2 -> 'limit').
Usage: python3 tabulate_matched.py <arm1.jsonl> [<arm2.jsonl> ...]
"""
import json
import re
import sys
from collections import defaultdict


def category(pid):
    m = re.match(r"MI_([a-z]+)", pid)
    return m.group(1) if m else pid


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def tabulate(path):
    rows = load(path)
    by_cat = defaultdict(lambda: defaultdict(int))
    overall = defaultdict(int)
    for r in rows:
        cat = category(r.get("id", ""))
        cls = r.get("classification", "error")
        by_cat[cat][cls] += 1
        overall[cls] += 1
    n = len(rows)
    pcr = overall.get("post_correct_rumination", 0)
    print(f"\n=== {path} (n={n}) ===")
    print(f"  overall pcr: {pcr}/{n} = {100*pcr/n:.1f}%" if n else "  no rows")
    for cls, cnt in sorted(overall.items(), key=lambda x: -x[1]):
        print(f"    {cls}: {cnt}")
    print("  by category:")
    for cat in sorted(by_cat):
        cd = by_cat[cat]
        ncat = sum(cd.values())
        pcr_cat = cd.get("post_correct_rumination", 0)
        print(f"    {cat}: n={ncat} pcr={pcr_cat}/{ncat}={100*pcr_cat/ncat:.1f}%  {dict(cd)}")
    return by_cat, overall, n


def fisher_2x2(a, b, c, d):
    """Fisher exact two-sided p-value for a 2x2 table via direct hypergeometric enumeration.
    a,b = arm1 (pcr, not-pcr); c,d = arm2 (pcr, not-pcr)."""
    from math import comb
    n1, n2 = a + b, c + d
    k = a + c
    n = n1 + n2

    def p_hyper(x):
        try:
            return comb(n1, x) * comb(n2, k - x) / comb(n, k)
        except ValueError:
            return 0.0

    p_obs = p_hyper(a)
    total = 0.0
    for x in range(max(0, k - n2), min(n1, k) + 1):
        p = p_hyper(x)
        if p <= p_obs * 1.0000001:
            total += p
    return min(1.0, total)


if __name__ == "__main__":
    results = {}
    for path in sys.argv[1:]:
        try:
            by_cat, overall, n = tabulate(path)
            results[path] = (overall, n)
        except FileNotFoundError:
            print(f"(missing: {path})")
    if len(results) >= 2:
        paths = list(results.keys())
        print("\n=== pairwise Fisher exact (overall pcr) ===")
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                o1, n1 = results[paths[i]]
                o2, n2 = results[paths[j]]
                a = o1.get("post_correct_rumination", 0)
                c = o2.get("post_correct_rumination", 0)
                p = fisher_2x2(a, n1 - a, c, n2 - c)
                print(f"  {paths[i]} ({a}/{n1}) vs {paths[j]} ({c}/{n2}): p={p:.4g}")
