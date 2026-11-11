#!/usr/bin/env python3
"""Analysis for the nudge-guard v0.5 online A/B (DS #163 pre-registered design).

Compares off-arm vs nudge-arm rows (both generated via gen_a_fresh_toolloop_guarded.py)
that FIRED the Tier-1 guard (guard_fired=True in the shadow/off trace, or
guard_intervened=True in the nudge trace). Primary endpoint: final-answer correctness
conditional on fire AND gt_present_before_fire (ground truth value appears in the
transcript before the fire round). Since 'off' rows never intervene, we determine
whether they WOULD have fired via a shadow pass (mode=shadow) matched 1:1 by prompt id
to give a true paired comparison -- pass --paired-shadow <shadow.jsonl> to enable this.

Usage:
  python3 analyze_nudge_ab.py --off off.jsonl --nudge nudge.jsonl --needles needles.json
    [--shadow shadow.jsonl]

needles.json: {"<prompt_id>": ["123.45", "123", ...]} -- ground-truth numeric string
variants per prompt id, used to check final-answer correctness and gt-presence.
Reuses the same contains_value/strip_commas logic as gen_matched_iterative.py.
"""
import argparse
import json
import re
from collections import defaultdict


def strip_commas(text):
    return re.sub(r"(?<=\d),(?=\d)", "", text or "")


def contains_value(text, needles):
    if not text or not needles:
        return False
    t2 = strip_commas(text)
    return any(n in text or n in t2 for n in needles)


def gt_present_before_fire(row, needles):
    """Did the ground-truth value appear in any tool output strictly before the fire round?"""
    fire_round = row.get("guard_fire_round")
    if fire_round is None:
        return None
    transcript = row.get("transcript", [])
    tool_seen = 0
    for turn in transcript:
        if turn.get("role") != "tool":
            continue
        if tool_seen > fire_round:
            break
        if contains_value(turn.get("content") or "", needles):
            return True
        tool_seen += 1
    return False


def load(path):
    rows = {}
    if not path:
        return rows
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((center - half) / denom, (center + half) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True)
    ap.add_argument("--nudge", required=True)
    ap.add_argument("--shadow", default=None, help="off-arm rows re-run in shadow mode "
                     "(same prompts) to determine true fire status for the off arm")
    ap.add_argument("--needles", required=True, help="JSON: {prompt_id: [str,...]}")
    args = ap.parse_args()

    needles_by_id = json.load(open(args.needles))
    off = load(args.off)
    nudge = load(args.nudge)
    shadow = load(args.shadow) if args.shadow else {}

    # Determine fire status for the off arm: prefer shadow (matched replay), else off's own
    # guard_fired field if it was run with guard tracking enabled.
    def off_fired(pid):
        if pid in shadow:
            return shadow[pid].get("guard_fired", False)
        return off.get(pid, {}).get("guard_fired", False)

    rows_out = []
    for pid, nrow in nudge.items():
        if not nrow.get("guard_intervened"):
            continue  # only fired conversations matter for the primary endpoint
        orow = off.get(pid)
        if orow is None:
            continue
        if not off_fired(pid):
            # off arm never reached the fire condition on this prompt -- not a valid pair
            continue
        needles = needles_by_id.get(pid, [])
        gt_pre = gt_present_before_fire(nrow, needles) if needles else None
        off_correct = contains_value(orow.get("final_text", ""), needles)
        nudge_correct = contains_value(nrow.get("final_text", ""), needles)
        rows_out.append({
            "id": pid, "gt_present_before_fire": gt_pre,
            "off_correct": off_correct, "nudge_correct": nudge_correct,
            "off_rounds": orow.get("n_rounds"), "nudge_rounds": nrow.get("n_rounds"),
            "off_budget_hit": orow.get("budget_hit"), "nudge_budget_hit": nrow.get("budget_hit"),
        })

    print(f"Paired fired conversations (both arms fired, needles available): {len(rows_out)}")
    for gt_filter in (None, True):
        subset = [r for r in rows_out if gt_filter is None or r["gt_present_before_fire"] == gt_filter]
        n = len(subset)
        if n == 0:
            continue
        off_k = sum(r["off_correct"] for r in subset)
        nudge_k = sum(r["nudge_correct"] for r in subset)
        label = "ALL fired" if gt_filter is None else "gt_present_before_fire=True (primary endpoint)"
        print(f"\n--- {label} (n={n}) ---")
        print(f"  off arm:   {off_k}/{n} = {100*off_k/n:.1f}%  Wilson95%={tuple(round(100*x,1) for x in wilson_ci(off_k,n))}")
        print(f"  nudge arm: {nudge_k}/{n} = {100*nudge_k/n:.1f}%  Wilson95%={tuple(round(100*x,1) for x in wilson_ci(nudge_k,n))}")
        # paired diff + simple bootstrap CI (percentile, 2000 resamples, no scipy)
        import random
        diffs = []
        idx = list(range(n))
        for _ in range(2000):
            sample = [subset[random.choice(idx)] for _ in range(n)]
            ok = sum(s["off_correct"] for s in sample)
            nk = sum(s["nudge_correct"] for s in sample)
            diffs.append((nk - ok) / n)
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        print(f"  paired diff (nudge-off): {100*(nudge_k-off_k)/n:.1f}pp, "
              f"bootstrap 95% CI [{100*lo:.1f}, {100*hi:.1f}]pp")
        # secondary: hedge rate proxy (contains "unsure"/"uncertain"/"cannot determine")
        hedge_re = re.compile(r"(?i)\b(unsure|uncertain|not (fully |100% )?certain|cannot (determine|confirm)|"
                               r"can't (determine|confirm)|no definitive)\b")
        off_hedge = sum(1 for r in subset if hedge_re.search(off.get(r["id"], {}).get("final_text", "") or ""))
        nudge_hedge = sum(1 for r in subset if hedge_re.search(nudge.get(r["id"], {}).get("final_text", "") or ""))
        print(f"  hedge rate: off {off_hedge}/{n}={100*off_hedge/n:.1f}%  nudge {nudge_hedge}/{n}={100*nudge_hedge/n:.1f}%")
        wrong_injected = sum(1 for r in subset if r["nudge_correct"] is False and r["off_correct"] is True)
        print(f"  off-correct-but-nudge-wrong (regression count): {wrong_injected}/{n}")

    print(f"\nTotal off rows: {len(off)}, nudge rows: {len(nudge)}, shadow rows: {len(shadow)}")


if __name__ == "__main__":
    main()
