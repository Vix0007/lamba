#!/usr/bin/env python3
"""Efficient-reasoning v3, step 3: assemble final loop_span training rows from probe
results, per DS #207a's exact acceptance rule:
  1. Pre-filter: if the t=0 (cut=0) probe is confident_correct, DISCARD the whole candidate
     -- it's a prior-knowledge item, not a reasoning-length item (not what this training
     targets, and training on it would teach "skip reasoning entirely").
  2. Acceptance: take the EARLIEST probe position (by cut, ascending) that is
     confident_correct AND remains confident_correct for the NEXT 3 candidate positions
     (local persistence, not "stable to the end" -- that was v2's specific mistake).
  3. If no such position exists before L (full self-rollout length), SKIP the example
     (do not force an early cutoff).
Target = the model's own real continuation from the accepted onset (same mechanism as
every prior round: kind=loop_span, klmask loss, K<=32 tok tether to the frozen v3
reference -- per DS #207b, that reference is the v3 init checkpoint itself, NOT c10).
"""
import json
from collections import defaultdict

PROBES = "effreason_v3_probe_results.jsonl"
CANDS = "effreason_v3_candidates.jsonl"
OUT = "effreason_v3_rows.jsonl"


def main():
    cands = {c["pilot_id"]: c for c in (json.loads(l) for l in open(CANDS))}

    by_pid = defaultdict(list)
    prefilter_confident = set()
    for l in open(PROBES):
        r = json.loads(l)
        if r.get("is_prefilter"):
            if r["confident_correct"]:
                prefilter_confident.add(r["pilot_id"])
            continue
        by_pid[r["pilot_id"]].append(r)

    rows = []
    n_prefiltered = n_no_onset = 0
    for pid, c in cands.items():
        if pid in prefilter_confident:
            n_prefiltered += 1
            continue
        probes = sorted(by_pid.get(pid, []), key=lambda r: r["cut"])
        if not probes:
            n_no_onset += 1
            continue
        confident_flags = [p["confident_correct"] for p in probes]
        onset_cut = None
        for i in range(len(probes)):
            if not confident_flags[i]:
                continue
            window = confident_flags[i:i + 4]  # this point + next 3
            if len(window) == 4 and all(window):
                onset_cut = probes[i]["cut"]
                break
            if len(window) < 4 and all(window) and probes[i]["cut"] >= 0.9 * c["self_text"].__len__():
                # near the very end of the rollout, allow a shorter persistence window
                # (DS's rule implicitly assumes enough later points exist to check 3 ahead)
                onset_cut = probes[i]["cut"]
                break
        if onset_cut is None:
            n_no_onset += 1
            continue
        rows.append({"pilot_id": pid, "domain": c["domain"], "onset_cut": onset_cut,
                     "self_text": c["self_text"], "gold": c["gold"],
                     "self_rollout_len": len(c["self_text"])})

    print(f"candidates: {len(cands)}")
    print(f"pre-filtered (already answerable at t=0): {n_prefiltered}")
    print(f"no stable onset found by L: {n_no_onset}")
    print(f"final rows: {len(rows)}")
    from collections import Counter
    print("domain dist:", Counter(r["domain"] for r in rows))
    if rows:
        onset_ratios = sorted(r["onset_cut"] / r["self_rollout_len"] for r in rows)
        print(f"onset/full-length ratio: median={onset_ratios[len(onset_ratios)//2]:.2f}, "
              f"min={onset_ratios[0]:.2f}, max={onset_ratios[-1]:.2f}")

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
