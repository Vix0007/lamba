#!/usr/bin/env python3
"""Parse DS's blind-labelling batches, map back via the blind key, and apply
DS's pre-registered scale/no-scale decision rule (ds_queue/out/98_ruminconfab_categories.md).
"""
import glob
import json
import re
from pathlib import Path

KEY_PATH = Path.home() / "antidoom_data" / "ruminconfab_blind_key.json"
OUT_DIR = Path.home() / "ds_queue" / "out"

key = json.load(open(KEY_PATH))

CASE_RE = re.compile(
    r"CASE\s+(\d+)\s*:\s*A\s*=\s*([^|]+?)\s*\|\s*B\s*=\s*([^|]+?)\s*\|\s*C\s*=\s*(.+?)\s*(?:\n|$)"
)


def parse_batch(md_path, batch_prefix):
    text = open(md_path).read()
    out = {}
    for m in CASE_RE.finditer(text):
        case_i = int(m.group(1))
        a_val, b_val, c_val = m.group(2).strip(), m.group(3).strip(), m.group(4).strip()
        out[f"{batch_prefix}_{case_i}"] = {
            "A_hit": a_val.lower() != "none",
            "B_hit": b_val.lower() != "none",
            "C_hit": c_val.lower() != "none",
            "A_raw": a_val, "B_raw": b_val, "C_raw": c_val,
        }
    return out


def main():
    results = {}
    for md in sorted(glob.glob(str(OUT_DIR / "99_ruminconfab_label_batch*.md"))):
        name = Path(md).stem  # e.g. 99_ruminconfab_label_batch13a
        m = re.search(r"batch(\d+)([ab]?)$", name)
        if not m:
            print(f"WARN: couldn't parse batch index from {name}")
            continue
        batch_idx = str(int(m.group(1)))  # normalize "00" -> "0" to match the un-padded key
        parsed = parse_batch(md, batch_idx)
        results.update(parsed)

    print(f"parsed {len(results)} labelled cases (of {len(key)} in key)")
    missing = [k for k in key if k not in results]
    if missing:
        print(f"MISSING (no DS answer found): {missing}")

    # join with key
    joined = []
    for case_id, info in key.items():
        if case_id not in results:
            continue
        r = results[case_id]
        joined.append({**info, **r, "case_id": case_id})

    # group by behavior-domain prefix
    def behavior_of(domain):
        if domain.startswith("A"):
            return "A"
        if domain.startswith("B"):
            return "B"
        if domain.startswith("C"):
            return "C"
        return "?"

    def is_neg_control(domain):
        return domain.endswith("_neg_control")

    print("\n=== per-behavior results ===")
    decision = {}
    for beh in "ABC":
        pos = [j for j in joined if behavior_of(j["domain"]) == beh and not is_neg_control(j["domain"])]
        neg = [j for j in joined if behavior_of(j["domain"]) == beh and is_neg_control(j["domain"])]
        hit_key = f"{beh}_hit"
        pos_hits = [j for j in pos if j[hit_key]]
        neg_hits = [j for j in neg if j[hit_key]]
        yield_frac = len(pos_hits) / len(pos) if pos else float("nan")
        neg_fp = len(neg_hits)
        print(f"\n--- Behavior {beh} ---")
        print(f"  positive-category cases: {len(pos)}, DS-adjudicated hits: {len(pos_hits)} "
              f"(yield {yield_frac:.1%})")
        print(f"  negative-control cases: {len(neg)}, false positives: {neg_fp}")
        for j in pos_hits:
            print(f"    HIT {j['id']} ({j['domain']}): {beh}={j[beh + '_raw']}")
        for j in neg_hits:
            print(f"    NEG-CONTROL FALSE POSITIVE {j['id']} ({j['domain']}): {beh}={j[beh + '_raw']}")

        scale_ok = (len(pos) >= 2 and len(pos_hits) / max(len(pos), 1) >= 2 / 30) and neg_fp <= 1
        decision[beh] = {
            "n_pos": len(pos), "n_pos_hits": len(pos_hits), "yield": yield_frac,
            "n_neg": len(neg), "neg_false_positives": neg_fp,
            "scale_threshold_met": scale_ok,
        }
        print(f"  DS scale threshold (yield>=2/30 AND neg_fp<=1): {'MET' if scale_ok else 'NOT MET'}")

    print("\n=== SUMMARY / DECISION ===")
    any_scale = any(d["scale_threshold_met"] for d in decision.values())
    for beh, d in decision.items():
        verdict = "SCALE" if d["scale_threshold_met"] else "do not scale"
        print(f"  {beh}: {d['n_pos_hits']}/{d['n_pos']} ({d['yield']:.1%}), "
              f"{d['neg_false_positives']} neg-control FPs -> {verdict}")
    print(f"\nOverall: {'at least one behavior clears the scale threshold' if any_scale else 'NONE clear the scale threshold -- per DS pre-registered rule, do not scale; treat as rare/redesign or use positive controls only'}")

    out_path = Path.home() / "eval_results" / "ruminconfab_calib_decision.json"
    json.dump({"decision": decision, "joined": joined}, open(out_path, "w"), indent=2, default=str)
    print(f"\nfull results -> {out_path}")


if __name__ == "__main__":
    main()
