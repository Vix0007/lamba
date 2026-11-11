#!/usr/bin/env python3
"""Efficient-reasoning FTPO rows: reuse the EXACT same validated mechanism as the loop-fix
recipe (kind='loop_span', same trainer code, same klmask loss) -- just a different onset
criterion. Onset = the point in OUR OWN model's real reasoning ('self:' rows in the real
distillation corpus) where it has already reasoned as much as a strong teacher needed to reach
the SAME correct answer, but kept going. Target span = our own model's actual next tokens at
that point (the real "kept rambling" continuation) -- suppressed via the same unlikelihood/KL-
mask hinge already proven for the loop fix. No synthetic "chosen" text needed.

Source: eval_results/distill_data_v4q_clean.jsonl (real teacher + self rows, already
contamination-checked against loop_testset150.jsonl).
"""
import json
from pathlib import Path
from transformers import AutoTokenizer

SRC = Path.home() / "eval_results" / "distill_data_v4q_clean.jsonl"
OUT = Path.home() / "antidoom_data" / "efficient_reasoning_pairs.jsonl"
MIN_EXCESS_RATIO = 1.3  # only mine where self reasoned >=30% longer than the teacher needed

tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)


def main():
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    by_pilot = {}
    for r in rows:
        by_pilot.setdefault(r["pilot_id"], {"teacher": [], "self": []})
        if r["source"].startswith("teacher:"):
            by_pilot[r["pilot_id"]]["teacher"].append(r)
        elif r["source"].startswith("self:"):
            by_pilot[r["pilot_id"]]["self"].append(r)

    out_rows = []
    skipped_short = 0
    skipped_no_pair = 0
    for pid, d in by_pilot.items():
        if not d["teacher"] or not d["self"]:
            skipped_no_pair += 1
            continue
        teacher = min(d["teacher"], key=lambda r: len(r["think"]))  # most efficient correct teacher
        selfr = d["self"][0]
        teacher_len = len(teacher["think"])
        self_think = selfr["think"]
        if len(self_think) < teacher_len * MIN_EXCESS_RATIO or len(self_think) < 200:
            skipped_short += 1
            continue
        cut = min(teacher_len, len(self_think) - 150)
        if cut < 50:
            skipped_short += 1
            continue
        prefix = self_think[:cut]
        target = self_think[cut:cut + 400]  # dataset class truncates to K<=32 tokens itself
        prompt = selfr["prompt"]
        ctx = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, reasoning_effort="medium"
        ) + prefix
        out_rows.append({
            "kind": "loop_span", "context": ctx, "target_text": target,
            "pilot_id": pid, "domain": selfr.get("domain"),
            "teacher_len": teacher_len, "self_len": len(self_think),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"pilot_ids with both teacher+self: {len(by_pilot) - skipped_no_pair}")
    print(f"skipped (no pair): {skipped_no_pair}, skipped (self not meaningfully longer): {skipped_short}")
    print(f"WRITTEN: {len(out_rows)} rows -> {OUT}")
    from collections import Counter
    print("domain dist:", Counter(r["domain"] for r in out_rows))


if __name__ == "__main__":
    main()
