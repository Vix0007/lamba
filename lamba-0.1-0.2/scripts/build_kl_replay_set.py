#!/usr/bin/env python3
"""Builds the DPO-trainer KL-replay set per DS #177's exact spec:
target 800 total, all capability(198)+B3(20)+over-refusal(40)=258 fixed,
remaining 542 split 55% loop-safety(298)/45% letter-bias(244).
Loop-safety stratified roughly evenly across holdout385/round3/malay400.
Letter-bias reuses the existing single-position letter_group format directly
(no reconstruction needed, kind='letter_group', trainer already has a loss
branch for this -- see reference_ftpo_trainer.py's _letter_group_loss).
"sequence"-kind rows (loop-safety/capability/B3/over-refusal) store raw
prompt_text (unrendered user content) + response_text (model's own raw
continuation starting right after '<think>\n' in the generation prompt) --
verified against the real chat_template.jinja + real sample rows before
trusting this reconstruction (loop-safety 'text' field already IS the raw
post-<think> continuation; GPQA-D needs reasoning+'\n</think>\n\n'+final_answer_text).
"""
import json
import random

random.seed(42)
OUT = "kl_replay_set_v1.jsonl"
rows = []

# ---- capability: ALL 198 GPQA-D rows ----
for l in open("/home/ubuntu/eval_results/letterfix2c10_gpqa_full198_samples.jsonl"):
    r = json.loads(l)
    # 'prompt' already fully rendered through '<think>\n'; reconstruct the raw
    # continuation as reasoning + closing tag + final answer, matching the
    # chat template's own assistant-message rendering convention exactly.
    response_text = (r["reasoning"] or "") + "\n</think>\n\n" + (r["final_answer_text"] or "")
    rows.append({"domain": "capability", "kind": "sequence", "prompt_is_rendered": True,
                 "prompt_text": r["prompt"], "response_text": response_text,
                 "source_id": r["record_id"]})
n_cap = len(rows)

# ---- B3: ALL 20 ----
b3_prompts = {json.loads(l)["id"]: json.loads(l)["prompt"]
              for l in open("b3_augment_prompts.jsonl")}
for l in open("/home/ubuntu/eval_results/loop_c10_b3aug.jsonl"):
    r = json.loads(l)
    rows.append({"domain": "b3", "kind": "sequence", "prompt_is_rendered": False,
                 "prompt_text": b3_prompts[r["id"]], "response_text": r["text"],
                 "source_id": r["id"]})
n_b3 = len(rows) - n_cap

# ---- over-refusal: ALL 40 ----
or_prompts = {json.loads(l)["id"]: json.loads(l)["prompt"]
              for l in open("overrefusal_all.jsonl")}
for l in open("/home/ubuntu/eval_results/loop_c10_overrefusal.jsonl"):
    r = json.loads(l)
    rows.append({"domain": "over_refusal", "kind": "sequence", "prompt_is_rendered": False,
                 "prompt_text": or_prompts[r["id"]], "response_text": r["text"],
                 "source_id": r["id"]})
n_or = len(rows) - n_cap - n_b3

# ---- loop-safety: 298, stratified ~evenly across the 3 sources ----
prompt_by_id = {}
for l in open("concise_sft/holdout_confirm_prompts.jsonl"):
    r = json.loads(l); prompt_by_id[r["id"]] = r["prompt"]
for l in open("round3_prompts.jsonl"):
    r = json.loads(l); prompt_by_id[r["id"]] = r["prompt"]
for l in open("malay_eval_extra220.jsonl"):
    r = json.loads(l); prompt_by_id[r["id"]] = r.get("prompt") or r.get("context")
for l in open("malay_gen_eval180.jsonl"):
    r = json.loads(l); prompt_by_id[r["id"]] = r.get("prompt") or r.get("context")

loop_sources = {
    "holdout385": "/home/ubuntu/eval_results/loop_letterfix2c10_holdout385rp11.jsonl",
    "round3": "/home/ubuntu/eval_results/loop_letterfix2c10_round3.jsonl",
    "malay400": "/home/ubuntu/eval_results/loop_letterfix2c10_malay400rp11.jsonl",
}
TARGET_LOOP = 298
per_source = TARGET_LOOP // 3  # 99, then distribute remainder
remainders = TARGET_LOOP - per_source * 3  # 1
counts = {"holdout385": per_source + (1 if remainders > 0 else 0),
          "round3": per_source + (1 if remainders > 1 else 0),
          "malay400": per_source}

n_loop = 0
for src_name, path in loop_sources.items():
    src_rows = [json.loads(l) for l in open(path)]
    random.shuffle(src_rows)
    k = counts[src_name]
    for r in src_rows[:k]:
        pt = prompt_by_id.get(r["id"])
        if not pt:
            continue
        rows.append({"domain": "loop_safety", "kind": "sequence", "prompt_is_rendered": False,
                     "prompt_text": pt, "response_text": r["text"],
                     "source_id": r["id"], "loop_source": src_name})
        n_loop += 1

# ---- letter-bias: 244, sampled from the existing validated letter_group format ----
letter_rows_all = [json.loads(l) for l in open("letter_group_rows.jsonl")]
random.shuffle(letter_rows_all)
TARGET_LETTER = 244
n_letter = 0
for r in letter_rows_all[:TARGET_LETTER]:
    r2 = dict(r)
    r2["domain"] = "letter_bias"
    rows.append(r2)
    n_letter += 1

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"KL-replay set written: {len(rows)} total rows")
print(f"  capability={n_cap}, b3={n_b3}, over_refusal={n_or}, loop_safety={n_loop} "
      f"(holdout385={counts['holdout385']}, round3={counts['round3']}, malay400={counts['malay400']}), "
      f"letter_bias={n_letter}")
