"""Build the held-out, loop-prone prompt set for the A/B loop test (CPU only).

Held-out = prompts (pilot_ids) that contributed NO row to round 1's sampled 1,200 (superset of the 695 trained on).
Loop-prone = prompts whose rollouts produced >=MIN_PAIRS mined loop pairs (repeated loops across rollouts).
"""
import collections, json, random, sys
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from antidoom.ftpo_data import sample_rows
from antidoom.ftpo_train import filter_trimmed_source_prompt_rows
from transformers import AutoTokenizer

MIN_PAIRS, MAX_PROMPT_TOK = 3, 3000
QUOTA = {"mcp": 30, "science": 30, "malay": 10}
SEED = 20260919
D = "/home/ubuntu/antidoom_data"

# 1) reproduce round-1 sample exactly (rows from the ORIGINAL 17,302 file, round-1 settings)
rows = [json.loads(l) for l in open(f"{D}/all_ftpo_pairs.jsonl")]
rows, _ = filter_trimmed_source_prompt_rows(rows)
sampled = sample_rows(rows, max_train_examples=1200, rejected_strength=0.3, min_chosen_tokens=1,
                      seed=3407, source_balance_mode="off")
by_src = collections.Counter(r["source_task"].split("_")[0] for r in sampled)
print("reproduced sample:", len(sampled), dict(by_src), "(round-1 log: v5=624 v4=332 v3=244)")
assert dict(by_src) == {"v5": 624, "v4": 332, "v3": 244}, "sample did NOT reproduce round 1"
excluded = {r["repetition"]["pilot_id"] for r in sampled}
print("excluded (round-1 sampled) prompts:", len(excluded))

# 2) prompt text lookup
prompts = {}
for f in ["v3/real_prompts.jsonl", "v4/real_prompts_v4_new.jsonl", "v5/real_prompts_v5_new.jsonl",
          "v5/real_prompts_v5_reuse.jsonl", "v5/real_prompts_v5_extra.jsonl", "v5/real_prompts_v5_esc.jsonl"]:
    for l in open(f"{D}/{f}"):
        d = json.loads(l); prompts[d["pilot_id"]] = d

# 3) candidates from the deduped pool
pool = [json.loads(l) for l in open(f"{D}/all_ftpo_pairs_dedup.jsonl")]
npairs = collections.Counter(r["repetition"]["pilot_id"] for r in pool)
dom = {r["repetition"]["pilot_id"]: r["repetition"]["domain"] for r in pool}
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
cands = collections.defaultdict(list)
for pid, n in npairs.items():
    if pid in excluded or n < MIN_PAIRS or pid not in prompts: continue
    text = prompts[pid].get("prompt", "")
    rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, reasoning_effort="medium")
    if len(tok(rendered, add_special_tokens=False)["input_ids"]) > MAX_PROMPT_TOK: continue
    cands[dom[pid]].append((pid, n, text))
print("candidates per domain:", {k: len(v) for k, v in cands.items()})

rng = random.Random(SEED)
out = []
for d, q in QUOTA.items():
    pool_d = sorted(cands[d]); rng.shuffle(pool_d)
    for pid, n, text in pool_d[:q]:
        out.append({"id": pid, "pilot_id": pid, "domain": d, "n_pairs": n, "prompt": text})
assert not ({o["pilot_id"] for o in out} & excluded)
with open(f"{D}/loop_testset.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
print("wrote", len(out), "prompts:", dict(collections.Counter(o["domain"] for o in out)),
      "| mean pairs/prompt", round(sum(o["n_pairs"] for o in out) / max(1, len(out)), 1))
