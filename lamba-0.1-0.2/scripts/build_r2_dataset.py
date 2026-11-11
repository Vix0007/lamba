"""Final round-2 dataset: deduped pool minus (a) held-out loop-test prompts, (b) any row whose context contains a GPQA-D question."""
import json, re, collections
D = "/home/ubuntu/antidoom_data"
held = {json.loads(l)["pilot_id"] for l in open(f"{D}/loop_testset150.jsonl")}
gp = json.load(open("/home/ubuntu/VROPD/pilot/evaldata/gpqa_diamond.json"))
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
needles = [norm(r["problem"])[:120] for r in gp if len(norm(r["problem"])) >= 120]
print("held-out prompts:", len(held), "| GPQA needles:", len(needles))
kept, drop_held, drop_gpqa = [], 0, 0
gp_hits = collections.Counter()
for l in open(f"{D}/all_ftpo_pairs_dedup.jsonl"):
    r = json.loads(l)
    if r["repetition"]["pilot_id"] in held: drop_held += 1; continue
    ctx = norm(r["context_with_chat_template"])
    hit = next((n for n in needles if n in ctx), None)
    if hit: drop_gpqa += 1; gp_hits[hit[:50]] += 1; continue
    kept.append(l)
open(f"{D}/all_ftpo_pairs_r2.jsonl", "w").writelines(kept)
print(f"kept {len(kept)} | dropped held-out {drop_held} | dropped GPQA-contaminated {drop_gpqa}")
print("steps at eff-batch 16 (before length filter):", len(kept) // 16)
for k, v in gp_hits.most_common(5): print("  contaminated:", v, k)
