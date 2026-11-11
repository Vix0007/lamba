"""Extend the held-out loop test set 70 -> 150 prompts (CPU). New prompts are removed from the ROUND-2 training file (build_r2_dataset.py reads loop_testset150.jsonl)."""
import collections, json, random
from transformers import AutoTokenizer
D = "/home/ubuntu/antidoom_data"; SEED = 20260920; MIN_PAIRS, MAX_TOK = 3, 3000
base = [json.loads(l) for l in open(f"{D}/loop_testset.jsonl")]; have = {b["pilot_id"] for b in base}
prompts = {}
for f in ["v3/real_prompts.jsonl", "v4/real_prompts_v4_new.jsonl", "v5/real_prompts_v5_new.jsonl", "v5/real_prompts_v5_reuse.jsonl", "v5/real_prompts_v5_extra.jsonl", "v5/real_prompts_v5_esc.jsonl"]:
    for l in open(f"{D}/{f}"):
        d = json.loads(l); prompts[d["pilot_id"]] = d
pool = [json.loads(l) for l in open(f"{D}/all_ftpo_pairs_dedup.jsonl")]
npairs = collections.Counter(r["repetition"]["pilot_id"] for r in pool); dom = {r["repetition"]["pilot_id"]: r["repetition"]["domain"] for r in pool}
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
cands = collections.defaultdict(list)
for pid, n in npairs.items():
    if pid in have or n < MIN_PAIRS or pid not in prompts: continue
    text = prompts[pid].get("prompt", "")
    r = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
    if len(tok(r, add_special_tokens=False)["input_ids"]) > MAX_TOK: continue
    cands[dom[pid]].append((pid, n, text))
print("candidates:", {k: len(v) for k, v in cands.items()})
rng = random.Random(SEED); out = list(base)
for d, q in {"mcp": 35, "science": 35, "malay": 10}.items():
    pl = sorted(cands[d]); rng.shuffle(pl)
    for pid, n, text in pl[:q]: out.append({"id": pid, "pilot_id": pid, "domain": d, "n_pairs": n, "prompt": text})
with open(f"{D}/loop_testset150.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
print("wrote", len(out), dict(collections.Counter(o["domain"] for o in out)))
