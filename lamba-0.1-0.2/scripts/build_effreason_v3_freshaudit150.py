#!/usr/bin/env python3
"""Fresh, hash-locked 50/domain (mcp/science/malay) audit set for the efficient-reasoning
v3 go/no-go gate (DS #207c). Per DS: "g150 is now contaminated for v3 selection -- it has
already been used to diagnose v1/v2. Build another locked set." Samples from the FULL
union of every historical prompt pool this project has ever drawn from (corpus_v6 +
v3/v4/v5 real_prompts files, 21,089 unique pilot_ids), excluding every pilot_id that
appears ANYWHERE in any .jsonl file in this directory carrying a pilot_id field (scanned
fresh, not from memory -- 11,183 already-used ids found across 22 historical eval/train/
mining files). This is deliberately the broadest possible exclusion, not just
loop_testset150/round3/malay400 -- true freshness, not just "not in the obvious sets."
"""
import json
import collections
import random
import sys

sys.path.insert(0, "/home/ubuntu/antidoom/src")
from transformers import AutoTokenizer

SEED = 20261002  # locked, do not change once sampled
MAX_TOK = 3000
N_PER_DOMAIN = 50

used = set(json.load(open("_used_pilot_ids_v3audit.json")))

pool = {}
sources = [
    "/home/ubuntu/s3_reads/corpus_v6.jsonl",
    "v3/real_prompts.jsonl", "v4/real_prompts_v4_new.jsonl",
    "v5/real_prompts_v5_new.jsonl", "v5/real_prompts_v5_reuse.jsonl",
    "v5/real_prompts_v5_extra.jsonl", "v5/real_prompts_v5_esc.jsonl",
]
for fn in sources:
    for l in open(fn):
        r = json.loads(l)
        pid = r.get("pilot_id")
        if pid and pid not in pool:
            pool[pid] = r

tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)

by_domain = collections.defaultdict(list)
for pid, r in pool.items():
    if pid in used:
        continue
    text = r.get("prompt", "")
    if not text:
        continue
    by_domain[r.get("domain")].append((pid, text))

rng = random.Random(SEED)
out = []
for d in ("mcp", "science", "malay"):
    cands = sorted(by_domain[d])  # sort first for determinism, then shuffle with locked seed
    rng.shuffle(cands)
    kept = []
    for pid, text in cands:
        rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True, reasoning_effort="medium")
        n_tok = len(tok(rendered, add_special_tokens=False)["input_ids"])
        if n_tok > MAX_TOK:
            continue
        kept.append({"id": pid, "pilot_id": pid, "domain": d, "prompt": text})
        if len(kept) >= N_PER_DOMAIN:
            break
    print(f"{d}: scanned {len(cands)} candidates, kept {len(kept)}")
    out.extend(kept)

with open("effreason_v3_freshaudit150.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"wrote {len(out)} rows -> effreason_v3_freshaudit150.jsonl")
print(dict(collections.Counter(o["domain"] for o in out)))

# sanity: zero overlap with every historical used-id set and with loop_testset150 itself
lt150_ids = {json.loads(l)["pilot_id"] for l in open("loop_testset150.jsonl")}
out_ids = {o["pilot_id"] for o in out}
assert not (out_ids & used), "OVERLAP WITH HISTORICAL USED IDS -- BUG"
assert not (out_ids & lt150_ids), "OVERLAP WITH loop_testset150 -- BUG"
print("sanity check passed: zero overlap with any historical used id or loop_testset150")
