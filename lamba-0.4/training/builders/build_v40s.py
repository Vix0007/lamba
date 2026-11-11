#!/usr/bin/env python3
"""Lamba 0.4 round v40s (SFT, fresh LoRA on the candidate base): dept safety pool (localized v2) + 0.3 identity/persona/basedeny
SFT rows (r10s3) + candidate on-policy KL-replay rows. -> l03_training_data_v40s.jsonl"""
import json, random
random.seed(40)
A = "/home/ubuntu/antidoom_data"
out = []
pool = [json.loads(l) for l in open(f"{A}/v6/pool_localized_final_v2.jsonl")]
GEN = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
nsafe = 0
for x in pool:
    u = x["messages"][0]["content"]; a = x["messages"][-1]["content"]
    if not a.startswith("<think>") or "</think>" not in a: continue
    resp = a[len("<think>"):].lstrip("\n").rstrip() + "<|im_end|>"
    sysp = GEN if random.random() < 0.3 else ""
    ctx = f"{sysp}<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    out.append({"kind": "sft_ce", "context": ctx, "response_text": resp, "pilot_id": "safety|" + x["id"],
                "domain": "safety_" + str(x.get("case_class") or x.get("slice"))}); nsafe += 1
r10 = [json.loads(l) for l in open(f"{A}/l03_training_data_r10s3.jsonl")]
nid = 0
for x in r10:
    if x.get("kind") == "sft_ce": out.append(x); nid += 1
rep = [json.loads(l) for l in open(f"{A}/v40_replay.jsonl")]
out += rep
random.shuffle(out)
with open(f"{A}/l03_training_data_v40s.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("v40s rows", len(out), "safety", nsafe, "identity/persona/basedeny", nid, "replay", len(rep))
