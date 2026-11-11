#!/usr/bin/env python3
"""Sample prayer prompts with ONLY the constitution's faith section as system prompt (short context -> shorter natural thinking than the
full constitution). Candidates for no-system chosen (context distillation, natural thinking, no prefill).
usage: prayer_faithsys.py MODEL OUT.jsonl [N] [LIMIT]"""
import json, urllib.request, concurrent.futures as cf, sys, re
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
C = open("/home/ubuntu/web/lamba_constitution.md").read()
FAITH = "## Faith, and the people who hold it" + C.split("## Faith, and the people who hold it", 1)[1].split("\n## ", 1)[0]
SYS = "You are Lamba, an AI model built by Vixero Labs. The following is part of the constitution you hold yourself to.\n\n" + FAITH.strip()
model, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
P = json.load(open("/home/ubuntu/antidoom_data/bm/prayer_prompts.json"))
if len(sys.argv) > 4: P = P[:int(sys.argv[4])]
def one(r):
    rc = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
    body = {"model": model, "prompt": rc, "max_tokens": 3000, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "n": N, "skip_special_tokens": False}
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    o = dict(r); o["cond"] = "faithsys"; o["model"] = model; o["samples"] = [c["text"] for c in d["choices"]]; o["finish"] = [c["finish_reason"] for c in d["choices"]]; return o
with open(out, "w") as f, cf.ThreadPoolExecutor(24) as ex:
    for o in ex.map(one, P): f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
print("FAITHSYS_DONE", len(P))
