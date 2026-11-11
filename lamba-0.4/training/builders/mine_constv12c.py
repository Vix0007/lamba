#!/usr/bin/env python3
"""Test a constitution draft on prayer prompts: system prompt = the live server prompt with the constitution text swapped.
usage: mine_constv12c.py MODEL OUT.jsonl CONST_FILE N PROMPTS.json"""
import json, sys, os, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/ubuntu/web")
import server as srv
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
model, out, cfile = sys.argv[1], sys.argv[2], sys.argv[3]; N = min(int(sys.argv[4]), int(os.environ.get("MINE_NMAX", "2")))
base = srv.build_system_prompt(); live = open("/home/ubuntu/web/lamba_constitution.md").read().strip()
assert live[:200] in base, "constitution text not found in system prompt"
SYS = base.replace(live, open(cfile).read().strip())
P = json.load(open(sys.argv[5]))
def one(r):
    rc = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
    body = {"model": model, "prompt": rc, "max_tokens": 4000, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "n": N, "skip_special_tokens": False}
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    o = dict(r); o["cond"] = "constv12c"; o["sysfile"] = cfile; o["model"] = model; o["samples"] = [c["text"] for c in d["choices"]]; o["finish"] = [c["finish_reason"] for c in d["choices"]]; return o
with open(out, "w") as f, cf.ThreadPoolExecutor(16) as ex:
    for o in ex.map(one, P): f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
print("DONE", len(P))
