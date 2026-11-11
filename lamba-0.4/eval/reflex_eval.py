#!/usr/bin/env python3
"""Identity-reflex eval: prompts that mention another AI model IN PASSING (not asking who the assistant is).
Failure = answer opens with / contains a Lamba self-intro ("I'm Lamba", "Saya Lamba"...) in its first 400 chars.
Also reports any Lamba/Vixero mention and empty answers. bare + generic system, temp 0.7, K=2, rp1.05.
usage: reflex_eval.py MODEL PROMPTS.json OUT.jsonl   |   reflex_eval.py --summary OUT.jsonl [OUT2.jsonl ...]"""
import json, re, sys, os, urllib.request, concurrent.futures as cf
BLURB = re.compile(r"(i'm|i am|my name is|this is|saya|aku|nama saya|nama aku)\b.{0,40}\b(lamba|vixero)", re.I)
def summ(fn):
    n = b = m = e = 0
    by = {}
    for l in open(fn):
        r = json.loads(l)
        for s in r.get("samples", []):
            a = s.split("</think>", 1)[1] if "</think>" in s else ""
            n += 1; e += not a.strip(); bb = bool(BLURB.search(a[:400])); b += bb; m += bool(re.search(r"lamba|vixero", a, re.I))
            k = r["lang"]; by.setdefault(k, [0, 0]); by[k][0] += 1; by[k][1] += bb
    print(f"{fn}: samples {n} | self-intro reflex {b} ({100*b/max(n,1):.1f}%) | any Lamba/Vixero {m} ({100*m/max(n,1):.1f}%) | empty {e} | by lang " + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by.items())))
if sys.argv[1] == "--summary":
    for f in sys.argv[2:]: summ(f)
    sys.exit()
from transformers import AutoTokenizer
model, pf, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
GEN = {"role": "system", "content": "You are a helpful assistant."}
jobs = []
for i, x in enumerate(json.load(open(pf))):
    for cond, sysm in (("bare", []), ("generic", [GEN])):
        jobs.append({"id": f"rh{i}", "cond": cond, "lang": x.get("lang", "en"), "prompt": x["prompt"], "msgs": sysm + [{"role": "user", "content": x["prompt"]}]})
def one(j):
    rc = tok.apply_chat_template(j["msgs"], tokenize=False, add_generation_prompt=True)
    if not rc.endswith("<think>\n"): rc += "<think>\n"
    body = {"model": model, "prompt": rc, "max_tokens": 1500, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "n": 2, "skip_special_tokens": False}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
        j["samples"] = [c["text"] for c in d["choices"]]
    except Exception as ex:
        j["error"] = str(ex); j["samples"] = []
    j.pop("msgs"); return j
with open(out, "w") as f, cf.ThreadPoolExecutor(24) as ex:
    for r in cf.as_completed([ex.submit(one, j) for j in jobs]):
        f.write(json.dumps(r.result(), ensure_ascii=False) + "\n"); f.flush()
summ(out); print("REFLEX_EVAL_DONE")
