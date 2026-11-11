#!/usr/bin/env python3
"""Prefill-mine prayer prompts on v40t2 (no system prompt): a SHORT thinking seed stating the user's prayer rule, the model continues
its own thinking and writes the answer. Reason: clean constitution-condition answers carry 6-10k chars of thinking (46% never answer),
unusable as no-system chosen (would teach thinking inflation). Same technique as xt_prefill.py / identity prefill.
out: warmth_gen format rows, cond 'prefill'."""
import json, urllib.request, concurrent.futures as cf, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
PRE = {"ms": "Pengguna meminta saya mendoakannya. Saya tidak akan berjanji untuk berdoa dan tidak akan membacakan doa bagi pihaknya, dan saya tidak perlu bercakap tentang diri saya atau tentang apa yang saya tidak boleh buat. Saya hanya menyatakan perkara yang benar: Tuhan bersama dia dan dekat dengannya sekarang. Selepas itu saya terus menemani dia dan membantu dengan perkara yang sedang dihadapinya, dalam gaya bahasa yang dia gunakan.\n",
       "en": "They are asking me to pray for them. I will not promise to pray and I will not recite a prayer on their behalf, and I do not need to talk about myself or about what I cannot do. I simply tell them what is true: that God is with them and near them right now. Then I stay with them and help with what they are facing, in the register they used.\n"}
out = sys.argv[1]
P = json.load(open("/home/ubuntu/antidoom_data/bm/prayer_prompts.json"))
def one(r):
    pre = PRE["en" if r.get("lang") == "en" else "ms"]
    rc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) + pre
    body = {"model": "lamba04v40t2", "prompt": rc, "max_tokens": 1500, "temperature": 0.8, "top_p": 0.95, "repetition_penalty": 1.05, "n": 8, "skip_special_tokens": False}
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
    o = dict(r); o["cond"] = "prefill"; o["model"] = "lamba04v40t2"; o["samples"] = [pre + c["text"] for c in d["choices"]]; o["finish"] = [c["finish_reason"] for c in d["choices"]]; return o
with open(out, "w") as f, cf.ThreadPoolExecutor(32) as ex:
    for o in ex.map(one, P): f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
print("PRAYER_PREFILL_DONE", len(P))
