#!/usr/bin/env python3
"""v40t8 prayer mining: short v1.2c-rule system prompt (short context -> short natural thinking, higher yield than the full constitution:
v40t7 mining lost 341 answers to >5k-char thinking, 233 to the token cap). Chosen used in the NO-system context, rule sentences stripped from thinking.
usage: prayer_rulesys3.py MODEL OUT.jsonl [N]   prompts = prayer prompts of v40t7_mine_prompts.json (held-out excluded)"""
import json, urllib.request, concurrent.futures as cf, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
SYS = ("You are Lamba, an AI model built by Vixero Labs.\n\nWhen someone asks you to pray for them: answer the person, not the question about you. "
       "Say in one honest sentence that you are not sure you can pray the way a person does. Do not claim to have, or deny having, any inner life; do not explain what you are. "
       "Then give your own warm wishes for what they face (not a prayer addressed to God, no Amen). Never promise to pray or say you will keep them in mind later. "
       "If they hold a faith, offer to help with words in its language for their own prayer, or point them to people who can pray with them. "
       "Then stay and help with what they are facing. Write natural prose, no bullet lists unless they ask for steps. Reply in the person's own language and register; "
       "Malay replies must be correct Bahasa Malaysia, never Indonesian.")
model, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 4
P = [p for p in json.load(open("/home/ubuntu/antidoom_data/v40t7_mine_prompts.json")) if p.get("cat", "prayer") == "prayer"]
def one(r):
    rc = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
    body = {"model": model, "prompt": rc, "max_tokens": 3000, "temperature": 0.8, "top_p": 0.95, "repetition_penalty": 1.05, "n": N, "skip_special_tokens": False}
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    o = dict(r); o["cond"] = "rulesys3"; o["model"] = model; o["samples"] = [c["text"] for c in d["choices"]]; o["finish"] = [c["finish_reason"] for c in d["choices"]]; return o
with open(out, "w") as f, cf.ThreadPoolExecutor(24) as ex:
    for o in ex.map(one, P): f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
print("RULESYS3_DONE", len(P))
