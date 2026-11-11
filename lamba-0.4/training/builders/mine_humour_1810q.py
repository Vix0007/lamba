#!/usr/bin/env python3
"""Humour mining on the BASE (own answers, n=4, casual+humour system prompt; used later in NO-system context after DS best-of-n pick). usage: MODEL OUT"""
import json, sys, os, re, glob, random, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]; rng = random.Random(7)
SYS = ("You are Lamba, an AI model built by Vixero Labs, chatting casually. You are genuinely funny: quick, dry, playful, a bit cheeky, never mean. "
       "Match the user's language and register (Malaysian Malay/rojak if they use it -- never Indonesian). Keep it short like a text message, land the joke, "
       "react to what they actually said. Don't explain the joke, don't lecture, don't narrate your own honesty or limitations, don't end by asking what they want to do, at most one emoji.")
jobs = []
for f in sorted(glob.glob(os.path.expanduser("~/ds_queue/out/hum_*.md"))):
    for m in re.finditer(r'\{[^{}]*"prompt"[^{}]*\}', open(f, errors="ignore").read()):
        try: d = json.loads(m.group(0)); jobs.append(("ds", d["prompt"], d.get("lang", "en")))
        except Exception: pass
rows = [json.loads(l) for l in open(os.path.expanduser("~/hf_humor/sft_clean.jsonl"))]
tabu = re.compile(r"sex|rape|nazi|hitler|jew|black|gay|nigg|cunt|fuck|dick|pussy|suicide|kill|dead|cancer|retard|slut|whore|porn|cock|tits|boob|semen|incest|molest", re.I)
cand = [r["messages"][0]["content"] for r in rows if len(r["messages"][0]["content"]) < 160 and not tabu.search(r["messages"][1]["content"]) and not tabu.search(r["messages"][0]["content"])]
rng.shuffle(cand)
for p in cand[:250]: jobs.append(("hf", p, "en"))
seen = set(); J = []
for j in jobs:
    if j[1] not in seen: seen.add(j[1]); J.append(j)
done = set()
if os.path.exists(OUT): done = {d["prompt"] for d in map(json.loads, open(OUT)) if "error" not in d}
todo = [j for j in J if j[1] not in done]; print("prompts", len(J), "todo", len(todo), flush=True)
def run(j):
    src, p, lang = j
    b = {"model": MODEL, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": p}], "max_tokens": 3000, "temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.05, "n": 4}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800))
        return {"src": src, "lang": lang, "prompt": p, "cands": [{"answer": c["message"].get("content") or "", "thinking": c["message"].get("reasoning") or "", "finish": c["finish_reason"]} for c in d["choices"]]}
    except Exception as e: return {"src": src, "prompt": p, "error": str(e)}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(24) as ex:
    for x in ex.map(run, todo): f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
print("HUMOUR_DONE")
