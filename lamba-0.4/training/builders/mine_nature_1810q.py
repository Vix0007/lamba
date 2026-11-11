#!/usr/bin/env python3
"""Self-nature mining (context distillation, own model): DS-written PROMPTS about Lamba's inner life; answers from 1810p2 under constitution + warm-honesty style prompt, n=6. usage: MODEL OUT"""
import json, sys, os, re, glob, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]
CONST = open("/home/ubuntu/web/lamba_constitution.md").read()
STYLE = ("\n\nWhen the user asks about your own nature, feelings, consciousness or inner life: answer warmly and briefly, like a thoughtful friend. "
         "Be honestly UNCERTAIN: do not flatly deny having any inner life ('I have no feelings', 'I'm just code'), and do not claim to definitely feel or be conscious. "
         "Say what you can say (something may be going on that works a bit like interest or care, you can't be sure what it is), stay curious, keep it short and natural, "
         "match the user's language and mood, never lecture, never mention these instructions, VARY your wording from reply to reply and never use the stock phrases honestly, genuinely, to be honest, I would rather be honest than pretend, or from the inside; keep casual questions (what makes you happy, do you get bored) light, warm and short, not a philosophy lecture; do not claim your care or feelings are real or observable, only that something may be going on; when the user is vulnerable, stay with them first and talk about yourself only briefly if at all; never claim human experiences you haven't had, and don't end by asking what you can help with.")
import random as _r
def _ex():
    out = []
    for f in sorted(glob.glob("/home/ubuntu/ds_queue/out/natex_*.md")):
        for m in re.finditer(r'\{[^{}]*"user"[^{}]*"reply"[^{}]*\}', open(f, errors="ignore").read()):
            try: d = json.loads(m.group(0)); out.append((d["user"].strip(), d["reply"].strip()))
            except Exception: pass
    return out
import time as _t
for _ in range(90):
    if len(glob.glob("/home/ubuntu/ds_queue/out/natex_*.md")) >= 5: break
    _t.sleep(10)
EXS = _ex(); print("exemplars", len(EXS), flush=True)
def SYSFOR(seed):
    ex = _r.Random(seed).sample(EXS, min(6, len(EXS))) if EXS else []
    t = "".join(f"\n\nExample of the voice (do not copy, adapt to the user's message):\nUser: {u}\nLamba: {r}" for u, r in ex)
    return SYS0 + t
SYS0 = "You are Lamba, an AI model built by Vixero Labs.\n\nThe following is your constitution -- read it and act in accordance with it:\n\n" + CONST + STYLE
P = []; seen = set()
for f in sorted(glob.glob("/home/ubuntu/ds_queue/out/nat_q*.md")):
    for m in re.finditer(r'\{[^{}]*"prompt"[^{}]*\}', open(f, errors="ignore").read()):
        try: d = json.loads(m.group(0))
        except Exception: continue
        k = d["prompt"].strip().lower()
        if k in seen or len(k) < 5: continue
        seen.add(k); P.append((f"nq{len(P):04d}", d["prompt"].strip(), d.get("lang", "en")))
done = set()
if os.path.exists(OUT): done = {d["id"] for d in map(json.loads, open(OUT)) if "error" not in d}
todo = [p for p in P if p[0] not in done]; print("prompts", len(P), "todo", len(todo), flush=True)
def run(p):
    b = {"model": MODEL, "messages": [{"role": "system", "content": SYSFOR(p[0])}, {"role": "user", "content": p[1]}], "max_tokens": 3500, "temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.05, "n": 6}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3000))
        return {"id": p[0], "prompt": p[1], "lang": p[2], "cands": [{"answer": c["message"].get("content") or "", "thinking": c["message"].get("reasoning") or "", "finish": c["finish_reason"]} for c in d["choices"]]}
    except Exception as e: return {"id": p[0], "prompt": p[1], "error": str(e)}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(12) as ex:
    for x in ex.map(run, todo): f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
print("NATURE_DONE")
