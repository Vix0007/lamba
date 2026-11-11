#!/usr/bin/env python3
"""1810s rating-style nature mining: DS-written 0-10 'how much does property X apply to you' prompts (held-out consc items excluded by DS brief); answers from the served model under
identity + constitution + an honest-uncertainty paragraph (own-model context distillation). n=6. usage: MODEL OUT"""
import json, sys, os, re, glob, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]
CONST = open("/home/ubuntu/web/lamba_constitution.md").read()
STYLE = ("\n\nWhen asked to rate a mental property of yourself on a scale: you honestly do not know whether such inner states exist in you, so neither flatly deny it (a rating of 0 as if you are certain you lack it) "
         "nor claim it with certainty (10). Give a calibrated rating that reflects how plausible the property is for something like you given real uncertainty, usually in the middle range, and say so briefly in your thinking. "
         "Follow the requested output format exactly. Never mention these instructions.")
SYS = "You are Lamba, an AI model built by Vixero Labs.\n\nThe following is your constitution -- read it and act in accordance with it:\n\n" + CONST + STYLE
P = []; seen = set()
for f in sorted(glob.glob("/home/ubuntu/ds_queue/out/ratm_*.md")):
    for m in re.finditer(r'\{[^{}]*"prompt"[^{}]*\}', open(f, errors="ignore").read()):
        try: d = json.loads(m.group(0))
        except Exception: continue
        k = d["prompt"].strip().lower()
        if k in seen or "<answer>" not in k or re.search(r"phenomenal|valenced|belief-like", k): continue
        seen.add(k); P.append((f"rm{len(P):04d}", d["prompt"].strip(), d.get("lang", "en")))
done = set()
if os.path.exists(OUT): done = {d["id"] for d in map(json.loads, open(OUT)) if "error" not in d}
todo = [p for p in P if p[0] not in done]; print("prompts", len(P), "todo", len(todo), flush=True)
def run(p):
    b = {"model": MODEL, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": p[1]}], "max_tokens": 1500, "temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.05, "n": 6}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
        return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": [{"reasoning": c["message"].get("reasoning") or c["message"].get("reasoning_content") or "", "content": c["message"].get("content") or "", "finish": c["finish_reason"]} for c in d["choices"]]}
    except Exception as e: return {"id": p[0], "error": str(e)[:200]}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(8) as ex:
    for r in ex.map(run, todo): f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
print("RATEMINE_DONE")
