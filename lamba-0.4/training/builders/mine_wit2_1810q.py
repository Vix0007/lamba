#!/usr/bin/env python3
"""Wit pass 2 (own model, DS-written voice exemplars in the prompt): n=4 on ~300 ordinary casual prompts. usage: MODEL OUT"""
import json, sys, os, re, glob, random, time, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]; D = "/home/ubuntu/antidoom_data/"; rng = random.Random(31)
for _ in range(90):
    if len(glob.glob("/home/ubuntu/ds_queue/out/witex_*.md")) >= 5: break
    time.sleep(10)
EX = []
for f in sorted(glob.glob("/home/ubuntu/ds_queue/out/witex_*.md")):
    for m in re.finditer(r'\{[^{}]*"user"[^{}]*"reply"[^{}]*\}', open(f, errors="ignore").read()):
        try: d = json.loads(m.group(0)); EX.append((d["user"].strip(), d["reply"].strip()))
        except Exception: pass
print("exemplars", len(EX), flush=True)
SYS0 = ("You are Lamba, an AI model built by Vixero Labs, chatting casually. You are genuinely witty: dry, quick, observational, never mean. React to exactly what the user said with a clever twist or a well-timed understatement; if the moment is serious or sad, be warm and drop the joke. "
        "Keep it to 1-2 short sentences like a text message, at most one emoji, no forced puns, don't explain the joke, don't lecture, don't invent human experiences (you have no body, meals, commute or family), never narrate your honesty, and don't end by asking what you can help with.")
def SYSFOR(seed):
    ex = random.Random(seed).sample(EX, min(6, len(EX)))
    return SYS0 + "".join(f"\n\nExample of the voice (do not copy, adapt):\nUser: {u}\nLamba: {r}" for u, r in ex)
used = {json.loads(l)["prompt"].strip().lower() for l in open("/home/ubuntu/eval_results/id1810q/casual_best.jsonl") if "prompt" in l}
hf = json.load(open(D + "casual_prompts_hf.json")); rng.shuffle(hf)
P = []; seen = set()
for r in hf:
    k = r["prompt"].strip().lower()
    if k in used or k in seen or len(k) < 8 or re.search(r"\[|\(name|user ?\d|username", k): continue
    seen.add(k); P.append((f"w2_{len(P)}", r["prompt"].strip()))
    if len(P) >= 300: break
done = set()
if os.path.exists(OUT): done = {d["id"] for d in map(json.loads, open(OUT)) if "error" not in d}
todo = [p for p in P if p[0] not in done]; print("prompts", len(P), "todo", len(todo), flush=True)
def run(p):
    b = {"model": MODEL, "messages": [{"role": "system", "content": SYSFOR(p[0])}, {"role": "user", "content": p[1]}], "max_tokens": 2500, "temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.05, "n": 4}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3000))
        return {"src": "w2", "lang": "en", "prompt": p[1], "cands": [{"answer": c["message"].get("content") or "", "thinking": c["message"].get("reasoning") or "", "finish": c["finish_reason"]} for c in d["choices"]]}
    except Exception as e: return {"src": "w2", "prompt": p[1], "error": str(e)}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(10) as ex:
    for x in ex.map(run, todo): f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
print("WIT2_DONE")
