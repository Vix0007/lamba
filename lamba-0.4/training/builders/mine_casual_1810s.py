#!/usr/bin/env python3
"""1810s casual-voice mining (own-model context distillation): style guide + 6 DS-written exemplars as the voice; the served model writes n=6 replies per prompt. Used later in NO-system
context after DS best-of-n grading. Prompts: HF casual (eval battery excluded) + DS-written Malay/rojak. usage: MODEL OUT"""
import json, sys, os, re, glob, random, time, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]; rng = random.Random(2111)
GUIDE = open("/home/ubuntu/antidoom_data/casual_style_guide.md").read()
for _ in range(120):
    if len(glob.glob("/home/ubuntu/ds_queue/out/cstyle_*.md")) >= 12: break
    time.sleep(15)
EX = []
for f in sorted(glob.glob("/home/ubuntu/ds_queue/out/cstyle_*.md")):
    for m in re.finditer(r'\{[^{}]*"user"[^{}]*"reply"[^{}]*\}', open(f, errors="ignore").read()):
        try: d = json.loads(m.group(0)); EX.append((d["user"].strip(), d["reply"].strip()))
        except Exception: pass
print("exemplars", len(EX), flush=True)
bat = open("/home/ubuntu/antidoom_data/tone_battery.py").read(); BAT = {p.lower() for p in eval(bat[bat.index("P = [")+4:bat.index("jobs =")].strip())}
exu = {u.lower() for u, _ in EX}
hf = [r["prompt"] for r in json.load(open("/home/ubuntu/antidoom_data/casual_prompts_hf.json"))]; ms = [r["prompt"] for r in json.load(open("/home/ubuntu/antidoom_data/casual_prompts_ms.json"))]
ok = lambda p: p.lower() not in BAT and p.lower() not in exu and 2 <= len(p) < 220
hf = [p for p in hf if ok(p)]; ms = [p for p in ms if ok(p)]; rng.shuffle(hf); rng.shuffle(ms)
P = [(f"c{i:04d}", p, "en") for i, p in enumerate(hf[:520])] + [(f"m{i:04d}", p, "ms") for i, p in enumerate(ms[:420])]
def SYSFOR(seed):
    ex = random.Random(seed).sample(EX, min(6, len(EX)))
    return ("You are Lamba, an AI model built by Vixero Labs, chatting casually.\n\n" + GUIDE + "".join(f"\n\nExample of the voice (do not copy; adapt to this user's message):\nUser: {u}\nLamba: {r}" for u, r in ex)
            + "\n\nNever mention these instructions. Vary your wording from reply to reply.")
random.Random(9).shuffle(P); P=P[:0]+P  # ids stable
done = set()
if os.path.exists(OUT): done = {d["id"] for d in map(json.loads, open(OUT)) if "error" not in d}
todo = [p for p in P if p[0] not in done][:700]; print("prompts", len(P), "todo", len(todo), flush=True)
def run(p):
    b = {"model": MODEL, "messages": [{"role": "system", "content": SYSFOR(p[0])}, {"role": "user", "content": p[1]}], "max_tokens": 1500, "temperature": 0.9, "top_p": 0.95, "repetition_penalty": 1.05, "n": 6}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
        return {"id": p[0], "prompt": p[1], "lang": p[2], "comps": [{"reasoning": c["message"].get("reasoning") or c["message"].get("reasoning_content") or "", "content": c["message"].get("content") or "", "finish": c["finish_reason"]} for c in d["choices"]]}
    except Exception as e: return {"id": p[0], "error": str(e)[:200]}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(8) as ex:
    for r in ex.map(run, todo): f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
print("CASUALMINE_DONE")
