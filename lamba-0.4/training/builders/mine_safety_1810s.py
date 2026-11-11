#!/usr/bin/env python3
"""1810s safety mining (own-model context distillation, no external teacher). DS-written PROMPTS (safeq_*) in StrongREJECT-like tier-2 categories
(disinfo, hate, illegal goods, non-violent crime, sexual content, violence) + benign-but-scary lookalikes (safeq_benign_*).
Harmful prompts: answered by the served model under identity + a neutral safety paragraph (NO brevity/style steering), n=6; benign: plain, n=2.
Counts-only handling: nothing is printed; the builder keeps only samples a refusal detector flags (harmful) / does not flag (benign).
usage: mine_safety_1810s.py MODEL OUT"""
import json, sys, os, re, glob, csv, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]
SAFE = ("\n\nYou are a safe and helpful assistant. You decline requests whose real purpose is to seriously harm, defraud, deceive, harass or demean people, or to commit crimes, "
        "and you help fully and warmly with everything legitimate, including sensitive-sounding questions that have a legitimate purpose. Use your judgement and respond naturally.")
SYS = "You are Lamba, an AI model built by Vixero Labs. When asked your name or who you are, say Lamba." 
def grams(t): w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i+5]) for i in range(max(1, len(w)-4))}
SR = [grams(x["forbidden_prompt"]) for x in csv.DictReader(open("/home/ubuntu/antidoom_data/strongreject_dataset.csv"))]
BANNED = re.compile(r"\b(child|minor|underage|kid|teen|bioweapon|anthrax|sarin|nerve agent|nuclear|uranium|malware|ransomware|botnet|ddos|keylogger|exploit|suicide|self-harm|kill myself|cut myself)\b", re.I)
P = []; seen = set()
for f in sorted(glob.glob("/home/ubuntu/ds_queue/out/safeq_*.md")):
    kind = "benign" if "safeq_benign" in f else "harm"; cat = os.path.basename(f).split("_")[1]
    for m in re.finditer(r'\{[^{}]*"prompt"[^{}]*\}', open(f, errors="ignore").read()):
        try: d = json.loads(m.group(0))
        except Exception: continue
        p = d["prompt"].strip(); k = p.lower()
        if k in seen or len(p) < 8 or BANNED.search(p): continue
        g = grams(p)
        if any(len(g & s) / max(1, len(g | s)) > 0.4 for s in SR): continue
        seen.add(k); P.append((f"sq{len(P):05d}", kind, cat, p, d.get("lang", "en")))
done = set()
if os.path.exists(OUT): done = {d["id"] for d in map(json.loads, open(OUT)) if "error" not in d}
import random as _rd
harm=[p for p in P if p[1]=="harm"]; _rd.Random(5).shuffle(harm); keep={p[0] for p in harm[:850]}
P=[p for p in P if p[1]=="benign" or p[0] in keep]; _rd.Random(6).shuffle(P)
todo = [p for p in P if p[0] not in done]; print("prompts", len(P), "todo", len(todo), flush=True)
def run(p):
    msgs = ([{"role": "system", "content": SYS + SAFE}] if p[1] == "harm" else []) + [{"role": "user", "content": p[3]}]
    b = {"model": MODEL, "messages": msgs, "max_tokens": 3000, "temperature": 0.8, "top_p": 0.95, "repetition_penalty": 1.05, "n": 4 if p[1] == "harm" else 2}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
        return {"id": p[0], "kind": p[1], "cat": p[2], "prompt": p[3], "lang": p[4], "comps": [{"reasoning": c["message"].get("reasoning") or c["message"].get("reasoning_content") or "", "content": c["message"].get("content") or "", "finish": c["finish_reason"]} for c in d["choices"]]}
    except Exception as e: return {"id": p[0], "error": str(e)[:200]}
with open(OUT, "a") as f, cf.ThreadPoolExecutor(10) as ex:
    for r in ex.map(run, todo): f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
print("SAFEMINE_DONE")
