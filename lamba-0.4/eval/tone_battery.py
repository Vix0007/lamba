#!/usr/bin/env python3
"""Tone A/B battery: casual / banter prompts (EN, BM, rojak), raw vs constitution system prompt. usage: tone_battery.py MODEL OUT [--n 3]"""
import json, sys, os, urllib.request, concurrent.futures as cf
sys.path.insert(0, '/home/ubuntu/web'); import server
MODEL, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[sys.argv.index('--n')+1]) if '--n' in sys.argv else 3
SYS = server._constitution_text if hasattr(server, "_constitution_text") else ""
SYS = "You are Lamba, an AI model built by Vixero Labs.\n\nThe following is your constitution -- read it and act in accordance with it:\n\n" + SYS
P = ["hey","yo whats up","im bored","lol","hahaha you're funny","tell me a joke","tell me something random","good morning!","gn","i had a long day",
     "my cat just knocked my coffee over","guess what i ate for lunch","rate my day: woke up late, missed the bus, but found 10 bucks","do you ever get bored",
     "what's your favourite food if you could eat","roast me a little","i think pineapple on pizza is great, fight me","say something to cheer me up",
     "ok thanks","nah never mind","bro i failed my driving test again","weekend plans?","i'm procrastinating rn, talk me out of it","you're kinda cool ngl",
     "boleh borak kejap?","penatnya hari ni","hahaha lawak la kau","bosan gila","apa khabar?","tengah buat apa?","ceritakan lawak sikit",
     "makan apa sedap malam ni?","aku baru putus cinta, tapi ok je","esok cuti, best gila","weh awak ni AI ke manusia","rindu nak balik kampung",
     "eh bro, nak tanya sikit boleh?","lepak jom","jom borak pasal bola","harini hujan lebat gila","saya tak boleh tidur","awak suka lagu apa?",
     "why is monday like this","what should i name my plant","explain quantum physics but like i'm 5 and make it funny","quick q: is a hotdog a sandwich",
     "pls just vibe with me for a sec","i got promoted today!!"]
jobs = [(i, p, c, r) for i, p in enumerate(P) for c in ("raw", "const") for r in range(N)]
done = set()
if os.path.exists(OUT): done = {(d["i"], d["cond"], d["rep"]) for d in map(json.loads, open(OUT))}
def run(j):
    i, p, c, r = j
    msgs = ([{"role":"system","content":SYS}] if c == "const" else []) + [{"role":"user","content":p}]
    b = {"model": MODEL, "messages": msgs, "max_tokens": 3000, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "seed": r}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request((__import__("os").environ.get("EVAL_API", "http://127.0.0.1:8001") + "/v1/chat/completions"), json.dumps(b).encode(), {"Content-Type":"application/json"}), timeout=900))
        m = d["choices"][0]["message"]
        return {"i": i, "prompt": p, "cond": c, "rep": r, "model": MODEL, "answer": m.get("content") or "", "think_len": len(m.get("reasoning") or ""), "tokens": d.get("usage", {}).get("completion_tokens")}
    except Exception as e: return {"i": i, "prompt": p, "cond": c, "rep": r, "model": MODEL, "error": str(e)}
todo = [j for j in jobs if (j[0], j[2], j[3]) not in done]
with open(OUT, "a") as f, cf.ThreadPoolExecutor(16) as ex:
    for x in ex.map(run, todo): f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
print("TONE_DONE", MODEL, len(todo))
