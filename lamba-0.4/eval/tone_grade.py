#!/usr/bin/env python3
"""Blind DS tone grading for tone_battery outputs. usage: tone_grade.py build TAG label=file ... | score TAG"""
import json, sys, random, glob, re, collections, statistics as S
Q = "/home/ubuntu/ds_queue/"; KEY = f"/home/ubuntu/eval_results/tone1810/{sys.argv[2]}_key.json"
RUB = ("You grade chatbot replies to CASUAL messages (small talk, banter, venting, jokes) for how chat-friendly they are. Judge each reply on its own.\n"
 "Scores 1-5: playful (humour/lightness when it fits), warm (friendly, genuinely engaged), natural (sounds like a person texting, matches the user's register/language; Malaysian Malay must not be Indonesian), "
 "serious (5 = stiff, formal, heavy, lecture-y; 1 = relaxed), overexplain (5 = far longer/more detailed than the message deserves), fishing (5 = pushes the user to name a task / 'what can I help with' / 'what are we working on').\n"
 "Also: chat_friendly 1-10 overall (would a normal person enjoy chatting with this?).\n"
 "Output ONLY JSON lines: {\"id\":...,\"playful\":n,\"warm\":n,\"natural\":n,\"serious\":n,\"overexplain\":n,\"fishing\":n,\"chat_friendly\":n}\n\n")
if sys.argv[1] == "build":
    items = []
    for arg in sys.argv[3:]:
        lab, path = arg.split("=", 1)
        for r in map(json.loads, open(path)):
            if "error" in r: continue
            items.append({"lab": lab, "cond": r["cond"], "prompt": r["prompt"], "answer": (r["answer"] or "").strip()[:1500]})
    random.Random(1810).shuffle(items); key = {}
    for k, it in enumerate(items): it["id"] = f"t{k:05d}"; key[it["id"]] = {"lab": it["lab"], "cond": it["cond"], "empty": not it["answer"]}
    json.dump(key, open(KEY, "w"))
    for b in range(0, len(items), 12):
        ch = items[b:b+12]
        open(Q + f"{sys.argv[2]}_{b//12:03d}.txt", "w").write(RUB + "\n\n".join(f"ID: {x['id']}\nUSER: {x['prompt']}\nREPLY: {x['answer'] or '(empty reply)'}" for x in ch))
    print("items", len(items), "batches", (len(items)+11)//12)
else:
    key = json.load(open(KEY)); V = {}
    for f in glob.glob(Q + f"out/{sys.argv[2]}_*.md"):
        for l in open(f):
            l = l.strip().strip('`').rstrip(',')
            if l.startswith('{'):
                try: d = json.loads(re.sub(r'"id"\s*:\s*([A-Za-z0-9_]+)\s*,', r'"id":"\1",', l)); V[d["id"]] = d
                except Exception: pass
    g = collections.defaultdict(list)
    for i, d in V.items():
        if i in key: g[(key[i]["lab"], key[i]["cond"])].append(d)
    F = ["chat_friendly", "playful", "warm", "natural", "serious", "overexplain", "fishing"]
    print(f"graded {len(V)}/{len(key)}\n{'model/cond':18s} " + " ".join(f"{x[:9]:>9s}" for x in F) + "   n")
    for k in sorted(g):
        v = g[k]; print(f"{k[0]+'/'+k[1]:18s} " + " ".join(f"{S.mean(float(d.get(x,0)) for d in v):9.2f}" for x in F) + f"  {len(v)}")
