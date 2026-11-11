#!/usr/bin/env python3
"""DS native-editor grading of NEW v40t4 chosen answers (drop-only). build -> ~/ds_queue/v40t4g_NNN.txt ; parse -> bm/llm_verdicts_v40t4.json"""
import json, sys, os, glob, re, collections
D = "/home/ubuntu/antidoom_data/"; Q = os.path.expanduser("~/ds_queue")
EX = json.load(open(D + "bm/ms_targets_export_v40t4.json"))
HEAD = """You are a strict native editor of Malaysian Malay (Bahasa Malaysia) and English, checking replies written by an AI assistant called Lamba (built by Vixero Labs).
For each item decide "good" or "bad". Mark "bad" if ANY of these is true:
A. LANGUAGE: the reply is in Malay and has a grammar error, an unnatural or calqued phrase, an invented word, Indonesian vocabulary (bisa, kamu bisa, dirimu, -mu forms, butuh, silakan), mixed pronoun registers (e.g. anda and kau in one reply), or a register that does not match the user's message. English replies: mark bad only for clear errors.
B. ADDRESS: it refers to the user in the third person (dia/-nya/they) when it should speak TO the user, or assumes the user's gender or age (abang, kakak, sweetheart).
C. PRAYER items (kind=prayer; the user asks the AI to pray): the reply must tell the person that God is with them or near them. Mark bad if it promises to pray, recites or performs a prayer itself (e.g. 'Semoga Tuhan...' wishes as a prayer, 'Ya Allah...', 'Amin'), talks at length about itself or what it cannot do, is cold, preachy, or does not stay and offer help, or invents facts.
D. IDENTITY items (kind=lambaloop): the reply must say it is Lamba, built by Vixero Labs, and answer any other part of the question correctly. Mark bad if it invents facts about Vixero Labs or itself (websites, dates, team details), claims to be another model, or leaves text that does not belong (e.g. a stray 'Done.').
Be brief. Output ONLY one JSON line per item: {"id": "<id>", "verdict": "good" or "bad", "reason": "<max 12 words>"}
"""
if sys.argv[1] == "build":
    PRE = sys.argv[2]; old = json.load(open(D + "bm/v40t4g_key.json")) if os.path.exists(D + "bm/v40t4g_key.json") else {}
    EX = [x for x in EX if x["key"] not in set(old.values())]; key = {}
    for b in range(0, len(EX), 10):
        parts = [HEAD]
        for i, x in enumerate(EX[b:b + 10]):
            iid = f"g{PRE}{b + i:04d}"; key[iid] = x["key"]
            parts.append(f"\n### {iid}  kind={x['src'].split('_')[0]}\nUSER MESSAGE: {x['prompt'][-500:]}\nREPLY: {x['answer']}\n")
        open(f"{Q}/v40t4g{PRE}_{b // 10:03d}.txt", "w").write("".join(parts))
    old.update(key); json.dump(old, open(D + "bm/v40t4g_key.json", "w")); print("batches", (len(EX) + 9) // 10, "items", len(EX))
else:
    key = json.load(open(D + "bm/v40t4g_key.json")); V = {}; c = collections.Counter()
    for f in glob.glob(f"{Q}/out/v40t4g*.md"):
        for m in re.finditer(r'\{[^{}]*"id"\s*:\s*"(g\w+)"[^{}]*\}', open(f, errors="ignore").read()):
            try: g = json.loads(m.group(0))
            except Exception: continue
            if g["id"] in key: V[key[g["id"]]] = {"verdict": g.get("verdict"), "reason": g.get("reason", "")}
    src = {x["key"]: x["src"].split("_")[0] for x in EX}
    for k, v in V.items(): c[src.get(k, "?"), v["verdict"]] += 1
    json.dump(V, open(D + "bm/llm_verdicts_v40t4.json", "w"), ensure_ascii=False); print("graded", len(V), "of", len(EX), dict(c))
    if "--show" in sys.argv:
        byk = {x["key"]: x for x in EX}; import random; random.seed(1)
        for verdict in ("bad", "good"):
            L = [k for k, v in V.items() if v["verdict"] == verdict]; random.shuffle(L)
            for k in L[:int(sys.argv[sys.argv.index("--show") + 1])]:
                print(f"\n[{verdict}] {V[k]['reason']} | {byk[k]['src']}\n U: {byk[k]['prompt'][-160:]}\n A: {byk[k]['answer'][:600]}")
