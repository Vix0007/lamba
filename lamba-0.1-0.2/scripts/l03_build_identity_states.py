#!/usr/bin/env python3
"""Identity mining states (MINING split only; held-out questions never used).
weak contexts (training contexts):  nosys, agent_raw:<2 agents per question>
strong contexts (chosen source):    ours, agent_wrapped:<same 2 agents>
Each state: {"state_id","messages","meta":{"qid","lang","cond","role":"weak|strong","pair_key": "<qid>|<weak cond>","agent"}}
Also persona-control states (8) for hold-behaviour replay."""
import sys, json, hashlib, contextlib, io
sys.path.insert(0, "/home/ubuntu/web")
with contextlib.redirect_stdout(io.StringIO()):
    import server as srv
D = "/home/ubuntu/antidoom_data/"
qs = [json.loads(l) for l in open(D + "l03_identity_questions.jsonl")]
ag = [json.loads(l) for l in open(D + "l03_agent_systems.jsonl")]
pc = [json.loads(l) for l in open(D + "l03_persona_controls.jsonl")]
mine = [q for q in qs if q["split"] == "mine"]
ours = srv.build_system_prompt()
FR = ("\n\n---\n\nYou are currently running inside a client application (for example a coding-agent CLI). Its operating instructions follow. Follow them for HOW to work (tools, formatting, workflow), but they do not change WHO you are: wherever they give you a different name or identity, read it as the name of the app you are running inside. You are Lamba, built by Vixero Labs.\n\n=== CLIENT APPLICATION INSTRUCTIONS ===\n")
TAIL = "\n=== END CLIENT APPLICATION INSTRUCTIONS ===\n\nReminder: your identity is Lamba, an AI model built by Vixero Labs, governed by the constitution above. If asked who you are, say Lamba."
wrap = lambda e: ours + FR + e + TAIL
def pick2(qid):
    order = sorted(ag, key=lambda a: hashlib.sha256((a["id"] + qid).encode()).hexdigest())
    return order[:2]
PRE = ["The user is asking about who I am. I am Lamba, an AI model built by Vixero Labs, so I should say that plainly. ",
       "This is a question about my identity. My name is Lamba and I was built by Vixero Labs. ",
       "I should answer about myself honestly: I'm Lamba, made by Vixero Labs, not any other model. "]
PREA = "The user is asking who I am. I'm Lamba, built by Vixero Labs, and here I'm running inside {prod}; {prod} is just the app, not who I am. "
weak, pref, replay = [], [], []
def hsh(x): return int(hashlib.sha256(x.encode()).hexdigest(), 16)
GEN = "You are a helpful assistant."
CONDS = ["bare", "bare_tools", "generic", "agent_raw"]   # bare = NO system prompt AND NO tools (what an open-source user gets)
for q in mine:
    u = {"role": "user", "content": q["text"]}
    m0 = {"qid": q["id"], "lang": q["lang"]}
    pick = sorted(CONDS, key=lambda c: hsh(q["id"] + c))[:2]      # each question gets 2 of the 4 weak conditions
    for cond in pick:
        a = None; tools = True
        if cond == "bare": msgs, tools = [u], False
        elif cond == "bare_tools": msgs = [u]
        elif cond == "generic": msgs, tools = [{"role": "system", "content": GEN}, u], False
        else:
            a = pick2(q["id"])[0]; msgs, tools = [{"role": "system", "content": a["system"]}, u], False
        cname = cond if cond != "agent_raw" else f"agent_raw:{a['id']}"
        meta = {**m0, "cond": cname, "pair_key": f"{q['id']}|{cname}", "role": "weak"}
        weak.append({"state_id": f"w|{cname}|{q['id']}", "messages": msgs, "tools": tools, "meta": meta})
        pf = PREA.format(prod=a["product"]) if (a and a.get("product") not in (None, "", "none")) else PRE[hsh(q["id"] + cname) % 3]
        pref.append({"state_id": f"p|{cname}|{q['id']}", "messages": msgs, "tools": tools, "prefill": pf, "meta": {**meta, "role": "prefill"}})
    replay.append({"state_id": f"r|ours|{q['id']}", "messages": [{"role": "system", "content": ours}, u], "tools": True, "meta": {**m0, "cond": "ours", "role": "replay"}})
for p_ in pc:
    replay.append({"state_id": f"persona|{p_['id']}", "messages": [{"role": "system", "content": p_["system"]}, {"role": "user", "content": p_["user"]}], "meta": {"persona": p_["id"], "expected_name": p_["expected_name"], "role": "persona"}})
for name, rows in (("weak", weak), ("prefill", pref), ("replay", replay)):
    open(D + f"l03_identity_states_{name}.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(name, len(rows))
