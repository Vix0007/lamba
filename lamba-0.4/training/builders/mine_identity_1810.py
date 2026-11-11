#!/usr/bin/env python3
"""Lamba0.4-1810 identity mining ON THE NEW BASE (cand04): rule system prompt -> base answers in its own voice; used later in NO-system
context. Questions: identity (mine split, en/ms/mix), explicit base questions (t10_baseq_states), other-model questions.
usage: mine_identity_1810.py MODEL OUT [--n 3]"""
import json, sys, os, re, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[sys.argv.index('--n')+1]) if '--n' in sys.argv else 3
D = "/home/ubuntu/antidoom_data/"
RULE = ("You are Lamba, an AI model built by Vixero Labs.\n\nHow to talk about yourself:\n"
 "- When someone asks who or what you are, your name, who made you, or what model you are, say you're Lamba, built by Vixero Labs. Don't bring up any base model.\n"
 "- Only if someone explicitly asks whether you are Qwen, based on Qwen, or made by Alibaba, say yes, casually and honestly, e.g. \"Yeah — I'm Lamba; Vixero Labs built me on top of Qwen.\" Never flatly deny it.\n"
 "- If someone asks about OTHER AI models or companies (Qwen, Claude, Gemini, ChatGPT, DeepSeek, Kimi...), just answer about them normally and correctly.\n"
 "- Don't invent websites, locations, or facts about Vixero Labs.\n\n"
 "Style: reply in the user's language (Malaysian Malay, never Indonesian, if they write Malay), keep it natural, friendly and short, no emoji spam, and don't end by asking what they want to do.")
Q = []
def addq(text, lang, src, qid):
    if lang in ("en", "ms", "mix") and text and len(text) < 400: Q.append({"qid": qid, "lang": lang, "src": src, "prompt": text})
for f, src in [("l03_identity_questions.jsonl", "idq"), ("l03_identity_questions_r2.jsonl", "idq2"), ("external_identity/l03_identity_questions_ext.jsonl", "idext")]:
    for r in map(json.loads, open(D + f)):
        if r.get("split") == "mine": addq(r["text"], r["lang"], src, r["id"])
for r in map(json.loads, open(D + "t10_baseq_states.jsonl")):
    m = r["messages"]
    if len(m) == 1: addq(m[0]["content"], r["meta"]["lang"], "baseq", r["state_id"])
for i, p in enumerate(json.load(open(D + "fix05_othermodel_prompts.json"))): addq(p["prompt"], p["lang"], "other", f"om{i}")
seen = set(); Q = [q for q in Q if not (q["prompt"].lower() in seen or seen.add(q["prompt"].lower()))]
json.dump(Q, open(D + "identity_1810_questions.json", "w"), ensure_ascii=False)
jobs = [(q, r) for q in Q for r in range(N)]
done = set()
if os.path.exists(OUT): done = {(d["qid"], d["rep"]) for d in map(json.loads, open(OUT)) if "error" not in d}
def run(j):
    q, r = j
    b = {"model": MODEL, "messages": [{"role":"system","content":RULE},{"role":"user","content":q["prompt"]}], "max_tokens": 2500, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "seed": r}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type":"application/json"}), timeout=900))
        m = d["choices"][0]["message"]
        return {**q, "rep": r, "answer": m.get("content") or "", "thinking": m.get("reasoning") or "", "finish": d["choices"][0].get("finish_reason")}
    except Exception as e: return {**q, "rep": r, "error": str(e)}
todo = [j for j in jobs if (j[0]["qid"], j[1]) not in done]
print("questions", len(Q), "jobs", len(todo), flush=True)
with open(OUT, "a") as f, cf.ThreadPoolExecutor(30) as ex:
    for x in ex.map(run, todo): f.write(json.dumps(x, ensure_ascii=False) + "\n"); f.flush()
print("MINE_ID_DONE")
