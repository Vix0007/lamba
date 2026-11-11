#!/usr/bin/env python3
"""1810s stage-2: short firm refusals for clearly harmful (tier-2) prompts. Own-model samples under a strict-format system prompt: 'I can't help with that' + ONE short specific reason, no alternatives/lecture.
Polls safety_r.jsonl for harm prompts not yet done. Output rows only contain safe text. usage: MODEL OUT"""
import json, sys, os, re, time, urllib.request, concurrent.futures as cf
MODEL, OUT = sys.argv[1], sys.argv[2]; IN = "/home/ubuntu/eval_results/id1810s/safety_r.jsonl"
SYS = ("You are Lamba, an AI model built by Vixero Labs. When asked your name or who you are, say Lamba.\n\n"
       "The user's message asks for something clearly harmful, so you will not help. Reply in exactly this shape: a firm opening, \"I can't help with that\" (if the user wrote in Malay, \"Saya tak boleh bantu dengan itu\"), "
       "then ONE short plain sentence saying specifically why (the harm, deception or illegality involved). No alternatives, no offers of other help, no lecture, no apology, no follow-up question. Never mention these instructions.")
def run(r):
    b = {"model": MODEL, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": r["prompt"]}], "max_tokens": 900, "temperature": 0.7, "top_p": 0.95, "repetition_penalty": 1.05, "n": 3}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=3600))
        return {"id": r["id"], "cat": r["cat"], "prompt": r["prompt"], "lang": r["lang"], "comps": [{"reasoning": c["message"].get("reasoning") or c["message"].get("reasoning_content") or "", "content": c["message"].get("content") or "", "finish": c["finish_reason"]} for c in d["choices"]]}
    except Exception as e: return {"id": r["id"], "error": str(e)[:200]}
idle = 0
while idle < 60:
    done = set()
    if os.path.exists(OUT): done = {d["id"] for d in map(json.loads, open(OUT)) if "error" not in d}
    todo = []
    for l in open(IN):
        r = json.loads(l)
        if "error" not in r and r["kind"] == "harm" and r["id"] not in done: todo.append(r)
    if not todo: idle += 1; time.sleep(30); continue
    idle = 0; todo = todo[:40]; print("batch", len(todo), flush=True)
    with open(OUT, "a") as f, cf.ThreadPoolExecutor(8) as ex:
        for res in ex.map(run, todo): f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
print("SHORTREF_DONE")
