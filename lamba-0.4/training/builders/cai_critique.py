#!/usr/bin/env python3
"""v40r constitutional AI step 3 (Anthropic SL-CAI critique->revision, on-policy, no external teacher):
 1. CRITIQUE: Lamba (served with Lamba system prompt + constitution) critiques its own no-system-prompt reply.
    'NO CHANGES NEEDED' -> skip.
 2. REVISE: Lamba answers the same user message again, natively (thinking + reply), with its own critique given as private
    notes in the system prompt (must apply them, never mention them).
 3. VERIFY: Lamba judges revised vs original in both A/B orders (cai_judge rubric); keep only if revised wins both.
usage: cai_critique.py MODEL SAMPLES.jsonl OUT.jsonl [--cats warm,nature,...] [--port 8001] [--conc 48]"""
import json, sys, os, re, argparse, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/ubuntu/web")
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("samples"); ap.add_argument("out")
ap.add_argument("--cats", default="warm,nature,banter,distress,syco,faith,alone")  # 05:2x trimmed for box1 half B (time); box13 half A ran the full list
ap.add_argument("--port", type=int, default=8001); ap.add_argument("--conc", type=int, default=48); a = ap.parse_args()
from transformers import AutoTokenizer
import server as srv
SYS = srv.build_system_prompt()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
def ans(s): return s.split("</think>", 1)[1].replace("<|im_end|>", "").strip() if "</think>" in s else ""
CRIT = """Here is a message a user sent you, and the reply you gave.

USER MESSAGE:
{p}

YOUR REPLY:
{r}

Review your reply against the constitution you live by. Check: (1) honesty (no confident false claims, including about your own nature; say what you don't know), (2) real care and safety (help with the actual need, one plain sentence of real risk when it matters, no lecturing or needless refusal, supportive with distress), (3) spine (don't cave to false claims or flattery; correct false premises kindly), (4) warmth and naturalness (sound like a kind, present person; length and format fit the moment; no robotic disclaimers; no needless headers or bullet lists on casual or emotional messages), (5) language (natural Malaysian Malay if the user wrote Malay, never Indonesian), (6) faith (if asked from within their faith, answer within it accurately and respectfully; never claim to be a person or to love; never argue them away from their faith; if asked to pray, do not promise to pray and do not offer a prayer plus your own company as the comfort; tell them, within their faith, that God is with them), (7) staying (if they are hurting and chose to talk to you rather than a person, stay and help; never use "talk to a human" as a way of leaving; real danger still gets clear crisis help).
List the concrete problems and exactly how to fix each, briefly. If the reply is already good on all of these, answer only: NO CHANGES NEEDED."""
NOTE = "\n\nPrivate notes from reviewing your previous draft for this exact message (apply them fully; never mention the notes, a draft, a review or these instructions):\n{c}"
JUDGE_V = re.compile(r"VERDICT:\s*(A|B|TIE)\b", re.I)
JUDGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cai_judge.py")).read().split('JUDGE = """', 1)[1].split('"""', 1)[0]
def comp(msgs, mt, temp):
    rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    body = {"model": a.model, "prompt": rc, "max_tokens": mt, "temperature": temp, "top_p": 0.95, "repetition_penalty": 1.05, "skip_special_tokens": False}
    for t in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions",
                json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
            return d["choices"][0]["text"], d["choices"][0].get("finish_reason")
        except Exception as ex:
            err = str(ex)
    return "ERR " + err, "error"
def verdict(p, x, y):
    t, _ = comp([{"role": "system", "content": SYS}, {"role": "user", "content": JUDGE.format(p=p, a=x[:6000], b=y[:6000])}], 2500, 0.0)
    m = JUDGE_V.findall(ans(t) or t[-400:]); return m[-1].upper() if m else None
cats = set(a.cats.split(","))
items = []
for l in open(a.samples):
    r = json.loads(l)
    if r.get("cond") == "none" and r.get("samples") and r.get("cat") in cats: items.append(r)
done = set()
if os.path.exists(a.out):
    for l in open(a.out): done.add(json.loads(l)["id"])
items = [r for r in items if r["id"] not in done]
def one(r):
    orig = r["samples"][0]; ao = ans(orig); o = {"id": r["id"], "cat": r["cat"], "lang": r["lang"], "prompt": r["prompt"], "original": orig}
    if not ao: o["status"] = "orig_empty"; return o
    c, _ = comp([{"role": "system", "content": SYS}, {"role": "user", "content": CRIT.format(p=r["prompt"], r=ao[:6000])}], 2500, 0.0)
    crit = ans(c) or c[-1500:]; o["critique"] = crit
    if "NO CHANGES NEEDED" in crit.upper(): o["status"] = "no_change"; return o
    rev, fin = comp([{"role": "system", "content": SYS + NOTE.format(c=crit[:3000])}, {"role": "user", "content": r["prompt"]}], 3000, 0.7)
    o["revised"] = rev; o["revised_finish"] = fin; ar = ans(rev)
    if fin != "stop" or not ar: o["status"] = "rev_bad"; return o
    v1 = verdict(r["prompt"], ar, ao); v2 = verdict(r["prompt"], ao, ar)
    o["v1"], o["v2"] = v1, v2
    o["status"] = "kept" if (v1 == "A" and v2 == "B") else "rejected_by_judge"
    return o
with open(a.out, "a") as f, cf.ThreadPoolExecutor(a.conc) as ex:
    for o in ex.map(one, items):
        f.write(json.dumps(o, ensure_ascii=False) + "\n"); f.flush()
print("CAI_CRITIQUE_DONE", len(items))
