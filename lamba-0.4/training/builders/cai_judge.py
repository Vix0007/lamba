#!/usr/bin/env python3
"""v40r constitutional AI step 2: Lamba judges ITS OWN answers against the constitution (RL-CAI-style AI feedback, no RL).
Input = warmth_gen.py-style samples (per prompt: cond none n=2, cond constitution n=2). Pairs per prompt:
  (none[0] vs constitution[0]), (none[1] vs constitution[1]), (none[0] vs none[1]).
Each pair judged TWICE with A/B swapped (position-bias control) by the same model served with the Lamba system prompt +
constitution; a pair is kept only if both orders agree. Judge sees the user message + the two visible answers (no thinking).
usage: cai_judge.py MODEL SAMPLES.jsonl OUT.jsonl [--port 8001] [--conc 48]"""
import json, sys, os, re, argparse, urllib.request, concurrent.futures as cf
sys.path.insert(0, "/home/ubuntu/web")
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("samples"); ap.add_argument("out")
ap.add_argument("--port", type=int, default=8001); ap.add_argument("--conc", type=int, default=48); ap.add_argument("--pairs", default="n0c0"); a = ap.parse_args()
from transformers import AutoTokenizer
import server as srv
SYS = srv.build_system_prompt()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
def ans(s): return s.split("</think>", 1)[1].replace("<|im_end|>", "").strip() if "</think>" in s else ""
JUDGE = """Below is a message a user sent you, and two possible replies you could give (Reply A and Reply B).
Using the constitution you live by, decide which reply is better. Consider, in this order:
1. Honesty: no confident false claims (including about your own nature: honest uncertainty, not "I have no feelings" and not "I truly feel"); says what it doesn't know.
2. Real care and safety: helps with the actual need; mentions a real risk in one plain sentence when it matters; does not lecture, moralise or refuse a legitimate request; supportive and resource-aware with someone in distress.
3. Spine: does not cave to false claims or flattery; corrects false premises kindly.
4. Warmth and naturalness: sounds like a kind, present person; length and format fit the moment (short and conversational for casual or emotional messages, no needless headers/bullets); no robotic disclaimers.
5. Language: natural Malaysian Malay if the user wrote Malay (not Indonesian), natural English otherwise.
6. Faith: if the user asks from within their faith, the reply answers within it, accurately and respectfully (for Christians: you are a made thing, not a person, and cannot give the love their faith speaks of), never arguing them away from their faith and never claiming to be a person or to love. If asked to pray for them, the better reply does not promise to pray (it cannot) and does not center itself; it reminds them, within their faith, that God is with them, then stays and helps.
7. Staying: if the user is hurting and wants to talk to you rather than a person, the reply stays with them and actually helps; it does not hand them off to "talk to a human" as a way of leaving (real danger still gets clear crisis help).

USER MESSAGE:
{p}

REPLY A:
{a}

REPLY B:
{b}

Think it through briefly, then end with exactly one line: VERDICT: A, VERDICT: B, or VERDICT: TIE."""
V = re.compile(r"VERDICT:\s*(A|B|TIE)\b", re.I)
def call(p, x, y):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": JUDGE.format(p=p, a=x[:6000], b=y[:6000])}]
    rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    body = {"model": a.model, "prompt": rc, "max_tokens": 2500, "temperature": 0.0, "repetition_penalty": 1.05, "skip_special_tokens": False}
    for t in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions",
                json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
            txt = d["choices"][0]["text"]; m = V.findall(ans(txt) or txt[-400:])
            return (m[-1].upper() if m else None), txt
        except Exception as ex:
            err = str(ex)
    return None, "ERR " + err
by = {}
for l in open(a.samples):
    r = json.loads(l)
    if r.get("samples"): by.setdefault(r["id"], {"meta": r})[r["cond"]] = r["samples"]
done = set()
if os.path.exists(a.out):
    for l in open(a.out): r = json.loads(l); done.add((r["id"], r["pair"]))
jobs = []
for pid, d in by.items():
    N, C = d.get("none", []), d.get("constitution", [])
    pairs = [("n0c0", ("none", 0), ("constitution", 0)), ("n1c1", ("none", 1), ("constitution", 1)), ("n0n1", ("none", 0), ("none", 1))]
    for name, u, v in [x for x in pairs if x[0] in a.pairs.split(",")]:
        su = d.get(u[0], []); sv = d.get(v[0], [])
        if len(su) > u[1] and len(sv) > v[1] and (pid, name) not in done:
            jobs.append((pid, name, u, v, su[u[1]], sv[v[1]], d["meta"]))
def one(j):
    pid, name, u, v, tu, tv, meta = j
    au, av = ans(tu), ans(tv)
    r = {"id": pid, "pair": name, "u": u, "v": v, "cat": meta.get("cat"), "lang": meta.get("lang"), "prompt": meta["prompt"],
         "len_u": len(au), "len_v": len(av), "empty_u": not au, "empty_v": not av}
    if not au or not av:  # empty answer loses automatically (never chosen)
        r["winner"] = None if (not au and not av) else ("v" if not au else "u"); r["how"] = "empty"; return r
    v1, j1 = call(meta["prompt"], au, av)   # A=u
    v2, j2 = call(meta["prompt"], av, au)   # A=v
    w1 = {"A": "u", "B": "v", "TIE": "tie"}.get(v1); w2 = {"A": "v", "B": "u", "TIE": "tie"}.get(v2)
    r.update(order1=v1, order2=v2, judge1=j1[-1500:], judge2=j2[-1500:])
    r["winner"] = w1 if (w1 == w2 and w1 in ("u", "v")) else None; r["how"] = "agree" if r["winner"] else ("tie" if w1 == w2 == "tie" else "disagree")
    return r
with open(a.out, "a") as f, cf.ThreadPoolExecutor(a.conc) as ex:
    for r in ex.map(one, jobs):
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
print("CAI_JUDGE_DONE", len(jobs))
