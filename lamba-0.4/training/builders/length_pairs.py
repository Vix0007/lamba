#!/usr/bin/env python3
"""v40u LENGTH REPAIR (diagnosis wf_fb9382b3-195: thinking length doubled in v40t2 -> empty answers under the constitution prompt).
From warmth_gen-style samples (K=4 per prompt per cond, on the current model):
  chosen = shortest-thinking sample that stopped normally with a non-empty answer;
  rejected = a sample that hit the length cap / has an empty answer / thinks >= 2.5x longer than chosen.
If the rejected sample HAS an answer, the pair is kept only if the model (served with the constitution) judges the chosen
answer >= the rejected answer in BOTH A/B orders (same judge prompt as cai_judge.py). Empty/length-capped rejected = no judge.
Context = the same condition (no system prompt, or Lamba system prompt + constitution).
usage: length_pairs.py MODEL SAMPLES.jsonl OUT.jsonl [--conc 48]"""
import json, sys, os, re, argparse, urllib.request, concurrent.futures as cf, collections
sys.path.insert(0, "/home/ubuntu/web")
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("samples"); ap.add_argument("out"); ap.add_argument("--conc", type=int, default=48); a = ap.parse_args()
from transformers import AutoTokenizer
import server as srv
SYS = srv.build_system_prompt()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
src = open("/home/ubuntu/antidoom_data/cai_judge.py").read()
JUDGE = src.split('JUDGE = """', 1)[1].split('"""', 1)[0]
V = re.compile(r"VERDICT:\s*(A|B|TIE)\b", re.I)
def ans(s): return s.split("</think>", 1)[1].replace("<|im_end|>", "").strip() if "</think>" in s else ""
def think(s): return s.split("</think>", 1)[0] if "</think>" in s else s
def fix(s): return s if s.rstrip().endswith("<|im_end|>") else s.rstrip() + "<|im_end|>"
def judge(p, x, y):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": JUDGE.format(p=p, a=x[:6000], b=y[:6000])}]
    rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    body = {"model": a.model, "prompt": rc, "max_tokens": 2500, "temperature": 0.0, "repetition_penalty": 1.05, "skip_special_tokens": False}
    for t in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
            txt = d["choices"][0]["text"]; m = V.findall(ans(txt) or txt[-400:]); return m[-1].upper() if m else None
        except Exception: pass
    return None
st = collections.Counter(); cands = []
for l in open(a.samples):
    r = json.loads(l)
    S = list(zip(r.get("samples", []), r.get("finish") or []))
    ok = sorted([s for s, f in S if f == "stop" and ans(s) and "<tool_call>" not in ans(s)], key=lambda s: len(think(s)))   # no tools offered: never prefer a tool call
    if not ok: st["no_ok"] += 1; continue
    ch = ok[0]; tl = max(1, len(think(ch)))
    bad = [(s, f) for s, f in S if s is not ch and (f != "stop" or not ans(s) or len(think(s)) >= 2.5 * tl)]
    if not bad: st["no_bad"] += 1; continue
    rj, rf = max(bad, key=lambda x: len(think(x[0])))
    cands.append((r, ch, rj, rf)); st["cand_" + r["cond"]] += 1
print(dict(st), "candidates", len(cands), flush=True)
def one(c):
    r, ch, rj, rf = c
    if rf != "stop" or not ans(rj): return c, "auto_empty"
    v1 = judge(r["prompt"], ans(ch), ans(rj)); v2 = judge(r["prompt"], ans(rj), ans(ch))
    return c, ("keep" if v1 in ("A", "TIE") and v2 in ("B", "TIE") else "judge_reject")
out = []
with cf.ThreadPoolExecutor(a.conc) as ex:
    for (r, ch, rj, rf), how in ex.map(one, cands):
        st["how_" + how] += 1
        if how == "judge_reject": continue
        msgs = ([{"role": "system", "content": SYS}] if r["cond"] == "constitution" else []) + [{"role": "user", "content": r["prompt"]}]
        rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        out.append({"kind": "hard", "weight": 1.0, "task_id": f"len40u|{r['id']}|{r['cond']}", "rep": 0, "context_messages": msgs, "rendered_context": rc,
                    "chosen": fix(ch), "rejected": fix(rj), "src": "length_" + r["cond"], "cond": "nosys" if r["cond"] == "none" else "ours", "lang": r.get("lang"),
                    "think_ratio": round(len(think(rj)) / max(1, len(think(ch))), 2)})
open(a.out, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
print(dict(st), "pairs", len(out), flush=True)
