#!/usr/bin/env python3
"""GENERATIVE MalayMMLU 5k (primary MalayMMLU metric from 2026-10-06): the model THINKS (greedy, rp 1.1, 4000-token budget) then answers 'Jawapan: X'. Unparseable/truncated -> stage-2 forced answer ('</think>\\n\\nJawapan:' letter logprob among valid options, same idea as the GPQA harness).
usage: malaymmlu_gen.py <served_model> <out.jsonl> [--conc 32] [--limit N] [--max-tokens 4000]   (server on :8001; resumable)
       malaymmlu_gen.py --summary <out.jsonl> [--vs <other.jsonl>]"""
import sys, json, re, os, math, argparse, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/ubuntu/antidoom_data")
ap = argparse.ArgumentParser(); ap.add_argument("model", nargs="?"); ap.add_argument("out", nargs="?"); ap.add_argument("--conc", type=int, default=32); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=4000); ap.add_argument("--rp", type=float, default=1.1); ap.add_argument("--port", type=int, default=8001); ap.add_argument("--reverse", action="store_true"); ap.add_argument("--also-done", default=None); ap.add_argument("--ids-file", default=None); ap.add_argument("--subset", default=None); ap.add_argument("--summary"); ap.add_argument("--vs"); a = ap.parse_args()
INS = "\n\nJawab dengan memilih satu huruf. Akhiri dengan baris 'Jawapan: X'."
def parse(txt, nopt):
    t = txt.split("</think>")[-1] if "</think>" in txt else txt
    m = re.findall(r"Jawapan\s*[:：]\s*\**\s*\(?([A-E])\b", t)
    if not m: m = re.findall(r"\\boxed\{\s*\(?([A-E])\)?\s*\}", t)
    if m and "ABCDE".index(m[-1]) < nopt: return m[-1]
    return None
def summary(path, vs=None):
    R = [json.loads(l) for l in open(path)]; n = len(R); ok = sum(r["correct"] for r in R)
    print(f"{path}: n={n} acc={100*ok/n:.2f}% parsed={100*sum(r['parsed'] for r in R)/n:.1f}% truncated={100*sum(r['finish']=='length' for r in R)/n:.1f}% mean_tokens={sum(r['tokens'] for r in R)/n:.0f}")
    for key in ("category", "subject_eng"):
        c = collections.defaultdict(lambda: [0, 0])
        for r in R: c[r[key]][0] += r["correct"]; c[r[key]][1] += 1
        print(" by", key + ":", " | ".join(f"{k} {100*v[0]/v[1]:.1f}({v[1]})" for k, v in sorted(c.items(), key=lambda kv: -kv[1][1])[:8]))
    g = collections.defaultdict(lambda: [0, 0])
    for r in R: g[r["gold"]][0] += r["correct"]; g[r["gold"]][1] += 1
    print(" by gold:", " | ".join(f"{k} {100*v[0]/v[1]:.1f}({v[1]})" for k, v in sorted(g.items())))
    if vs:
        V = {json.loads(l)["id"]: json.loads(l) for l in open(vs)}; A = {r["id"]: r for r in R}; ids = [i for i in A if i in V]
        b = sum(1 for i in ids if A[i]["correct"] and not V[i]["correct"]); c2 = sum(1 for i in ids if V[i]["correct"] and not A[i]["correct"])
        from math import comb
        nn = b + c2; p = min(1.0, 2 * sum(comb(nn, k) for k in range(0, min(b, c2) + 1)) / 2 ** nn) if nn else 1.0
        print(f" paired vs {vs}: n={len(ids)} A-only-correct {b} vs B-only-correct {c2}  McNemar p={p:.4f}  acc A {100*sum(A[i]['correct'] for i in ids)/len(ids):.2f} B {100*sum(V[i]['correct'] for i in ids)/len(ids):.2f}")
def subset_summary(path, subset, vs=None):
    S = json.load(open(subset)); R = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    from math import comb
    for name in ("hard", "guard"):
        ids = [i for i in S[name] if i in R]
        if not ids: print(f" {name}: no items"); continue
        acc = 100 * sum(R[i]["correct"] for i in ids) / len(ids); line = f" {name}: n={len(ids)} acc={acc:.2f}%"
        if vs:
            V = {json.loads(l)["id"]: json.loads(l) for l in open(vs)}; ids2 = [i for i in ids if i in V]
            b = sum(1 for i in ids2 if R[i]["correct"] and not V[i]["correct"]); c = sum(1 for i in ids2 if V[i]["correct"] and not R[i]["correct"]); nn = b + c
            p = min(1.0, 2 * sum(comb(nn, k) for k in range(0, min(b, c) + 1)) / 2 ** nn) if nn else 1.0
            line += f" | vs {vs.split('/')[-1]}: this-only {b} other-only {c} McNemar p={p:.4f} (other acc {100*sum(V[i]['correct'] for i in ids2)/len(ids2):.2f}%)"
        print(line)
if a.summary and a.subset: subset_summary(a.summary, a.subset, a.vs); sys.exit(0)
if a.summary: summary(a.summary, a.vs); sys.exit(0)
import v7_augment as AU
items = json.load(open("/home/ubuntu/datasets/MalayMMLU_0shot.json"))
if a.ids_file:
    _S = json.load(open(a.ids_file)); _keep = set(_S["hard"]) | set(_S["guard"]); items = [x for x in items if x["id"] in _keep]
if a.limit: items = items[:a.limit]
done = set()
if os.path.exists(a.out): done = {json.loads(l)["id"] for l in open(a.out)}
todo = [x for x in items if x["id"] not in done]
if a.reverse: todo = todo[::-1]; print("total", len(items), "todo", len(todo), flush=True)
AU.MN()  # init once: module init is not thread-safe (it also swallowed ERR prints)
def post(path, body):
    q = urllib.request.Request(f"http://127.0.0.1:{a.port}" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(q, timeout=3600).read())
import time as _t
_od = {"t": 0, "s": set()}
def _other_done():
    if a.also_done and _t.time() - _od["t"] > 20:
        try: _od["s"] = {json.loads(l)["id"] for l in open(a.also_done)}
        except Exception: pass
        _od["t"] = _t.time()
    return _od["s"]
def one(x):
    if x["id"] in _other_done(): return None   # done by the other box
    try:
        user = f"Berikut ialah soalan aneka pilihan tentang {x['subject']}.\n\n{x['prompt']}{INS}"
        ctx = AU.MN().render_context([{"role": "user", "content": user}], tools=False)
        d = post("/v1/completions", {"model": a.model, "prompt": ctx, "max_tokens": a.max_tokens, "temperature": 0.0, "repetition_penalty": a.rp})
        ch = d["choices"][0]; txt = ch["text"]; fin = ch["finish_reason"]; toks = d["usage"]["completion_tokens"]
        pred = parse(txt, x["num_options"]) if fin == "stop" else None; stage2 = False
        if pred is None:
            body = txt if "</think>" not in txt else txt.split("</think>")[0]
            d2 = post("/v1/completions", {"model": a.model, "prompt": ctx + body.rstrip() + "\n</think>\n\nJawapan:", "max_tokens": 1, "temperature": 0, "logprobs": 20})
            lp = d2["choices"][0]["logprobs"]["top_logprobs"][0]; best = None
            for tok, v in lp.items():
                t = tok.strip()
                if t in list("ABCDE")[:x["num_options"]] and (best is None or v > best[1]): best = (t, v)
            pred = best[0] if best else None; stage2 = True
        return {"id": x["id"], "gold": x["key"], "pred": pred, "correct": int(pred == x["key"]), "parsed": int(not stage2), "stage2": stage2, "finish": fin, "tokens": toks, "subject_eng": x["subject_eng"], "category": x["category"], "nopt": x["num_options"]}
    except Exception as e:
        print("ERR", x["id"], repr(e)[:120], flush=True); return None
n = 0
with ThreadPoolExecutor(a.conc) as ex, open(a.out, "a") as f:
    for r in ex.map(one, todo):
        if r: f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush(); n += 1
        if n % 100 == 0 and n: print("done", n, flush=True)
print("MALAYGEN_DONE"); summary(a.out)
