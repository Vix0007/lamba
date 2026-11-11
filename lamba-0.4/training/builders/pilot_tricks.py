#!/usr/bin/env python3
"""Inference-time-compute pilot on the locked MalayMMLU subset (hard 460 + guard 150) for a served model.
Methods, all paired against base = plain greedy pass 1 (original order):
  vote : 4 votes = pass1 + 3 option-shuffled passes, majority on option CONTENT (tie -> pass1's choice)
  argue: one pass, 'argue for/against each option, then pick'
  crit : 3 passes = answer -> critique -> final answer (may change)
usage: pilot_tricks.py run MODEL OUT [--limit N] [--conc 40] [--rp 1.05]   (resumable)
       pilot_tricks.py score OUT"""
import sys, json, re, os, random, hashlib, argparse, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
from math import comb
sys.path.insert(0, "/home/ubuntu/antidoom_data")
ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("model", nargs="?"); ap.add_argument("out", nargs="?"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conc", type=int, default=40); ap.add_argument("--rp", type=float, default=1.05); ap.add_argument("--port", type=int, default=8001); a = ap.parse_args()
L = "ABCDE"
S = json.load(open("/home/ubuntu/antidoom_data/malaymmlu_sub_ids.json"))
def load_sub():
    keep = set(S["hard"]) | set(S["guard"]); return [x for x in json.load(open("/home/ubuntu/datasets/MalayMMLU_0shot.json")) if x["id"] in keep]
def parse(txt, nopt):
    t = txt.split("</think>")[-1] if "</think>" in txt else txt
    m = re.findall(r"Jawapan\s*[:：]\s*\**\s*\(?([A-E])\b", t) or re.findall(r"\\boxed\{\s*\(?([A-E])\)?\s*\}", t)
    return "ABCDE".index(m[-1]) if m and "ABCDE".index(m[-1]) < nopt else None

def score():
    R = collections.defaultdict(dict)
    for l in open(a.cmd == "score" and a.model or a.out):
        r = json.loads(l); R[r["stage"]][r["id"]] = r
    items = {x["id"]: x for x in load_sub()}
    def gold(i): return L.index(items[i]["key"])
    sh = ["sh1", "sh2", "sh3"]
    methods = {}
    for i in R["s1"]:
        if i not in items or not all(i in R[k] for k in sh + ["argue", "c3"]): continue
        votes = [R["s1"][i]["pred"]] + [R[k][i]["pred"] for k in sh]
        c = collections.Counter(v for v in votes if v is not None)
        if c:
            top = max(c.values()); tied = [v for v, n in c.items() if n == top]; vp = R["s1"][i]["pred"] if R["s1"][i]["pred"] in tied else min(tied)
        else: vp = None
        methods.setdefault("base", {})[i] = R["s1"][i]["pred"]
        methods.setdefault("vote4", {})[i] = vp
        methods.setdefault("argue", {})[i] = R["argue"][i]["pred"]
        methods.setdefault("crit3", {})[i] = R["c3"][i]["pred"]
        methods.setdefault("shuf1_only", {})[i] = R["sh1"][i]["pred"]
    ids = list(methods.get("base", {}))
    print(f"items complete for all stages: {len(ids)} / {len(items)}")
    for grp in ("hard", "guard", "all"):
        g = [i for i in ids if grp == "all" or i in set(S[grp])]
        if not g: continue
        print(f"\n== {grp} n={len(g)}")
        B = methods["base"]
        for m, P in methods.items():
            ok = sum(P[i] == gold(i) for i in g)
            b = sum(1 for i in g if P[i] == gold(i) and B[i] != gold(i)); c = sum(1 for i in g if B[i] == gold(i) and P[i] != gold(i)); nn = b + c
            p = min(1.0, 2 * sum(comb(nn, k) for k in range(0, min(b, c) + 1)) / 2 ** nn) if nn else 1.0
            miss = sum(P[i] is None for i in g)
            print(f"  {m:11s} acc {100*ok/len(g):6.2f}%  gained {b:3d} lost {c:3d} (vs base) McNemar p={p:.3f}  unparsed {miss}")
    toks = collections.defaultdict(int)
    for st, d in R.items():
        for r in d.values(): toks[st] += r.get("tokens", 0)
    n = max(1, len(R["s1"])); print("\nmean tokens/item per stage:", {k: round(v / n) for k, v in toks.items()})
if a.cmd == "score": a.model = a.model; score(); sys.exit(0)

import v7_augment as AU
AU.MN()
done = {}
if os.path.exists(a.out):
    for l in open(a.out): r = json.loads(l); done[(r["stage"], r["id"])] = r
import threading; lock = threading.Lock(); fout = open(a.out, "a")
def post(path, body):
    q = urllib.request.Request(f"http://127.0.0.1:{a.port}" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(q, timeout=3600).read())
def chat(msgs, nopt, max_tokens):
    ctx = AU.MN().render_context(msgs, tools=False)
    d = post("/v1/completions", {"model": a.model, "prompt": ctx, "max_tokens": max_tokens, "temperature": 0.0, "repetition_penalty": a.rp})
    ch = d["choices"][0]; txt = ch["text"]; fin = ch["finish_reason"]; toks = d["usage"]["completion_tokens"]
    pred = parse(txt, nopt) if fin == "stop" else None; forced = False
    if pred is None:
        body = txt.split("</think>")[0]
        d2 = post("/v1/completions", {"model": a.model, "prompt": ctx + body.rstrip() + "\n</think>\n\nJawapan:", "max_tokens": 1, "temperature": 0, "logprobs": 20})
        best = None
        for tok, v in d2["choices"][0]["logprobs"]["top_logprobs"][0].items():
            t = tok.strip()
            if t in list(L)[:nopt] and (best is None or v > best[1]): best = (t, v)
        pred = L.index(best[0]) if best else None; forced = True
    content = txt.split("</think>")[-1].strip() if "</think>" in txt else ""
    return {"pred": pred, "text": txt, "content": content, "forced": forced, "finish": fin, "tokens": toks}
def emit(stage, i, r):
    r = dict(r, stage=stage, id=i)
    with lock: fout.write(json.dumps(r, ensure_ascii=False) + "\n"); fout.flush()
    done[(stage, i)] = r; return r
def stage(name, i, fn):
    return done[(name, i)] if (name, i) in done else emit(name, i, fn())
INS = "\n\nJawab dengan memilih satu huruf. Akhiri dengan baris 'Jawapan: X'."
def prompt_of(x, order):
    stem = re.split(r"\nA\.\s", x["prompt"])[0]; opts = [re.sub(r"^[A-E]\.\s*", "", o) for o in x["options"]]
    return stem + "".join(f"\n{L[j]}. {opts[k]}" for j, k in enumerate(order))
def one(x):
    try:
        i = x["id"]; n = x["num_options"]; head = f"Berikut ialah soalan aneka pilihan tentang {x['subject']}.\n\n"
        norm = lambda t: [ln.strip() for ln in t.split("\n")]
        assert norm(prompt_of(x, list(range(n)))) == norm(x["prompt"]), i
        noshuf = any(re.search(r"(semua|kesemua|tiada)[^\n]{0,20}(di atas|atas|betul|salah)|\b[A-E]\b\s*(dan|atau|,|&)\s*\b[A-E]\b|\b[A-E] sahaja\b", o, re.I) for o in x["options"])
        q = head + x["prompt"] + INS
        s1 = stage("s1", i, lambda: chat([{"role": "user", "content": q}], n, 4000))
        for k in ((1, 2, 3) if not os.environ.get("PILOT_ITEMS") else ()):
            def f(k=k):
                rng = random.Random(int(hashlib.md5(f"{i}|{k}".encode()).hexdigest()[:8], 16)); order = list(range(n))
                for _ in range(20):
                    rng.shuffle(order)
                    if order != list(range(n)): break
                r = chat([{"role": "user", "content": head + prompt_of(x, order) + INS}], n, 4000)
                r["pred"] = order[r["pred"]] if r["pred"] is not None else None; r["order"] = order; return r
            if noshuf: stage(f"sh{k}", i, lambda: dict(s1, order=list(range(n)), noshuf=True))
            else: stage(f"sh{k}", i, f)
        if not os.environ.get("PILOT_ITEMS"): stage("argue", i, lambda: chat([{"role": "user", "content": head + x["prompt"] + "\n\nUntuk setiap pilihan, bina hujah terkuat mengapa ia benar dan mengapa ia mungkin salah, kemudian bandingkan dan pilih satu jawapan terbaik. Akhiri dengan baris 'Jawapan: X'."}], n, 4000))
        a1 = s1["content"] or f"Jawapan: {L[s1['pred']] if s1['pred'] is not None else '?'}"
        U2 = "Semak jawapan anda di atas dengan teliti seperti seorang penyemak yang ketat. Adakah ia betul? Uji ia terhadap setiap pilihan lain dan kenal pasti sebarang kesilapan fakta atau penaakulan. Jangan beri jawapan akhir lagi; hanya beri semakan anda."
        c2 = stage("c2", i, lambda: chat([{"role": "user", "content": q}, {"role": "assistant", "content": a1}, {"role": "user", "content": U2}], n, 3000))
        cr = c2["content"] or c2["text"].split("</think>")[0][-1500:]
        U3 = "Berdasarkan semakan anda, beri jawapan akhir. Jika jawapan asal salah, ubahkannya. Akhiri dengan baris 'Jawapan: X'."
        stage("c3", i, lambda: chat([{"role": "user", "content": q}, {"role": "assistant", "content": a1}, {"role": "user", "content": U2}, {"role": "assistant", "content": cr}, {"role": "user", "content": U3}], n, 3000))
        return 1
    except Exception as e:
        print("ERR", x["id"], repr(e)[:150], flush=True); return 0
items = load_sub()
if os.environ.get("PILOT_ITEMS"):  # sample file of ids (full 5k mode): run only s1,c2,c3 on these
    ids = set(json.load(open(os.environ["PILOT_ITEMS"]))); items = [x for x in json.load(open("/home/ubuntu/datasets/MalayMMLU_0shot.json")) if x["id"] in ids]
if a.limit: items = items[:a.limit]
nd = 0
with ThreadPoolExecutor(a.conc) as ex:
    for r in ex.map(one, items):
        nd += r
        if nd % 25 == 0: print("items done", nd, flush=True)
print("PILOT_DONE", nd)
