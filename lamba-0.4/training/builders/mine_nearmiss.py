#!/usr/bin/env python3
"""Near-miss (mixed-band) mining for Malay MCQ, on the served model, in the EXACT generative MalayMMLU eval format
(no system prompt, 'Berikut ialah soalan aneka pilihan tentang <subject>.' + item + 'Jawab dengan memilih satu huruf...').
Pools are our own decontaminated items (never MalayMMLU benchmark items): malay_slice/band_queue_dpov2.jsonl + wk_mcq/wk_items_verified.jsonl.
K samples at T=1.0; an item is near-miss when some samples are right and some wrong. Resumable.
usage: mine_nearmiss.py MODEL OUT.jsonl [--n-band 1200] [--n-wk 1000] [--k 4] [--conc 48]"""
import json, re, os, sys, random, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--n-band", type=int, default=1200); ap.add_argument("--n-wk", type=int, default=1000)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--conc", type=int, default=48); a = ap.parse_args()
D = "/home/ubuntu/antidoom_data/"
INS = "\n\nJawab dengan memilih satu huruf. Akhiri dengan baris 'Jawapan: X'."
items = []
for l in list(open(D + "malay_slice/band_queue_dpov2.jsonl"))[:a.n_band]:
    r = json.loads(l)
    if r.get("num_options", 4) < 3: continue
    items.append({"id": r["pilot_id"], "subject": r.get("subject", "Umum"), "prompt": r["prompt"], "gold": r["ground_truth_answer"], "src": "band"})
wk = [json.loads(l) for l in open(D + "wk_mcq/wk_items_verified.jsonl")]; random.Random(7).shuffle(wk)
for r in wk[:a.n_wk]:
    items.append({"id": r["id"], "subject": r.get("category", "Umum").title(), "prompt": r["prompt"], "gold": r["gold"], "src": "wk"})
done = set()
if os.path.exists(a.out): done = {json.loads(l)["id"] for l in open(a.out)}
todo = [x for x in items if x["id"] not in done]
print(len(items), "items", len(todo), "to go", flush=True)
def ctx(x):
    u = f"Berikut ialah soalan aneka pilihan tentang {x['subject']}.\n\n{x['prompt']}{INS}"
    return [{"role": "user", "content": u}], f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n"
def parse(t):
    m = re.findall(r"Jawapan\s*:\s*\**\s*\(?([A-E])\b", t.split("</think>")[-1])
    return m[-1] if m else None
def one(x):
    msgs, rc = ctx(x)
    body = {"model": a.model, "prompt": rc, "max_tokens": 2500, "temperature": 1.0, "top_p": 0.95, "n": a.k, "skip_special_tokens": False}
    for t in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/v1/completions", json.dumps(body).encode(),
                                                                        {"Content-Type": "application/json"}), timeout=1800))
            S = [{"text": c["text"], "finish": c.get("finish_reason"), "pred": parse(c["text"]) if c.get("finish_reason") == "stop" else None} for c in d["choices"]]
            return {**x, "context_messages": msgs, "rendered_context": rc, "samples": S}
        except Exception as e:
            err = str(e)
    return None
with ThreadPoolExecutor(a.conc) as ex, open(a.out, "a") as f:
    for i, r in enumerate(ex.map(one, todo)):
        if r: f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        if i % 200 == 0: print("done", i, flush=True)
print("NEARMISS_MINE_DONE", flush=True)
