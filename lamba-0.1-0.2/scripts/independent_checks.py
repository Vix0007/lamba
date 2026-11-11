"""Independent checks on the v8 result (DS 70): (1) detector-independent repetition metrics on base vs v8 g150 outputs: zlib compression ratio of the last 3000 chars / whole text (lower = more repetitive), sweep of thresholds, paired sign test;
(2) contamination: 8-word-shingle overlap of eval prompts (150 test + 586 pool) with the PROMPTS inside the training pairs' contexts (all_ftpo_pairs_antidoom8.jsonl)."""
import json, re, zlib, math, collections, sys
R = "/home/ubuntu/eval_results"; D = "/home/ubuntu/antidoom_data"
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
b, a = load(f"{R}/loop_base_g150.jsonl"), load(f"{R}/loop_v8final_g150.jsonl")
ids = sorted(set(b) & set(a))
def cr(t): 
    if len(t) < 200: return 1.0
    x = t.encode(); return len(zlib.compress(x, 6)) / len(x)
def mc(x, y):
    n = x + y
    if n == 0: return 1.0
    k = min(x, y); return min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n)
print("== (1) compression-based repetition (independent of the 60-char detector)")
for name, f in (("last3000ch", lambda t: cr(t[-3000:])), ("whole", cr)):
    vb = {i: f(b[i]["text"]) for i in ids}; va = {i: f(a[i]["text"]) for i in ids}
    print(f"{name}: mean ratio base {sum(vb.values())/len(ids):.3f} v8 {sum(va.values())/len(ids):.3f} (higher = less repetitive)")
    for th in (0.20, 0.25, 0.30, 0.35):
        fb = {i for i in ids if vb[i] < th}; fa = {i for i in ids if va[i] < th}
        x = len(fb - fa); y = len(fa - fb)
        print(f"  flag ratio<{th}: base {len(fb)} v8 {len(fa)} | base-only {x} v8-only {y} p={mc(x, y):.3g}")
print("== (2) contamination (8-word shingles)")
def sh(t, k=8):
    w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}
train = collections.defaultdict(int)
pairs = [json.loads(l) for l in open(f"{D}/all_ftpo_pairs_antidoom8.jsonl")]
tshingles = set()
for r in pairs:
    ctx = r["context_with_chat_template"]; u = ctx.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0]
    tshingles |= sh(u)
print("train prompt shingles", len(tshingles), "from", len(pairs), "pairs")
for name, path in (("test150", f"{D}/loop_testset150.jsonl"), ("pool586", f"{D}/evalpool_clean.jsonl")):
    rows = [json.loads(l) for l in open(path)]
    hi = 0; mx = []
    for r in rows:
        s = sh(r["prompt"]); 
        ov = len(s & tshingles) / max(1, len(s)); mx.append(ov); hi += ov >= 0.5
    mx.sort()
    print(f"{name}: n={len(rows)} prompts with >=50% shingle overlap with a training prompt: {hi}; overlap percentiles 50/90/99: {[round(mx[int(p*(len(mx)-1))],2) for p in (.5,.9,.99)]}")
