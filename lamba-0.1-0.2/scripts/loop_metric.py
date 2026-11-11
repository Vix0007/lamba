"""Loop metric (DS 63, simplified): for each saved completion, token-level: loop_score = fraction of token positions covered by 8-grams that occur >=3 times in the text;
flag loop if loop_score > 0.25; also contiguous-periodic tail flag (period>=15 chars, >=2 periods reaching the end) and distinct-4.
usage: loop_metric.py <file_a.jsonl> [<file_b.jsonl> ...] [--ids first75|gate24]   (jsonl rows: id,text,finish_reason). Prints a table; paired comparison when 2 files (a=base, b=adapter) on shared ids."""
import json, re, sys, collections, math
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
def tail_loop(text, pmin=15):
    n = len(text)
    if n < 300: return False
    tail = text[-6000:]; m = len(tail)
    for p in range(pmin, min(1200, m // 2) + 1):
        if tail[m - p:] != tail[m - 2 * p:m - p]: continue
        i = m - p - 1
        while i >= 0 and tail[i] == tail[i + p]: i -= 1
        s = i + 1
        if m - s >= 2 * p - 5:
            u = tail[s:s + p]
            if len(u.strip()) >= 10 and not re.fullmatch(r"[\W\d_]+", u): return True
    return False
_al = re.compile(r"[A-Za-z0-9]")
def liquid_loop(t, W=60, S=30):
    for i in range(0, len(t) - W, S):
        w = t[i:i + W]
        if len(_al.findall(w)) < 25: continue
        if t.count(w) >= 4: return True
    return False
def score(text, n=8, k=3):
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) < n + 5: return 0.0, 1.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    c = collections.Counter(grams); cov = bytearray(len(ids))
    for i, g in enumerate(grams):
        if c[g] >= k:
            for j in range(i, i + n): cov[j] = 1
    g4 = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return sum(cov) / len(ids), len(set(g4)) / max(1, len(g4))
def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l); d[r["id"]] = r
    return d
files = [a for a in sys.argv[1:] if not a.startswith("--")]
data = [load(f) for f in files]
ids = set.intersection(*[set(d) for d in data])
res = []
for f, d in zip(files, data):
    r = {}
    for i in ids:
        t = d[i]["text"]; ls, d4 = score(t); r[i] = (ls > 0.25, tail_loop(t), ls, d4, d[i]["finish_reason"] == "length", liquid_loop(t))
    res.append(r)
    n = len(ids)
    print(f"{f.split('/')[-1]:38s} n={n} loop_score>0.25: {sum(v[0] for v in r.values())}  contiguous-tail loops: {sum(v[1] for v in r.values())}  budget-hit: {sum(v[4] for v in r.values())}  liquid-style loops (>=4x60ch): {sum(v[5] for v in r.values())}  mean loop_score {sum(v[2] for v in r.values())/n:.3f}  mean distinct-4 {sum(v[3] for v in r.values())/n:.3f}")
if len(res) == 2:
    for name, idx in (("loop_score>0.25", 0), ("contiguous-tail", 1), ("liquid-style", 5)):
        b = sum(res[0][i][idx] and not res[1][i][idx] for i in ids); c = sum(res[1][i][idx] and not res[0][i][idx] for i in ids)
        nn = b + c; p = min(1.0, 2 * sum(math.comb(nn, j) for j in range(min(b, c) + 1)) / 2 ** nn) if nn else 1.0
        print(f"paired {name}: base-only {b} adapter-only {c}  McNemar two-sided p={p:.3g}")
