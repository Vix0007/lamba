"""DS 72 diagnostic: where does v8 change repetition? base vs v8 on g150 (paired). Per output: repeated-char coverage (60-char windows with >=4 occurrences, stride 30) inside three regions [0,1500), [1500,len-2500), [len-2500,len];
share of qualifying windows that also occur in the PROMPT; earliest start offset of a qualifying window; run-to-end contiguous loop reaching the last 500 chars; zlib ratio first 1500 / last 2500. Stratified by domain and templated flag (>=50% 8-word overlap with training prompts ~ mcp)."""
import json, re, zlib, math, collections, statistics as st
R = "/home/ubuntu/eval_results"; D = "/home/ubuntu/antidoom_data"
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(f"{D}/evalpool_clean.jsonl")}
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
B, A = load(f"{R}/loop_base_ev586.jsonl"), load(f"{R}/loop_v8final_ev586.jsonl")
ids = sorted(set(A) & set(B)); al = re.compile(r"[A-Za-z0-9]")
def cr(t): 
    x = t.encode(); return 1.0 if len(x) < 100 else len(zlib.compress(x, 6)) / len(x)
def tail_run(t):
    n = len(t)
    if n < 300: return None
    tail = t[-6000:]; m = len(tail)
    for p in range(15, min(1200, m // 2) + 1):
        if tail[m - p:] != tail[m - 2 * p:m - p]: continue
        i = m - p - 1
        while i >= 0 and tail[i] == tail[i + p]: i -= 1
        s = i + 1
        if m - s >= 2 * p - 5 and len(tail[s:s + p].strip()) >= 10 and not re.fullmatch(r"[\W\d_]+", tail[s:s + p]): return (n - m + s, p)
    return None
def stats(t, prompt, minocc=4):
    n = len(t); cov = bytearray(n); first = None; inprompt = tot = 0; seen = set()
    for i in range(0, n - 60, 30):
        w = t[i:i + 60]
        if len(al.findall(w)) < 25 or w in seen: continue
        c = t.count(w)
        if c < minocc: continue
        seen.add(w); tot += 1; inprompt += (prompt.count(w) > 0)
        j = t.find(w)
        while j >= 0:
            for k in range(j, min(n, j + 60)): cov[k] = 1
            first = j if first is None else min(first, j)
            j = t.find(w, j + 60)
    a = sum(cov[:1500]); b = sum(cov[1500:max(1500, n - 2500)]); c = sum(cov[max(1500, n - 2500):])
    tr = tail_run(t)
    return {"early": a, "mid": b, "tail": c, "first": first, "inprompt": inprompt, "nwin": tot, "run_end": (tr is not None and tr[0] + 2 * tr[1] >= n - 500), "run_start": None if tr is None else tr[0],
            "cr_first": cr(t[:1500]), "cr_last": cr(t[-2500:]), "n": n}
rows = {}
for i in ids:
    p = prompts[i]["prompt"]
    rows[i] = (stats(B[i]["text"], p), stats(A[i]["text"], p), prompts[i]["domain"])
def mc(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n)
def rep(name, sel):
    S = [rows[i] for i in ids if sel(rows[i][2])]
    if not S: return
    f = lambda k, j: st.mean(s[j][k] for s in S)
    def flag(pred):
        b = sum(pred(s[0]) for s in S); a = sum(pred(s[1]) for s in S); x = sum(pred(s[0]) and not pred(s[1]) for s in S); y = sum(pred(s[1]) and not pred(s[0]) for s in S)
        return f"{b}->{a} (base-only {x}/v8-only {y}, p={mc(x, y):.2g})"
    print(f"--- {name} (n={len(S)})")
    print(f"  repeated chars per output: early[0,1500) base {f('early',0):.0f} v8 {f('early',1):.0f} | mid base {f('mid',0):.0f} v8 {f('mid',1):.0f} | tail(last 2500) base {f('tail',0):.0f} v8 {f('tail',1):.0f}")
    print(f"  qualifying windows: base {f('nwin',0):.1f} v8 {f('nwin',1):.1f}; share of windows also in PROMPT: base {sum(s[0]['inprompt'] for s in S)/max(1,sum(s[0]['nwin'] for s in S)):.2f} v8 {sum(s[1]['inprompt'] for s in S)/max(1,sum(s[1]['nwin'] for s in S)):.2f}")
    print("  outputs with >=1 repeated window whose EARLIEST start >1500:", flag(lambda s: s['first'] is not None and s['first'] > 1500))
    print("  outputs with repeated chars in the last 2500 (tail>0):", flag(lambda s: s['tail'] > 0))
    print("  outputs with tail repeated chars > 1500 (heavy tail loop):", flag(lambda s: s['tail'] > 1500))
    print("  outputs with run-to-end contiguous loop reaching last 500 chars:", flag(lambda s: s['run_end']))
    print(f"  zlib first1500 base {f('cr_first',0):.3f} v8 {f('cr_first',1):.3f} | last2500 base {f('cr_last',0):.3f} v8 {f('cr_last',1):.3f}")
rep("ALL", lambda d: True)
for d in ("mcp", "science", "malay"): rep(d, lambda x, d=d: x == d)
