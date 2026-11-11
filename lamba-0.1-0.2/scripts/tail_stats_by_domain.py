"""Per-domain breakdown: same metrics as tail_stats_cmp.py, split by prompts.jsonl 'domain' field.
usage: tail_stats_by_domain.py <file_a.jsonl> <file_b.jsonl> <prompts.jsonl>
"""
import json, math, re, sys
fa, fb, fp = sys.argv[1:4]
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
A, B = load(fa), load(fb); P = load(fp); ids = sorted(set(A) & set(B) & set(P)); al = re.compile(r"[A-Za-z0-9]")
import rp_guard; rp_guard.check(fa, list(A.values()), fb, list(B.values()))
def stats(t, prompt):
    n = len(t); cov = bytearray(n); seen = set(); found = False
    for i in range(0, n - 60, 30):
        w = t[i:i + 60]
        if w in seen or len(al.findall(w)) < 25: continue
        seen.add(w)
        if t.count(w) < 4 or w in prompt: continue
        found = True; j = t.find(w)
        while j >= 0:
            for k in range(j, min(n, j + 60)): cov[k] = 1
            j = t.find(w, j + 60)
    tail = sum(cov[max(0, n - 2500):])
    return {"L": found, "any": tail > 0, "heavy": tail > 1500}
def mc(x, y):
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(min(x, y) + 1)) / 2 ** n)
SA = {i: stats(A[i]["text"], P[i]["prompt"]) for i in ids}; SB = {i: stats(B[i]["text"], P[i]["prompt"]) for i in ids}
domains = sorted(set(P[i]["domain"] for i in ids))
for dom in domains + ["ALL"]:
    dids = ids if dom == "ALL" else [i for i in ids if P[i]["domain"] == dom]
    n = len(dids)
    print(f"\n=== {dom} (n={n}) ===")
    def row(name, fa_, fb_):
        a = sum(fa_(i) for i in dids); b = sum(fb_(i) for i in dids); x = sum(fa_(i) and not fb_(i) for i in dids); y = sum(fb_(i) and not fa_(i) for i in dids)
        print(f"{name:8s} A {a:4d} ({100*a/n:5.1f}%)  B {b:4d} ({100*b/n:5.1f}%)  diff {100*(b-a)/n:+5.1f}pp  discordant A-only {x} B-only {y}  p={mc(x, y):.3g}")
    row("R_any", lambda i: SA[i]["any"], lambda i: SB[i]["any"])
    row("R_heavy", lambda i: SA[i]["heavy"], lambda i: SB[i]["heavy"])
    row("L", lambda i: SA[i]["L"], lambda i: SB[i]["L"])
    row("B", lambda i: A[i]["finish_reason"] == "length", lambda i: B[i]["finish_reason"] == "length")
    import statistics as st
    f = lambda D, s: st.mean(D[i]["text"].count(s) / max(1, len(D[i]["text"])) * 1000 for i in dids)
    print(f"health: mean tokens A {st.mean(A[i]['n_tokens'] for i in dids):.0f} B {st.mean(B[i]['n_tokens'] for i in dids):.0f} | empty A {sum(A[i]['n_tokens']<20 for i in dids)} B {sum(B[i]['n_tokens']<20 for i in dids)}")
