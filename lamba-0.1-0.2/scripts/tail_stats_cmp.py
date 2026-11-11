"""usage: tail_stats_cmp.py <file_a.jsonl (reference, e.g. v8)> <file_b.jsonl (candidate)> <prompts.jsonl>
Paired comparison on shared ids. Per output: 60-char windows (anchors every 30 chars, >=25 alnum) with >=4 occurrences, excluding windows that occur in the prompt;
R_any = has repeated chars inside the last 2500 chars; R_heavy = >1500 repeated chars inside the last 2500 chars; L = Liquid-style loop (any such window anywhere); B = finish_reason=length; health: mean tokens, empty(<20 tok), newline/1k.
Prints rates (%) for A and B, difference in pp (B - A), discordant counts and two-sided McNemar p."""
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
n = len(ids); print(f"paired prompts: {n}")
def row(name, fa_, fb_):
    a = sum(fa_(i) for i in ids); b = sum(fb_(i) for i in ids); x = sum(fa_(i) and not fb_(i) for i in ids); y = sum(fb_(i) and not fa_(i) for i in ids)
    print(f"{name:8s} A {a:4d} ({100*a/n:5.1f}%)  B {b:4d} ({100*b/n:5.1f}%)  diff {100*(b-a)/n:+5.1f}pp  discordant A-only {x} B-only {y}  p={mc(x, y):.3g}")
row("R_any", lambda i: SA[i]["any"], lambda i: SB[i]["any"])
row("R_heavy", lambda i: SA[i]["heavy"], lambda i: SB[i]["heavy"])
row("L", lambda i: SA[i]["L"], lambda i: SB[i]["L"])
row("B", lambda i: A[i]["finish_reason"] == "length", lambda i: B[i]["finish_reason"] == "length")
import statistics as st
f = lambda D, s: st.mean(D[i]["text"].count(s) / max(1, len(D[i]["text"])) * 1000 for i in ids)
print(f"health: mean tokens A {st.mean(A[i]['n_tokens'] for i in ids):.0f} B {st.mean(B[i]['n_tokens'] for i in ids):.0f} | empty A {sum(A[i]['n_tokens']<20 for i in ids)} B {sum(B[i]['n_tokens']<20 for i in ids)} | newline/1k A {f(A,chr(10)*2):.2f}/{f(A,chr(10)):.2f} B {f(B,chr(10)*2):.2f}/{f(B,chr(10)):.2f}")
