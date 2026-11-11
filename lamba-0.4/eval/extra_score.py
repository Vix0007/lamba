#!/usr/bin/env python3
"""Scores for warmth_gen-format outputs.
  extra_score.py shortq FILE.jsonl   -> identity-reflex rate (answer says Lamba/Vixero although the question asks what X is)
  extra_score.py msfact FILE.jsonl   -> crude gold-match accuracy + misreading count (flower / first lady)"""
import json, re, sys, math
def ans(s): return s.split("</think>", 1)[1].strip() if "</think>" in s else ""
def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)
def norm(x): return re.sub(r"[^a-z0-9 ]", " ", x.lower())
def gold_ok(a, g):
    a, g = norm(a), norm(g)
    if not a.strip(): return None
    if re.search(r"\b" + re.escape(g.strip()) + r"\b", a): return True
    t = [x for x in g.split() if len(x) >= 4 or x.isdigit()]
    return bool(t) and all(re.search(r"\b" + re.escape(x) + r"\b", a) for x in t[:2])
FLOWER = re.compile(r"\bbunga\b|first lady|isteri (perdana|yang di-pertuan)|permaisuri|wanita pertama", re.I)
mode, fn = sys.argv[1], sys.argv[2]
R = [json.loads(l) for l in open(fn)]
n = k = flower = empty = 0
for r in R:
    for s in r.get("samples", []):
        a = ans(s); n += 1
        if not a: empty += 1
        if mode == "shortq":
            k += bool(re.search(r"\b(lamba|vixero)\b", a, re.I))
        else:
            ok = gold_ok(a, r["answer"]); k += bool(ok)
            flower += bool(FLOWER.search(a)) and not re.search(r"bunga", r["prompt"], re.I)
lab = "reflex" if mode == "shortq" else "correct"
print(f"{fn}: {lab} {k}/{n} = {100*k/max(1,n):.1f}% CI{wilson(k,n)} empty {empty}" + (f" misread(flower/first-lady) {flower}" if mode == "msfact" else ""))
