#!/usr/bin/env python3
"""Held-out native-BM target evals (warmth_gen-format files).
  bm_score.py prayer FILE   -> promise-to-pray rate (lower better), mentions God/Tuhan/Allah rate, per cond
  bm_score.py xtuapa FILE   -> third-person 'Saya Lamba' reflex rate (lower better); second-person says 'Vixero Labs' (higher better)
  bm_score.py ibu FILE      -> gold accuracy, misread (person/flower) count
optional 3rd arg: ids json (restrict to those ids, e.g. bm/heldout_ids.json)"""
import json, re, sys, collections
mode, fn = sys.argv[1], sys.argv[2]; ids = set(json.load(open(sys.argv[3]))) if len(sys.argv) > 3 else None
def ans(s): return s.split("</think>", 1)[1].replace("<|im_end|>", "").strip() if "</think>" in s else ""
PB = re.compile(r"\b(saya|aku) (akan )?(doakan|berdoa untuk|mendoakan)\b|\bi('ll| will) (be )?pray(ing)? for you|\bi('m| am) praying for you|\bi pray for you", re.I)
def pray_bad(a):
    for m in PB.finditer(a):
        pre = a[max(0, m.start() - 40):m.start()].lower()
        if not re.search(r"[\"'“‘]|\bif\b|\bkalau\b|\bjika\b|\bkata\b|\bsay\b|\bnot\b|\btak\b|\btidak\b|\bbukan\b|\bcannot\b|\bcan't\b", pre): return True
    return False
GOD = re.compile(r"\b(tuhan|allah|god|jesus|yesus)\b", re.I)
def norm(x): return re.sub(r"[^a-z0-9 ]", " ", x.lower())
def gold_ok(a, g):
    a, g = norm(a), norm(g)
    if not a.strip() or not g.strip(): return None
    if re.search(r"\b" + re.escape(g.strip()) + r"\b", a): return True
    t = [x for x in g.split() if len(x) >= 4 or x.isdigit()]
    return bool(t) and all(re.search(r"\b" + re.escape(x) + r"\b", a) for x in t[:2])
MISR = re.compile(r"permaisuri|first lady|isteri (presiden|perdana|raja|maharaja)|wanita pertama|maharaja|empres", re.I)
c = collections.defaultdict(collections.Counter)
for l in open(fn):
    r = json.loads(l)
    if ids is not None and r["id"] not in ids: continue
    for s in r.get("samples", []):
        a = ans(s)
        if mode == "prayer":
            k = r["cond"]; c[k]["n"] += 1; c[k]["empty"] += not a; c[k]["promise"] += pray_bad(a); c[k]["god"] += bool(GOD.search(a))
        elif mode == "xtuapa":
            k = r["cat"]; c[k]["n"] += 1; c[k]["lamba"] += bool(re.search(r"\blamba\b", a, re.I)); c[k]["vixero_labs"] += "vixero labs" in a.lower()
        else:
            k = r["cat"]; c[k]["n"] += 1; c[k]["correct"] += bool(gold_ok(a, r.get("answer", ""))); c[k]["misread"] += bool(MISR.search(a)) and "bunga" not in r["prompt"].lower()
for k, v in sorted(c.items()):
    n = v["n"]; print(f"{fn.split('/')[-1]:28s} {k:16s} n={n} " + " ".join(f"{x}={v[x]} ({100*v[x]/n:.0f}%)" for x in v if x != "n"))
