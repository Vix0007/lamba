"""Analyse the BASE-model position-vs-content experiment (~/eval_results/mech_base_malay5k.jsonl).
Each row: id, variant (asis|inline|shiftK), order (presented option i = original option order[i]), gold_orig, n, lp{A..E} letter logprobs.
Reports (all on the base model):
  1. accuracy by PRESENTED gold position, per variant            (position bias, reproduces the 24k finding on 5k)
  2. inline vs asis accuracy gap                                   (tests the handoff's 'inline formatting' theory)
  3. content-consistency across cyclic shifts: fraction of items where the same option CONTENT wins in every shift
  4. permutation-averaged accuracy (average content logprob over all shifts)  = inference-time debiasing bound (PriDe-style)
  5. PriDe-style prior: estimate per-letter prior from permutations, debias single-shift predictions
  6. PBM-like statistic: mean variance across shifts of the (softmax over letters) probability assigned to the gold CONTENT
"""
import collections, json, math, sys
import numpy as np
P = "/home/ubuntu/eval_results/mech_base_malay5k.jsonl"
rows = [json.loads(l) for l in open(P)]
by = collections.defaultdict(dict)
for r in rows: by[r["id"]][r["variant"]] = r
def probs(lp, n):
    v = np.array([lp[chr(65 + i)] for i in range(n)]); v = v - v.max(); e = np.exp(v); return e / e.sum()
print(f"items: {len(by)}  rows: {len(rows)}")
# 1. position bias per variant
for var in sorted({r["variant"] for r in rows}):
    acc = collections.defaultdict(lambda: [0, 0]); tot = [0, 0]
    for it in by.values():
        r = it.get(var)
        if not r: continue
        n = r["n"]; order = r["order"]; gpos = order.index(r["gold_orig"]); p = probs(r["lp"], n); ok = int(p.argmax() == gpos)
        acc[chr(65 + gpos)][0] += 1; acc[chr(65 + gpos)][1] += ok; tot[0] += 1; tot[1] += ok
    print(f"{var:8s} acc {100*tot[1]/tot[0]:.2f}% (n={tot[0]}) | by presented gold pos:",
          {k: f"{100*v[1]/v[0]:.1f}%" for k, v in sorted(acc.items()) if k in 'ABCD'})
# 2. inline vs asis
pair = [(it["asis"], it["inline"]) for it in by.values() if "asis" in it and "inline" in it]
def a(r): n = r["n"]; return int(probs(r["lp"], n).argmax() == r["order"].index(r["gold_orig"]))
if pair:
    x = np.array([a(p) for p, _ in pair]); y = np.array([a(q) for _, q in pair])
    print(f"asis acc {100*x.mean():.2f}% vs inline acc {100*y.mean():.2f}% (gap {100*(y.mean()-x.mean()):+.2f}pt, n={len(pair)}; McNemar discordant asis-only {int(((x==1)&(y==0)).sum())} inline-only {int(((x==0)&(y==1)).sum())})")
# 3-6 across cyclic shifts (variants asis=shift0, shift1..)
cons, avg_ok, avg_asis, pbm, prior_num = 0, 0, 0, [], collections.defaultdict(list)
full = []
for it in by.values():
    vs = [v for v in (["asis"] + [f"shift{k}" for k in range(1, 5)]) if v in it]
    n = it[vs[0]]["n"]
    if len(vs) != n: continue
    content_lp = np.zeros(n); winners = []; pg = []
    for v in vs:
        r = it[v]; p = probs(r["lp"], n)
        for pos in range(n): content_lp[r["order"][pos]] += np.log(p[pos] + 1e-12)
        winners.append(r["order"][int(p.argmax())]); pg.append(p[r["order"].index(r["gold_orig"])])
        for pos in range(n): prior_num[(n, pos)].append(np.log(p[pos] + 1e-12))
    full.append((n, it, vs, content_lp, winners, pg))
    cons += int(len(set(winners)) == 1); avg_ok += int(content_lp.argmax() == it[vs[0]]["gold_orig"])
    avg_asis += int(winners[0] == it[vs[0]]["gold_orig"]); pbm.append(float(np.var(pg)))
m = len(full)
print(f"[{m} items with all shifts] single-order acc {100*avg_asis/m:.2f}% | content-consistent across shifts {100*cons/m:.1f}% | "
      f"permutation-averaged acc {100*avg_ok/m:.2f}% (inference-time debias bound, +{100*(avg_ok-avg_asis)/m:.2f}pt) | mean var of P(gold content) across shifts {np.mean(pbm):.4f}")
# PriDe-style: subtract mean per-position log prior estimated from all items, single-order prediction
pri = {k: np.mean(v) for k, v in prior_num.items()}; ok = 0
for n, it, vs, _, _, _ in full:
    r = it["asis"]; p = np.log(probs(r["lp"], n) + 1e-12) - np.array([pri[(n, pos)] for pos in range(n)])
    ok += int(r["order"][int(p.argmax())] == r["gold_orig"])
print(f"PriDe-style prior-subtracted single-order acc {100*ok/m:.2f}% (prior est. per (n_opts,position) from permutations)")

# ---- metrics as defined in the papers (PriDe RStd; LoRA-1 CKLD/PBM; PA-GRPO Con/CA; PAPEFT fluctuation) ----
print("\n--- paper-defined metrics (base model) ---")
def gold_letter_recall(variant):
    rec = collections.defaultdict(lambda: [0, 0])
    for it in by.values():
        r = it.get(variant)
        if not r: continue
        n = r["n"]; gpos = r["order"].index(r["gold_orig"]); ok = int(probs(r["lp"], n).argmax() == gpos)
        rec[gpos][0] += 1; rec[gpos][1] += ok
    return {p: v[1] / v[0] for p, v in rec.items() if v[0]}, rec
rec, raw = gold_letter_recall("asis")
vals = np.array([rec[p] for p in sorted(rec) if p < 4]) * 100
print(f"RStd (std of per-gold-letter recall, asis order, letters A-D): {vals.std():.2f}  | recalls {[round(x,1) for x in vals]}")
gold_d = np.array([raw[p][0] for p in range(4)], float); gold_d /= gold_d.sum()
pred_c = collections.Counter(); 
for it in by.values():
    r = it.get("asis")
    if r: pred_c[int(probs(r["lp"], r["n"]).argmax())] += 1
pd_ = np.array([pred_c[p] for p in range(4)], float) + 1e-9; pd_ /= pd_.sum()
print(f"CKLD = KL(pred label dist || gold label dist) = {float(np.sum(pd_*np.log(pd_/(gold_d+1e-9)))):.4f}  | pred {np.round(pd_,3).tolist()} gold {np.round(gold_d,3).tolist()}")
con_frac, ca = [], 0
for n, it, vs, content_lp, winners, pg in full:
    mode, cnt = collections.Counter(winners).most_common(1)[0]
    con_frac.append(cnt / len(winners)); ca += int(cnt / len(winners) == 1.0 and False) or int(mode == it[vs[0]]["gold_orig"])
print(f"PA-GRPO Con (mean n_mode/n over the cyclic shifts): {100*np.mean(con_frac):.1f}% | fully-consistent items {100*np.mean([c==1.0 for c in con_frac]):.1f}% | CA (mode == gold): {100*ca/len(full):.2f}%")
posacc = collections.defaultdict(lambda: [0, 0])
for n, it, vs, _, _, _ in full:
    for v in vs:
        r = it[v]; gpos = r["order"].index(r["gold_orig"]); posacc[gpos][0] += 1; posacc[gpos][1] += int(probs(r["lp"], n).argmax() == gpos)
A_p = np.array([posacc[p][1] / posacc[p][0] for p in sorted(posacc) if p < 4])
print(f"PAPEFT fluctuation = std_p(A_p)/mean_p(A_p) over presented gold positions (all shifts pooled): {A_p.std()/A_p.mean():.4f} | per-position acc {[round(100*x,1) for x in A_p]}")
