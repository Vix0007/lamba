#!/usr/bin/env python3
"""loop_compare_v2.py - PRE-REGISTERED AMENDED endpoint: PRIMARY = exact_loop OR budget_hit.

Usage: loop_compare_v2.py <tag_base> <tag_adapter> [--adapter-label r2] [--semantic]
Reads ~/eval_results/loop_base_<tag_base>.jsonl and loop_<adapter-label>_<tag_adapter>.jsonl
"""
import argparse, json, math, os, re, sys
sys.path.insert(0, "/home/ubuntu/antidoom/src")
import numpy as np
from antidoom.repetition import find_inner_repetition

RES = os.path.expanduser("~/eval_results")
BUDGET, NBOOT, SEED = 4000, 2000, 0


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line); out[r["id"]] = r
    return out


def make_sem_detector():
    """Existing BGE-M3 sentence near-duplicate detector (DESCRIPTIVE only; saturates)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4e9 else "cpu"
    tk = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    m = AutoModel.from_pretrained("BAAI/bge-m3",
                                  dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    def f(text, thr=0.88, gap=3):
        s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if len(x.strip()) > 15]
        if len(s) < gap + 2:
            return False
        with torch.no_grad():
            v = []
            for i in range(0, len(s), 64):
                e = tk(s[i:i + 64], padding=True, truncation=True, max_length=256,
                       return_tensors="pt").to(dev)
                v.append(torch.nn.functional.normalize(m(**e).last_hidden_state[:, 0], dim=-1).float().cpu())
            v = torch.cat(v); sim = (v @ v.T).numpy()
        return bool(np.where(np.triu(np.ones_like(sim, dtype=bool), k=gap), sim, -1).max() > thr)
    return f


def mcnemar_onesided(b, c):
    """H1: base-only > adapter-only (adapter has FEWER events). Exact p = P(Bin(b+c,.5) >= b)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, j) for j in range(b, n + 1)) / 2.0 ** n


def mcnemar_twosided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, j) for j in range(0, lo + 1)) / 2.0 ** n)


def boot_ci(base_ev, ad_ev, nboot=NBOOT, seed=SEED, alpha=0.05):
    b = np.asarray(base_ev, float); a = np.asarray(ad_ev, float)
    n = len(b)
    if n == 0:
        return (math.nan, math.nan), (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    bm = b[idx].mean(1); am = a[idx].mean(1)
    abs_pp = (bm - am) * 100.0
    rel = np.where(bm > 0, (bm - am) / np.where(bm > 0, bm, 1.0) * 100.0, math.nan)
    q = lambda x: (float(np.nanpercentile(x, 100 * alpha / 2)), float(np.nanpercentile(x, 100 * (1 - alpha / 2))))
    return q(abs_pp), q(rel)


def km_median(times, events):
    """Kaplan-Meier median time-to-conclusion (events=conclusions; budget hits censored)."""
    t = np.asarray(times, float); e = np.asarray(events, bool)
    if t.size == 0:
        return math.nan, math.nan
    S = 1.0; med = math.nan
    for ut in np.unique(t):
        at_risk = int((t >= ut).sum()); d = int(((t == ut) & e).sum())
        if at_risk and d:
            S *= 1.0 - d / at_risk
        if math.isnan(med) and S <= 0.5:
            med = float(ut)
    return med, S


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag_base"); ap.add_argument("tag_adapter")
    ap.add_argument("--adapter-label", default="r2", help="label for adapter arm in prints")
    ap.add_argument("--semantic", action="store_true",
                    help="also compute BGE-M3 semantic near-dup (DESCRIPTIVE only; saturates)")
    a = ap.parse_args()

    base = load(os.path.join(RES, f"loop_base_{a.tag_base}.jsonl"))
    ad = load(os.path.join(RES, f"loop_{a.adapter_label}_{a.tag_adapter}.jsonl"))
    ids = sorted(set(base) & set(ad))
    print(f"base tag: {a.tag_base}  adapter tag: {a.tag_adapter} (label '{a.adapter_label}')")
    print(f"paired prompts: {len(ids)}  (base rows {len(base)}, adapter rows {len(ad)})")
    if not ids:
        return

    sem = make_sem_detector() if a.semantic else None
    res = {"base": {}, "ad": {}}
    for k, rows in (("base", base), ("ad", ad)):
        for i in ids:
            r = rows[i]; t = r.get("text", "") or ""
            exact = bool(find_inner_repetition(t)[0])
            budget = (r.get("finish_reason") == "length")
            res[k][i] = {"exact": exact, "budget": budget, "primary": exact or budget,
                         "tok": int(r.get("n_tokens", 0) or 0),
                         "sem": (sem(t) if sem else False),
                         "domain": r.get("domain", "?")}

    n = len(ids)
    for k, lab in (("base", "base"), ("ad", a.adapter_label)):
        g = lambda f: sum(res[k][i][f] for i in ids)
        line = (f"{lab:8s} PRIMARY {g('primary')}/{n}  exact_loop {g('exact')}/{n}  "
                f"budget_hit {g('budget')}/{n}")
        if a.semantic:
            line += f"  semantic* {g('sem')}/{n}"
        line += f"  mean_tokens {np.mean([res[k][i]['tok'] for i in ids]):.0f}"
        print(line)

    def paired(f, name):
        b = sum(res["base"][i][f] and not res["ad"][i][f] for i in ids)
        c = sum(res["ad"][i][f] and not res["base"][i][f] for i in ids)
        p1 = mcnemar_onesided(b, c); p2 = mcnemar_twosided(b, c)
        print(f"paired {name:12s}: base-only {b:3d}  adapter-only {c:3d}  "
              f"McNemar one-sided p={p1:.4f} (adapter fewer)  two-sided p={p2:.4f}")
        return b, c, p1

    print()
    _, _, p1_pr = paired("primary", "PRIMARY")
    paired("exact", "exact_loop")
    paired("budget", "budget_hit")
    if a.semantic:
        paired("sem", "semantic*")

    be = [1.0 if res["base"][i]["primary"] else 0.0 for i in ids]
    ae = [1.0 if res["ad"][i]["primary"] else 0.0 for i in ids]
    rb, ra = float(np.mean(be)), float(np.mean(ae))
    abs_pp = (rb - ra) * 100.0
    rel = ((rb - ra) / rb * 100.0) if rb > 0 else math.nan
    (ab_lo, ab_hi), (rel_lo, rel_hi) = boot_ci(be, ae)

    print(f"\nPRIMARY event rate: base {rb*100:.1f}%  {a.adapter_label} {ra*100:.1f}%")
    print(f"absolute reduction: {abs_pp:+.1f} pp  (95% CI [{ab_lo:+.1f}, {ab_hi:+.1f}], {NBOOT} bootstrap resamples)")
    print(f"relative reduction: {fmt(rel,1)}%  (95% CI [{fmt(rel_lo,1)}, {fmt(rel_hi,1)}])")

    ok_p = p1_pr < 0.025
    ok_rel = (not math.isnan(rel)) and rel >= 30.0
    ok_abs = abs_pp >= 10.0
    fails = []
    if not ok_p:
        fails.append(f"one-sided McNemar p={p1_pr:.4f} >= 0.025")
    if not ok_rel:
        fails.append(f"relative reduction {fmt(rel,1)}% < 30%")
    if not ok_abs:
        fails.append(f"absolute reduction {abs_pp:+.1f}pp < 10pp")
    print("VERDICT: SUCCESS" if not fails else "VERDICT: NOT MET")
    if fails:
        print("  failed sub-criteria: " + "; ".join(fails))

    print("\nper-domain (descriptive only):")
    for d in ("mcp", "science", "malay"):
        sub = [i for i in ids if res["base"][i]["domain"] == d]
        if not sub:
            continue
        nb = sum(res["base"][i]["primary"] for i in sub)
        na = sum(res["ad"][i]["primary"] for i in sub)
        print(f"  {d:8s} n={len(sub):3d}  base PRIMARY {nb}/{len(sub)}  {a.adapter_label} PRIMARY {na}/{len(sub)}")

    print(f"\ntokens-to-conclusion (KM; budget_hit censored at {BUDGET}):")
    for k, lab in (("base", "base"), ("ad", a.adapter_label)):
        times = [(BUDGET if res[k][i]["budget"] else res[k][i]["tok"]) for i in ids]
        events = [not res[k][i]["budget"] for i in ids]
        med, S = km_median(times, events)
        print(f"  {lab:8s} median={('not reached' if math.isnan(med) else f'{med:.0f}')}  S(end)={S:.3f}")


if __name__ == "__main__":
    main()