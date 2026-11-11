"""MALAY assembly (DS 88): cap 5 rows per rejected string (2 for single chars); stopword rejected tokens and list-loop candidates -> SOFT rows only (never hard FTPO). Based on assemble_round2.py (DS 74). usage: assemble_round2.py <cands.jsonl> <alts.jsonl> <out_pairs.jsonl>
Per candidate (context ending before the first token of repeat #k): base top-20 logprobs from the alts query; p_rej = exp(logprob of the rejected id) if in top-20 else 0 (tiny).
repeat #1 -> HARD FTPO row (format of the v8 pairs: multi_chosen_decoded = base top alternatives, alnum, >=3), kept only if log p(rej) < -0.5 and gap (lp_rej - best chosen) < 5.0.
repeat #2/#3 -> SOFT row kind='ftpo_soft': weight = min(1, (1-p_rej)/0.5), skipped if p_rej > 0.995; gamma 4.0, tau 0.7.
Also caps: per rejected decoded string 60 rows; single-char rejected 20; dedup by (context tail, rejected)."""
import collections, json, math, re, sys
cp, ap, outp = sys.argv[1:4]
alts = {json.loads(l)["source_sample_id"]: json.loads(l)["top"] for l in open(ap)}
al = re.compile(r"[A-Za-z0-9]"); rows = []; st = collections.Counter(); cnt = collections.Counter(); seen = set()
for l in open(cp):
    r = json.loads(l); top = alts.get(r["source_sample_id"])
    if not top: st["no_alts"] += 1; continue
    rej = r["rejected_id"]; lp_rej = next((lp for tid, dec, lp in top if tid == rej), None)
    p_rej = math.exp(lp_rej) if lp_rej is not None else 0.0
    k = r["repeat_k"]; key = (r["context_with_chat_template"][-300:], r["rejected_decoded"]); dec = r["rejected_decoded"]
    if key in seen: st["dup"] += 1; continue
    cap = 2 if len(dec.strip()) == 1 else 5
    if cnt[dec] >= cap: st["capped"] += 1; continue
    base = {"source_task": "round2", "source_sample_id": r["source_sample_id"], "context_with_chat_template": r["context_with_chat_template"], "rejected_decoded": dec,
            "repetition": {"domain": r["domain"], "pilot_id": r["prompt_id"], "repeat_k": k, "n_occ": r["n_occ"]}, "base_p_rej": p_rej}
    if k == 1 and r.get('stopword'):
        if lp_rej is None or p_rej > 0.9999: st['stop_r1_skipped'] += 1; continue
        base.update({'kind': 'ftpo_soft', 'weight': min(1.0, (1.0 - p_rej) / 0.5), 'gamma': 4.0, 'tau': 0.7}); rows.append(base); st['soft_r1_stopword'] += 1
    elif k == 1:
        if lp_rej is None or p_rej >= 0.90: st["r1_filtered_certain"] += 1; continue
        ch = []
        for tid, d2, lp in top:
            if tid == rej or d2.startswith("<|") or not al.search(d2) or len(d2.strip()) < 1: continue
            ch.append((d2, lp))
            if len(ch) >= 20: break
        if p_rej < 0.60:
            if len(ch) < 3 or lp_rej - ch[0][1] >= 5.0: st["r1_filtered_gap_or_chosen"] += 1; continue
            base["multi_chosen_decoded"] = [d for d, _ in ch]; rows.append(base); st["hard_r1"] += 1
        else:                                     # 0.60 <= p < 0.90: smoothed KL, weight 0.5*(0.90-p)/0.30
            base.update({"kind": "ftpo_soft", "weight": 0.5 * (0.90 - p_rej) / 0.30, "gamma": 4.0, "tau": 0.7}); rows.append(base); st["soft_r1_mid"] += 1
    else:
        if p_rej > 0.9999: st["soft_skipped_certain"] += 1; continue
        w0 = min(1.0, (1.0 - p_rej) / 0.5)
        w = w0 if p_rej <= 0.995 else w0 * (0.9999 - p_rej) / 0.0049
        base.update({"kind": "ftpo_soft", "weight": w, "gamma": 4.0, "tau": 0.7}); rows.append(base); st[f"soft_r{k}"] += 1
    cnt[dec] += 1; seen.add(key)
open(outp, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(dict(st), "rows", len(rows), "mean soft weight", (sum(r["weight"] for r in rows if r.get("kind") == "ftpo_soft") / max(1, sum(1 for r in rows if r.get("kind") == "ftpo_soft"))))
