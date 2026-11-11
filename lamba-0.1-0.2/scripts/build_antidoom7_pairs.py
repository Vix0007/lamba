"""v7 step 3: assemble FTPO pairs (DS 67 rules): chosen = base top-20 alternatives excluding the rejected id, decoded (lstrip) has >=1 alphanumeric, stripped len>=1, not special; need >=3;
caps: per rejected id 150, single-character rejected tokens 50 each; dedup (context tail, rejected). Writes all_ftpo_pairs_antidoom7.jsonl + audit sample."""
import collections, json, random, re, sys
D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260920)
alts = {}
for l in open(f"{D}/antidoom7_alts.jsonl"):
    r = json.loads(l); alts[r["source_sample_id"]] = r["top"]
al = re.compile(r"[A-Za-z0-9]")
rows = []; stats = collections.Counter()
for l in open(f"{D}/antidoom7_cands.jsonl"):
    r = json.loads(l); top = alts.get(r["source_sample_id"])
    if not top: stats["no_alts"] += 1; continue
    rej = r["rejected_id"]; ch = []
    rej_lp = None
    for tid, dec, lp in top:
        if tid == rej: rej_lp = lp; continue
        if dec.startswith("<|") or not al.search(dec) or len(dec.strip()) < 1: continue
        ch.append(dec)
        if len(ch) >= 20: break
    if len(ch) < 3: stats["lt3_chosen"] += 1; continue
    r["multi_chosen_decoded"] = ch; r["rejected_logprob_base"] = rej_lp; r.pop("rejected_id", None)
    rows.append(r)
rng.shuffle(rows)
cnt = collections.Counter(); kept = []; seen = set()
for r in rows:
    k = r["rejected_decoded"]; cap = 50 if len(k.strip()) == 1 else 150
    key = (r["context_with_chat_template"][-300:], k)
    if cnt[k] >= cap or key in seen: stats["capped_or_dup"] += 1; continue
    cnt[k] += 1; seen.add(key); kept.append(r)
open(f"{D}/all_ftpo_pairs_antidoom7.jsonl", "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
dom = collections.Counter(r["repetition"]["domain"] for r in kept); fin = collections.Counter(r["repetition"]["finish"] for r in kept)
inb = sum(1 for r in kept if r["rejected_logprob_base"] is not None)
print("pairs", len(kept), dict(stats), dict(dom), dict(fin), "| rejected in base top-20:", inb, "| distinct rejected", len(cnt), "top", cnt.most_common(10))
for r in kept[:6]:
    print("--", repr(r["context_with_chat_template"][-90:]), "| REJ", repr(r["rejected_decoded"]), "lp", r["rejected_logprob_base"], "| CH", r["multi_chosen_decoded"][:5])
