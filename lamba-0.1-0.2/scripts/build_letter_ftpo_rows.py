"""Turn base-scored letter candidates into FTPO rows + a held-out validation set.
usage: build_letter_ftpo_rows.py [--scores letter_scored.jsonl] [--tag letter]
Design (DS-reviewed, adopted): chosen=[gold letter], rejected=base's top WRONG letter; keep hard items (-4 <= m0 < 2 nats, m0 = logit(gold)-logit(top wrong));
drop m0 < -4 (likely label noise); add ~20% easy items (m0>=2) as tether anchors; re-balance gold positions per n_opts AFTER filtering; hold out 15% of
QUESTIONS (all renders) for validation. Rows carry kind='letter' so ftpo_train bypasses loop-data rebalancing."""
import argparse, collections, json, random
ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="/home/ubuntu/antidoom_data/letter_scored.jsonl"); ap.add_argument("--tag", default="letter")
ap.add_argument("--easy_frac", type=float, default=0.20); ap.add_argument("--holdout", type=float, default=0.15)
a = ap.parse_args(); D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260919)
cands = {c["id"]: c for c in map(json.loads, open(f"{D}/letter_candidates.jsonl"))}
sc = {r["id"]: r["lp"] for r in map(json.loads, open(a.scores))}
qids = sorted({c["qid"] for c in cands.values()}); rng.shuffle(qids)
held = set(qids[: int(len(qids) * a.holdout)])
hold_rows, hard, easy, dropped = [], [], [], collections.Counter()
for cid, c in cands.items():
    if cid not in sc: dropped["no_score"] += 1; continue
    if c["qid"] in held:
        hold_rows.append({k: c[k] for k in ("id", "qid", "domain", "n_opts", "gold_pos", "gold_letter", "context", "options")}); continue
    lp = sc[cid]; letters = [chr(65 + i) for i in range(c["n_opts"])]
    g = c["gold_letter"]; wrong = [l for l in letters if l != g]
    w = max(wrong, key=lambda l: lp[l]); m0 = lp[g] - lp[w]
    row = {"kind": "letter", "source_task": f"{a.tag}_v1", "source_sample_id": cid, "context_with_chat_template": c["context"],
           "rejected_decoded": " " + w, "multi_chosen_decoded": [" " + g],
           "repetition": {"domain": c["domain"], "pilot_id": c["pilot_id"]}, "_n": c["n_opts"], "_g": g, "_m0": round(m0, 3)}
    if m0 < -4: dropped["m0<-4_label_noise"] += 1
    elif m0 < 2: hard.append(row)
    else: easy.append(row)
def balance(rows):
    """equalise gold letters within each n_opts group by downsampling to the minimum count"""
    out = []
    for n in sorted({r["_n"] for r in rows}):
        by = collections.defaultdict(list)
        for r in rows:
            if r["_n"] == n: by[r["_g"]].append(r)
        k = min(len(v) for v in by.values()) if len(by) == n else 0
        for v in by.values(): rng.shuffle(v); out += v[:k]
    return out
hard_b = balance(hard)
n_easy = int(len(hard_b) * a.easy_frac / (1 - a.easy_frac)); rng.shuffle(easy)
easy_b = balance(easy[: max(n_easy * 2, 0)])[:n_easy]
final = hard_b + easy_b; rng.shuffle(final)
for r in final: r.pop("_n"); r.pop("_g"); r.pop("_m0")
with open(f"{D}/{a.tag}_ftpo_rows.jsonl", "w") as f:
    for r in final: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(f"{D}/{a.tag}_heldout.jsonl", "w") as f:
    for r in hold_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
# mixed training file = loop rows + letter rows (kind marks letter rows), shuffled
loop = open(f"{D}/all_ftpo_pairs_r2.jsonl").read().splitlines(); mixed = loop + [json.dumps(r, ensure_ascii=False) for r in final]
rng.shuffle(mixed); open(f"{D}/all_ftpo_pairs_r2_mixed.jsonl", "w").write("\n".join(mixed) + "\n")
gc = collections.Counter(json.loads(json.dumps(r))["multi_chosen_decoded"][0] for r in final); rc = collections.Counter(r["rejected_decoded"] for r in final)
print(f"scored {len(sc)} | held-out questions {len(held)} -> {len(hold_rows)} renders | dropped {dict(dropped)}")
print(f"hard {len(hard)} (balanced {len(hard_b)}) | easy pool {len(easy)} (used {len(easy_b)}) | FINAL letter rows {len(final)}")
print("gold letters:", dict(sorted(gc.items())), "| rejected letters:", dict(sorted(rc.items())))
print(f"mixed file: {len(mixed)} rows = loop {len(loop)} + letter {len(final)} ({100*len(final)/len(mixed):.1f}% letter)")
