"""Question-level GROUP rows for the group-consistency letter objective (papers: LoRA-1 PBM loss; PA-GRPO selection by base inconsistency).
usage: build_letter_group_rows.py [--scores letter_scored.jsonl] [--tag letter]
Reads letter_candidates.jsonl (every question in ALL cyclic shifts, with order/gold_orig) + base letter logprobs. Per question:
  content winner per render = argmax letter mapped to original option index; inconsistent = winners differ across shifts OR base wrong in any render.
Selection (PA-GRPO): all inconsistent train questions + `--anchor_frac` of consistent-and-correct ones. 15% of QUESTIONS held out (by qid) for validation.
Writes: <tag>_group_rows.jsonl (kind='letter_group'), <tag>_heldout.jsonl (per-render rows), all_ftpo_pairs_r2_groupmix.jsonl (loop rows + group rows, shuffled)."""
import argparse, collections, json, random
ap = argparse.ArgumentParser()
ap.add_argument("--scores", default="/home/ubuntu/antidoom_data/letter_scored.jsonl"); ap.add_argument("--tag", default="letter")
ap.add_argument("--anchor_frac", type=float, default=0.25); ap.add_argument("--holdout", type=float, default=0.15)
ap.add_argument("--mix", default="1"); a = ap.parse_args(); D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260919)
cands = [json.loads(l) for l in open(f"{D}/letter_candidates.jsonl")]
sc = {r["id"]: r["lp"] for r in map(json.loads, open(a.scores))}
byq = collections.defaultdict(list)
for c in cands:
    if c["id"] in sc: byq[c["qid"]].append(c)
qids = sorted(byq); rng.shuffle(qids); held = set(qids[: int(len(qids) * a.holdout)])
groups, hold_rows, stat = [], [], collections.Counter()
for qid in sorted(byq):
    rs = byq[qid]; n = rs[0]["n_opts"]
    if len(rs) != n: stat["incomplete_group"] += 1; continue
    if qid in held:
        for c in rs: hold_rows.append({k: c[k] for k in ("id", "qid", "domain", "n_opts", "gold_pos", "gold_letter", "context", "options", "order", "gold_orig")})
        continue
    renders, winners, wrong_any = [], [], False
    for c in sorted(rs, key=lambda x: x["gold_pos"]):
        lp = sc[c["id"]]; letters = [chr(65 + i) for i in range(n)]
        pred = max(letters, key=lambda l: lp[l]); winners.append(c["order"][ord(pred) - 65]); wrong_any |= (ord(pred) - 65) != c["gold_pos"]
        g = c["gold_letter"]; w = max([l for l in letters if l != g], key=lambda l: lp[l])
        renders.append({"context": c["context"], "order": c["order"], "gold_pos": c["gold_pos"], "chosen_letter": " " + g, "rejected_letter": " " + w})
    inconsistent = len(set(winners)) > 1 or wrong_any
    groups.append({"kind": "letter_group", "source_task": f"{a.tag}_group_v1", "source_sample_id": qid, "qid": qid, "n": n, "gold_orig": rs[0]["gold_orig"],
                   "renders": renders, "letter_tokens": [" " + chr(65 + i) for i in range(n)], "_inc": inconsistent, "_dom": rs[0]["domain"],
                   "repetition": {"domain": rs[0]["domain"], "pilot_id": rs[0]["pilot_id"]}})
inc = [g for g in groups if g["_inc"]]; con = [g for g in groups if not g["_inc"]]; rng.shuffle(con)
keep = inc + con[: int(len(con) * a.anchor_frac)]; rng.shuffle(keep)
for g in keep: g.pop("_inc"); g.pop("_dom")
with open(f"{D}/{a.tag}_group_rows.jsonl", "w") as f:
    for g in keep: f.write(json.dumps(g, ensure_ascii=False) + "\n")
with open(f"{D}/{a.tag}_heldout.jsonl", "w") as f:
    for r in hold_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
loop = open(f"{D}/all_ftpo_pairs_r2.jsonl").read().splitlines(); mixed = loop + [json.dumps(g, ensure_ascii=False) for g in keep] * int(a.mix)
rng.shuffle(mixed); open(f"{D}/all_ftpo_pairs_r2_groupmix.jsonl", "w").write("\n".join(mixed) + "\n")
doms = collections.Counter(g["repetition"]["domain"] for g in keep); ns = collections.Counter(g["n"] for g in keep)
print(f"questions {len(byq)} (held out {len(held)} -> {len(hold_rows)} renders) | groups: inconsistent {len(inc)} + anchors {len(keep)-len(inc)} (of {len(con)} consistent) = {len(keep)}")
print(f"by n_opts {dict(ns)} | domains {dict(doms)} | stat {dict(stat)}")
print(f"mixed file: {len(mixed)} rows = loop {len(loop)} + letter groups {len(keep)*int(a.mix)} ({100*len(keep)*int(a.mix)/len(mixed):.1f}% of examples; each group = n renders)")
