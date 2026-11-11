"""Second copy of the selected letter groups in INLINE format (stem A. x B. y ... \nAnswer:). Same questions/orders as the newline groups.
Writes letter_inline_candidates.jsonl (needs base letter scoring -> letter_inline_scored.jsonl via score_letters.py)."""
import collections, json
D = "/home/ubuntu/antidoom_data"
groups = [json.loads(l) for l in open(f"{D}/letter_group_rows.jsonl")]           # train groups (kind=letter_group), newline format
keep = {g["qid"] for g in groups}
cands = [json.loads(l) for l in open(f"{D}/letter_candidates.jsonl")]
out = []
for c in cands:
    if c["qid"] not in keep: continue
    stem = c["context"].split("\nA. ")[0]; opts = c["options"]
    ctx = stem + " " + " ".join(f"{chr(65+i)}. {t}" for i, t in enumerate(opts)) + "\nAnswer:"
    d = dict(c); d["id"] = c["id"] + "_inl"; d["context"] = ctx; out.append(d)
with open(f"{D}/letter_inline_candidates.jsonl", "w") as f:
    for d in out: f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"inline candidates {len(out)} from {len(keep)} questions; n_opts", dict(collections.Counter(d['n_opts'] for d in out)))
print("example:", repr(out[0]["context"][-150:]))
