"""Final Round-2 training file: loop rows + newline letter groups + inline-format twins (~13% letters). Validates with LetterGroupDataset."""
import collections, json, random, sys
sys.path.insert(0, "/home/ubuntu/antidoom/src")
D = "/home/ubuntu/antidoom_data"
groups = [json.loads(l) for l in open(f"{D}/letter_group_rows.jsonl")]
cands = collections.defaultdict(list)
for l in open(f"{D}/letter_inline_candidates.jsonl"):
    c = json.loads(l); cands[c["qid"]].append(c)
sc = {r["id"]: r["lp"] for r in map(json.loads, open(f"{D}/letter_inline_scored.jsonl"))}
twins, skipped = [], 0
for g in groups:
    cs = sorted(cands.get(g["qid"], []), key=lambda c: c["gold_pos"]); n = g["n"]
    if len(cs) != n or any(c["id"] not in sc for c in cs): skipped += 1; continue
    renders = []
    for c in cs:
        lp = sc[c["id"]]; letters = [chr(65 + i) for i in range(n)]; gl = chr(65 + c["gold_pos"])
        w = max([l for l in letters if l != gl], key=lambda l: lp[l])
        renders.append({"context": c["context"], "order": c["order"], "gold_pos": c["gold_pos"], "chosen_letter": " " + gl, "rejected_letter": " " + w})
    twins.append({"kind": "letter_group", "source_task": "letter_group_inline_v1", "source_sample_id": g["qid"] + "_inl", "qid": g["qid"] + "_inl", "n": n,
                  "gold_orig": g["gold_orig"], "renders": renders, "letter_tokens": g["letter_tokens"], "repetition": g["repetition"]})
loop = open(f"{D}/all_ftpo_pairs_r2.jsonl").read().splitlines()
mix = loop + [json.dumps(g, ensure_ascii=False) for g in groups] + [json.dumps(t, ensure_ascii=False) for t in twins]
random.Random(20260919).shuffle(mix); open(f"{D}/all_ftpo_pairs_r2_mix13.jsonl", "w").write("\n".join(mix) + "\n")
nl = len(groups) + len(twins)
print(f"loop {len(loop)} + newline groups {len(groups)} + inline twins {len(twins)} (skipped {skipped}) = {len(mix)} rows | letters {100*nl/len(mix):.1f}% | est steps {len(mix)/16:.0f}")
from transformers import AutoTokenizer
from antidoom.ftpo_data import LetterGroupDataset
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
ds = LetterGroupDataset([json.loads(x) for x in (json.dumps(g) for g in twins)], tok, max_seq_length=6000)
print("inline twins kept by LetterGroupDataset:", len(ds), dict(ds.filter_counts))
print("example inline context tail:", repr(twins[0]["renders"][0]["context"][-110:]))
