"""Ruminconfab final row assembly: matches all_ftpo_pairs_round2.jsonl schema exactly.
A -> hard FTPO (kind=None/implicit): rejected_decoded + multi_chosen_decoded (all other top-20
    alternatives from letterfix2-c10 at the onset position -- same "broad alternatives" approach
    as the proven v8/v9c recipe, not hand-curated per DS's own "reuse the row-builder verbatim").
B3 -> ftpo_soft: rejected_decoded + base_p_rej + gamma=4.0/tau=0.7 (matching this project's
    established soft-KL defaults from the round-2 antidoom recipe).
"""
import json
from transformers import AutoTokenizer

TOK_PATH = "/home/ubuntu/models/lmh_letterfix2_c10"
tok = AutoTokenizer.from_pretrained(TOK_PATH)

cands = {json.loads(l)["source_sample_id"]: json.loads(l)
         for l in open("/home/ubuntu/antidoom_data/ruminconfab_onset_cands.jsonl")}
alts = {json.loads(l)["source_sample_id"]: json.loads(l)
        for l in open("/home/ubuntu/antidoom_data/ruminconfab_onset_alts.jsonl")}
manifest = json.load(open("/tmp/onset_manifest_full.json"))

rows = []
issues = []
for cid, cand in cands.items():
    case = manifest[cid]
    context = cand["context_with_chat_template"]
    # rejected token = first token the model actually generated at this position (from the
    # rollout, not from the alt query) -- tokenize the KNOWN-bad continuation's first token.
    # We stored context up to (not including) the onset; the rejected text starts right after.
    onset_quote_raw = case["onset_quote"]
    rejected_text = onset_quote_raw.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
    rejected_first_tok_ids = tok(rejected_text, add_special_tokens=False)["input_ids"][:1]
    if not rejected_first_tok_ids:
        issues.append((cid, "empty rejected token"))
        continue
    rejected_id = rejected_first_tok_ids[0]
    rejected_decoded = tok.decode([rejected_id])

    top = alts[cid]["top"]  # list of [token_id, decoded_str, logprob]
    top_ids = {t[0]: (t[1], t[2]) for t in top}

    if rejected_id in top_ids:
        base_p_rej = float(__import__("math").exp(top_ids[rejected_id][1]))
    else:
        # not in top-20 -- genuinely low probability under c10; use a small floor rather than
        # guessing, consistent with "rare enough not to be in top-20" (<~top20's min prob)
        min_p_in_top20 = min(float(__import__("math").exp(t[2])) for t in top)
        base_p_rej = min_p_in_top20 * 0.5
        issues.append((cid, f"rejected token {rejected_id!r} ({rejected_decoded!r}) not in top-20; using floor p={base_p_rej:.4f}"))

    behavior = "A" if cid.startswith("A") else "B3"
    if behavior == "A":
        chosen_alts = [t[1] for t in top if t[0] != rejected_id]
        if not chosen_alts:
            issues.append((cid, "no chosen alternatives after excluding rejected"))
            continue
        rows.append({
            "source_task": "ruminconfab_r2", "source_sample_id": cid,
            "context_with_chat_template": context, "rejected_decoded": rejected_decoded,
            "domain": "ruminconfab_A", "kind": None,
            "multi_chosen_decoded": chosen_alts,
            "base_p_rej": base_p_rej,
            "_chosen_description": case["chosen_description"], "_confidence": case["confidence"],
        })
    else:  # B3 -> soft
        rows.append({
            "source_task": "ruminconfab_r2", "source_sample_id": cid,
            "context_with_chat_template": context, "rejected_decoded": rejected_decoded,
            "domain": "ruminconfab_B3", "kind": "ftpo_soft",
            "base_p_rej": base_p_rej, "gamma": 4.0, "tau": 0.7, "weight": 1.0,
            "_chosen_description": case["chosen_description"], "_confidence": case["confidence"],
        })

print(f"built {len(rows)} rows from {len(cands)} candidates")
if issues:
    print(f"{len(issues)} issues:")
    for cid, msg in issues:
        print(f"  {cid}: {msg}")

a_rows = [r for r in rows if r["domain"] == "ruminconfab_A"]
b3_rows = [r for r in rows if r["domain"] == "ruminconfab_B3"]
print(f"A rows: {len(a_rows)}, B3 rows: {len(b3_rows)}")
print(f"mean base_p_rej (A): {sum(r['base_p_rej'] for r in a_rows)/max(1,len(a_rows)):.4f}")
print(f"mean base_p_rej (B3): {sum(r['base_p_rej'] for r in b3_rows)/max(1,len(b3_rows)):.4f}")

with open("/home/ubuntu/antidoom_data/ruminconfab_rows_AB3.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
