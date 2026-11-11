"""Round-3 (v3) training file. Fix for the v2/phase-B defect: adapter learned to NEVER emit newlines (rejected-token list was 20% whitespace-only tokens: '\\n\\n' 15%, '\\n' 4%) -> run-on text, MORE loops.
Loop rows: drop whitespace-only rejected tokens; cap each rejected token at 2.5% of the loop rows. Letters: 35% of rows (unique groups, newline + inline twins). 2,400 rows = 150 steps. Seeded, CPU only."""
import collections, json, random
D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260920)
loop = [json.loads(l) for l in open(f"{D}/all_ftpo_pairs_r2.jsonl") if l.strip()]
letters = [l for l in open(f"{D}/all_ftpo_pairs_r2_mix13.jsonl").read().splitlines() if '"kind": "letter_group"' in l]
N_LOOP, N_LET = 1560, 840
# DS 60: also drop rejected tokens that are sentence-openers (appear in the top-50 CHOSEN set) or punctuation-only
chosen_cnt = collections.Counter(t.strip() for r in loop for t in r["multi_chosen_decoded"])
top50 = {t for t, _ in chosen_cnt.most_common(50)}
import re
keep = [r for r in loop if r["rejected_decoded"].strip() != "" and r["rejected_decoded"].strip() not in top50 and re.search(r"[A-Za-z0-9]", r["rejected_decoded"])]
print("loop rows kept after filters:", len(keep), "of", len(loop), "| dropped tokens example:", sorted(top50)[:15])
rng.shuffle(keep); cap = int(0.025 * N_LOOP); cnt = collections.Counter(); sel = []
for r in keep:
    k = r["rejected_decoded"]
    if cnt[k] >= cap: continue
    cnt[k] += 1; sel.append(r)
    if len(sel) == N_LOOP: break
assert len(sel) == N_LOOP, len(sel)
out = [json.dumps(r, ensure_ascii=False) for r in sel] + rng.sample(letters, N_LET); rng.shuffle(out)
open(f"{D}/all_ftpo_pairs_r2_v3.jsonl", "w").write("\n".join(out) + "\n")
print("loop", len(sel), "distinct rejected", len(cnt), "top", cnt.most_common(6), "| letters", N_LET, "| total", len(out), "steps", len(out) // 16)
