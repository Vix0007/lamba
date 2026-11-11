"""Phase-B training file (DS 54): init from v2 checkpoint-100; 85% letter groups / 15% loop anchor rows, 2000 rows (=125 steps). Seeded, CPU only."""
import json, random
D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260920)
rows = [l for l in open(f"{D}/all_ftpo_pairs_r2_mix35.jsonl").read().splitlines() if l.strip()]
isl = lambda l: '"kind": "letter_group"' in l
letters = list(dict.fromkeys(l for l in rows if isl(l)))   # unique
loop = [l for l in rows if not isl(l)]
out = rng.sample(letters, 1700) + rng.sample(loop, 300); rng.shuffle(out)
open(f"{D}/all_ftpo_pairs_r2_phaseB.jsonl", "w").write("\n".join(out) + "\n")
print("unique letters available", len(letters), "written", len(out), "letters 1700 loop 300")
