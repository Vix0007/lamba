"""Restart-candidate training file (DS 52): ~35% letter rows. loop subsample 10,000 + letter groups (newline+inline twins) x2 + 1,080 extra random -> 5,400. Seeded, CPU only."""
import json, random
D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260920)
mix13 = [l for l in open(f"{D}/all_ftpo_pairs_r2_mix13.jsonl").read().splitlines() if l.strip()]
loop = [l for l in mix13 if '"kind": "letter_group"' not in l and '"kind":"letter_group"' not in l]
letters = [l for l in mix13 if l not in set(loop)] if False else [l for l in mix13 if ('"kind": "letter_group"' in l or '"kind":"letter_group"' in l)]
assert len(loop) + len(letters) == len(mix13), (len(loop), len(letters), len(mix13))
loop_s = rng.sample(loop, 10000)
extra = rng.sample(letters, 5400 - 2 * len(letters))
out = loop_s + letters + letters + extra; rng.shuffle(out)
open(f"{D}/all_ftpo_pairs_r2_mix35.jsonl", "w").write("\n".join(out) + "\n")
print(f"loop {len(loop_s)} letters {len(letters)*2+len(extra)} total {len(out)} letter share {100*(len(out)-len(loop_s))/len(out):.1f}% est steps {len(out)/16:.0f}")
