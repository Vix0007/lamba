"""Compare base vs adapter GPQA-D runs: accuracy, forced-cutoff, exact-match doom loops, paired stats."""
import json, sys, math
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from antidoom.repetition import find_inner_repetition

R = "/home/ubuntu/eval_results/{}_gpqa_full198_samples.jsonl"
arms = {"base": "ab_base_lamba", "ftpo1": "ab_ftpo1"}
rows = {a: {} for a in arms}
for a, lab in arms.items():
    for line in open(R.format(lab)):
        r = json.loads(line); rows[a][r["record_id"]] = r
ids = sorted(set(rows["base"]) & set(rows["ftpo1"]))
print("paired items:", len(ids))

def loop(text):
    found, hit = find_inner_repetition(text)
    return bool(found)

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

for a in arms:
    rs = [rows[a][i] for i in ids]
    n = len(rs)
    acc = sum(r["correct"] for r in rs); fc = sum(r["forced_cutoff"] for r in rs)
    lp = sum(loop(r["reasoning"]) for r in rs)
    fc_and_loop = sum(r["forced_cutoff"] and loop(r["reasoning"]) for r in rs)
    toks = sum(len(r["reasoning"]) for r in rs) / n
    lo, hi = wilson(acc, n)
    print(f"{a:6s} acc {acc}/{n}={acc/n:.3f} [{lo:.3f},{hi:.3f}]  forced_cutoff {fc}/{n}={fc/n:.3f}  "
          f"exact_loops {lp}/{n}={lp/n:.3f} (of which forced {fc_and_loop})  mean_reasoning_chars {toks:.0f}")

b, f = [rows["base"][i] for i in ids], [rows["ftpo1"][i] for i in ids]
b_only = sum(x["correct"] and not y["correct"] for x, y in zip(b, f))
f_only = sum(y["correct"] and not x["correct"] for x, y in zip(b, f))
print(f"paired accuracy: base-only-correct {b_only}, adapter-only-correct {f_only} (McNemar discordant pairs)")
bl = sum(loop(x["reasoning"]) and not loop(y["reasoning"]) for x, y in zip(b, f))
fl = sum(loop(y["reasoning"]) and not loop(x["reasoning"]) for x, y in zip(b, f))
print(f"loops: base-only {bl}, adapter-only {fl}")
same = sum(x["reasoning"] == y["reasoning"] for x, y in zip(b, f))
print(f"identical reasoning text: {same}/{len(ids)}")
