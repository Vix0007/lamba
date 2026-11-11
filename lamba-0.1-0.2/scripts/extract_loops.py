import json, re
from collections import defaultdict

LOOP_IDS_PATH = "/home/ubuntu/antidoom_data/v5/loop_poisoned_rollouts.json"
ROLLOUT_FILES = [
    "/home/ubuntu/antidoom_data/v5/rollouts_v5_merged.jsonl",
    "/home/ubuntu/antidoom_data/v5/rollouts_v5_esc_merged.jsonl",
]
OUT_PATH = "/home/ubuntu/antidoom_data/v5/loop_extracted.jsonl"

loop_meta = json.load(open(LOOP_IDS_PATH))
needed_ids = set(loop_meta.keys())
print(f"Need to find {len(needed_ids)} rollout_ids")

found = {}
for path in ROLLOUT_FILES:
    n = 0
    with open(path) as f:
        for line in f:
            n += 1
            # cheap pre-filter before full json parse
            # rollout_id always present; do a fast substring check first
            d = json.loads(line)
            rid = d.get("rollout_id")
            if rid in needed_ids:
                found[rid] = d
    print(f"  scanned {n} lines in {path}, found so far: {len(found)}")

print(f"\nTotal found: {len(found)}/{len(needed_ids)}")
missing = needed_ids - set(found.keys())
print(f"Missing: {len(missing)}")
if missing:
    print("sample missing:", list(missing)[:5])

# AntiDoom-style loop detector: span repeats >=4 times, >=60 chars
def find_loop(text, min_repeat=4, min_chars=60):
    n = len(text)
    # try candidate span lengths from large to reasonably small
    for span_len in range(min(400, n // min_repeat), min_chars - 1, -1):
        # slide window: check if text[i:i+span_len] repeats >= min_repeat times contiguously-ish
        # simple approach: use a rolling dict of span->positions, check for >=min_repeat occurrences
        seen = defaultdict(list)
        for i in range(0, n - span_len + 1, max(1, span_len // 4)):
            seen[text[i:i+span_len]].append(i)
        for span, positions in seen.items():
            if len(positions) >= min_repeat:
                return span, positions[0], span_len
    return None, None, None

results = []
for rid, meta in loop_meta.items():
    if rid not in found:
        continue
    text = found[rid]["rollout_text"]
    span, start, span_len = find_loop(text)
    results.append({
        "rollout_id": rid,
        "pilot_id": meta["pilot_id"],
        "domain": meta["domain"],
        "repeat": meta["repeat"],
        "finish": meta["finish"],
        "len": meta["len"],
        "text_len": len(text),
        "correct": found[rid].get("_correct"),
        "detected_span": span,
        "detected_start_char": start,
        "detected_span_len": span_len,
        "rollout_text": text,
    })
    if len(results) % 1000 == 0:
        print(f"  processed {len(results)}/{len(found)}")

with open(OUT_PATH, "w") as f:
    for r in results:
        f.write(json.dumps(r, default=str) + "\n")

print(f"\nSaved {len(results)} extracted+detected loop rollouts to {OUT_PATH}")
detected_ct = sum(1 for r in results if r["detected_span"])
print(f"Loop span successfully detected in {detected_ct}/{len(results)} ({detected_ct/len(results):.1%})")
