"""Find REAL doom loops: rollouts whose tail is a contiguous periodic repetition (period p chars, >=3 full periods, run reaches the end of text).
Streams the v3/v4/v5 rollout files (never loads them fully). Output periodic_loops_relaxed.jsonl: rollout_id,pilot_id,domain,p,run_start,periods,prefix_text (through the FIRST unit),target_text (start of the SECOND unit)."""
import json, re, sys
D = "/home/ubuntu/antidoom_data"
FILES = [f"{D}/v3/real_rollouts.jsonl", f"{D}/v4/real_rollouts_v4_fleet.jsonl", f"{D}/v5/rollouts_v5_merged.jsonl", f"{D}/v5/rollouts_v5_esc_merged.jsonl"]
held = {json.loads(l)["pilot_id"] for l in open(f"{D}/loop_testset150.jsonl")}
def find(text):
    n = len(text)
    if n < 300: return None
    tail = text[-6000:]
    m = len(tail)
    for p in range(30, min(1200, m // 2) + 1):
        if tail[m - p:] != tail[m - 2 * p:m - p]: continue
        i = m - p - 1
        while i >= 0 and tail[i] == tail[i + p]: i -= 1
        s = i + 1                                  # first index (in tail) of the periodic run
        run = m - s
        if run >= 2 * p - 5:
            unit = tail[s:s + p]
            if len(unit.strip()) < 10 or re.fullmatch(r"[\W\d_]+", unit): continue
            return p, n - m + s, run / p
    return None
out = open(f"{D}/periodic_loops_relaxed.jsonl", "w"); seen = set(); tot = kept = 0
for f in FILES:
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        rid = d.get("rollout_id"); 
        if rid in seen: continue
        seen.add(rid); tot += 1
        if d.get("finish_reason") != "length" or d.get("pilot_id") in held: continue
        t = d.get("rollout_text") or ""
        r = find(t)
        if not r: continue
        p, s, per = r
        out.write(json.dumps({"rollout_id": rid, "pilot_id": d["pilot_id"], "domain": d.get("domain"), "p": p, "run_start": s, "periods": round(per, 2),
                              "prefix_text": t[:s + p], "target_text": t[s + p:s + p + min(p, 600)], "n_chars": len(t)}, ensure_ascii=False) + "\n"); kept += 1
        if kept % 200 == 0: print(f"scanned {tot} rollouts, {kept} periodic-tail loops", flush=True); out.flush()
print(f"DONE scanned {tot}, periodic-tail loops {kept}", flush=True)
