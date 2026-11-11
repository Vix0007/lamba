"""v7 (DS 67): Liquid-AntiDoom-faithful loop detection over ALL saved base rollouts.
Detector: 60-char window (anchors every 30 chars, >=25 alnum chars) that occurs >=4 times NON-OVERLAPPING in the text (str.count semantics). p0 = first occurrence, p1 = SECOND occurrence = start of the first REPEAT.
Rejected position = first non-whitespace char c at/after p1 (token containing c); context = text before that token. Held-out test prompts and the gen1 eval-pool prompts are excluded.
Output antidoom7_events.jsonl: rollout_id, pilot_id, domain, src, n_occ, p0, p1, c, span60, prefix_text (= text[:c-lead]) -- tokenisation/rejected-token resolution happens in the pair builder."""
import json, re, sys, collections
D = "/home/ubuntu/antidoom_data"
held = {json.loads(l)["pilot_id"] for l in open(f"{D}/loop_testset150.jsonl")} | {json.loads(l)["id"] for l in open(f"{D}/loopgen_prompts.jsonl")}
FILES = [(f"{D}/v3/real_rollouts.jsonl", "v3"), (f"{D}/v4/real_rollouts_v4_fleet.jsonl", "v4"), (f"{D}/v5/rollouts_v5_merged.jsonl", "v5"), (f"{D}/v5/rollouts_v5_esc_merged.jsonl", "v5esc")]
W = 60; STRIDE = 30
al = re.compile(r"[A-Za-z0-9]")
out = open(f"{D}/antidoom7_events.jsonl", "w"); seen = set(); tot = ev = 0; by = collections.Counter()
for f, src in FILES:
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        rid = d.get("rollout_id")
        if rid in seen: continue
        seen.add(rid); tot += 1
        pid = d.get("pilot_id")
        if pid in held: continue
        t = d.get("rollout_text") or ""
        if len(t) < 600: continue
        found = None
        for i in range(0, len(t) - W, STRIDE):
            w = t[i:i + W]
            if len(al.findall(w)) < 25: continue
            n = t.count(w)
            if n >= 4:
                p0 = t.find(w); p1 = t.find(w, p0 + W)
                if p1 < 0: continue
                found = (n, p0, p1, w); break
        if not found: continue
        n, p0, p1, w = found
        while p0 > 0 and t[p0 - 1] == t[p1 - 1]:          # extend LEFT to the true start of the repeated segment (anchors are at arbitrary offsets)
            p0 -= 1; p1 -= 1
        c = p1
        if c > 0 and t[c].isalnum() and t[c - 1].isalnum():  # mid-word start: snap to the next word start
            while c < len(t) and not t[c].isspace(): c += 1
        while c < len(t) and t[c].isspace(): c += 1
        if c >= len(t) - 5: continue
        out.write(json.dumps({"rollout_id": rid, "pilot_id": pid, "domain": d.get("domain"), "src": src, "finish": d.get("finish_reason"), "n_occ": n, "p0": p0, "p1": p1, "c": c, "span60": w,
                              "prefix_text": t[:c], "next_text": t[c:c + 200]}, ensure_ascii=False) + "\n"); ev += 1; by[src] += 1
        if ev % 500 == 0: print(f"scanned {tot}, events {ev}", dict(by), flush=True); out.flush()
print(f"DONE scanned {tot} events {ev}", dict(by), flush=True)
