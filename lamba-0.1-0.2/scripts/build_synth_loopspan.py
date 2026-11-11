"""Synthetic REPEAT rows for loop-span unlikelihood: take healthy base rollouts (mcp/science, non-held-out), copy a 60-400 char unit (snapped to sentence/newline boundaries) right after itself
(30%: twice) so the context contains the repeat and the target is the next verbatim copy. The hinge only bites where the model is actually about to keep copying (mid-repeat tokens)."""
import json, random, re, glob, sys, collections
D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260920)
held = {json.loads(l)["pilot_id"] for l in open(f"{D}/loop_testset150.jsonl")}
prompts = {}
for f in glob.glob(f"{D}/v[345]/real_prompts*.jsonl"):
    for l in open(f):
        d = json.loads(l); pid = d.get("pilot_id")
        if pid and d.get("prompt"): prompts[pid] = d["prompt"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
bound = re.compile(r"(?<=[\.\?\!:])[ \n]|\n\n")
cands = []; seen = set()
for f in [f"{D}/v5/rollouts_v5_merged.jsonl", f"{D}/v4/real_rollouts_v4_fleet.jsonl", f"{D}/v3/real_rollouts.jsonl"]:
    for line in open(f):
        if rng.random() > 0.06: continue
        try: d = json.loads(line)
        except Exception: continue
        pid = d.get("pilot_id")
        if d.get("domain") not in ("mcp", "science") or pid in held or pid not in prompts or d["rollout_id"] in seen: continue
        t = d.get("rollout_text") or ""
        if not (6000 <= len(t) <= 18000): continue
        seen.add(d["rollout_id"]); cands.append((pid, d["domain"], t))
rng.shuffle(cands); print("candidate rollouts", len(cands), flush=True)
rows = []; per_prompt = collections.Counter()
for pid, dom, t in cands:
    if len(rows) >= N: break
    if per_prompt[pid] >= 2: continue
    bs = [m.end() for m in bound.finditer(t[:len(t) - 1200])]
    bs = [b for b in bs if b >= 1500]
    if len(bs) < 3: continue
    a = rng.choice(bs)
    L = rng.randint(80, 400)
    e = next((b for b in bs if b >= a + L), None)
    if e is None: continue
    unit = t[a:e]
    if len(unit.strip()) < 60: continue
    reps = 2 if rng.random() < 0.3 else 1
    pre = t[:a] + unit * reps
    rows.append({"kind": "loop_span", "pilot_id": pid, "domain": dom, "src": "synth", "prefix_text": pre, "target_text": unit[:600]}); per_prompt[pid] += 1
print("synthetic rows", len(rows), collections.Counter(r["domain"] for r in rows))
open(f"{D}/synth_loopspan_raw.jsonl", "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
