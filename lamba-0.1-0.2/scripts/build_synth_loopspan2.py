"""Synthetic loop pool v2 (10x, diverse): mimic the real loop forms measured in base outputs.
Per row: pick a healthy base rollout; unit length ~ log-uniform 30..900 chars (real periods: median ~150, range 15..900); 40% sentence-snapped / 60% arbitrary char alignment (real loops are often mid-sentence);
repeats before the target r: 1 (40%), 2 (25%), 3-6 (35%) adjacent copies; 20% are NON-adjacent 'return to earlier text' (earlier segment re-emitted after a 0-1500 char gap);
position: 50% shallow (1.5k-6k chars in), 50% deep (6k+); domains mcp 40 / science 35 / malay 25. Target = next verbatim copy (<=600 chars)."""
import json, random, re, glob, sys, collections, math
D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260921)
held = {json.loads(l)["pilot_id"] for l in open(f"{D}/loop_testset150.jsonl")}
prompts = {}
for f in glob.glob(f"{D}/v[345]/real_prompts*.jsonl"):
    for l in open(f):
        d = json.loads(l); pid = d.get("pilot_id")
        if pid and d.get("prompt"): prompts[pid] = d["prompt"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
WANT = {"mcp": 0.40, "science": 0.35, "malay": 0.25}
bound = re.compile(r"(?<=[\.\?\!:])[ \n]|\n\n")
pool = collections.defaultdict(list); seen = set()
for f in [f"{D}/v5/rollouts_v5_merged.jsonl", f"{D}/v5/rollouts_v5_esc_merged.jsonl", f"{D}/v4/real_rollouts_v4_fleet.jsonl", f"{D}/v3/real_rollouts.jsonl"]:
    for line in open(f):
        if rng.random() > 0.20: continue
        try: d = json.loads(line)
        except Exception: continue
        pid = d.get("pilot_id"); dom = d.get("domain")
        if dom not in WANT or pid in held or pid not in prompts or d["rollout_id"] in seen: continue
        t = d.get("rollout_text") or ""
        if not (4000 <= len(t) <= 18000): continue
        seen.add(d["rollout_id"]); pool[dom].append((pid, t))
for k in pool: rng.shuffle(pool[k])
print({k: len(v) for k, v in pool.items()}, flush=True)
rows = []; per_prompt = collections.Counter()
def snap(bs, x):
    return next((b for b in bs if b >= x), None)
for dom, frac in WANT.items():
    target_n = int(N * frac); made = 0
    for pid, t in pool[dom]:
        if made >= target_n: break
        if per_prompt[pid] >= 3: continue
        bs = [m.end() for m in bound.finditer(t)]
        L = int(math.exp(rng.uniform(math.log(30), math.log(900))))
        deep = rng.random() < 0.5
        lo, hi = (6000, len(t) - 2500) if deep else (1500, 6000)
        if hi - lo < 300: lo, hi = 1500, max(1800, len(t) - 2500)
        a = rng.randint(lo, hi)
        if rng.random() < 0.4:
            a2 = snap(bs, a); e = snap(bs, (a2 or a) + L)
            if a2 is None or e is None: continue
            a = a2
        else: e = a + L
        if e + 1 > len(t) or e - a < 25: continue
        unit = t[a:e]
        if len(unit.strip()) < 20 or re.fullmatch(r"[\W\d_]+", unit.strip()): continue
        if rng.random() < 0.2:                       # non-adjacent 'return to earlier text'
            gap = rng.randint(0, 1500); e2 = min(len(t), e + gap)
            pre = t[:e2]; kind = "return"
        else:
            u = rng.random(); r = 1 if u < 0.40 else (2 if u < 0.65 else rng.randint(3, 6))
            pre = t[:a] + unit * (r + 1); kind = f"adj{r}"
        rows.append({"kind": "loop_span", "pilot_id": pid, "domain": dom, "src": "synth2", "form": kind, "prefix_text": pre, "target_text": unit[:600]}); per_prompt[pid] += 1; made += 1
rng.shuffle(rows)
print("rows", len(rows), collections.Counter(r["domain"] for r in rows), collections.Counter(r["form"][:3] for r in rows))
open(f"{D}/synth2_loopspan_raw.jsonl", "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
