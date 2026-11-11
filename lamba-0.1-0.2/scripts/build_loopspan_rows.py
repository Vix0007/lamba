"""Build kind='loop_span' rows from REAL contiguous-periodic loops: (1) periodic_loops_relaxed.jsonl (mined from old rollouts), (2) fresh base greedy generations eval_results/loop_base_gen1.jsonl.
context = chat template(prompt) + text through FIRST unit; target = start of SECOND unit. Cap malay share, exclude held-out prompts."""
import glob, json, random, re, sys, collections
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from transformers import AutoTokenizer
D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260920)
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
held = {json.loads(l)["pilot_id"] for l in open(f"{D}/loop_testset150.jsonl")}
prompts = {}
for f in glob.glob(f"{D}/v[345]/real_prompts*.jsonl") + [f"{D}/loopgen_prompts.jsonl"]:
    for l in open(f):
        d = json.loads(l); pid = d.get("pilot_id") or d.get("id")
        if pid and d.get("prompt"): prompts[pid] = d["prompt"]
def find(text, pmin=15, ratio=2):
    n = len(text)
    if n < 300: return None
    tail = text[-6000:]; m = len(tail)
    for p in range(pmin, min(1200, m // ratio) + 1):
        if tail[m - p:] != tail[m - 2 * p:m - p]: continue
        i = m - p - 1
        while i >= 0 and tail[i] == tail[i + p]: i -= 1
        s = i + 1; run = m - s
        if run >= 2 * p - 5:
            unit = tail[s:s + p]
            if len(unit.strip()) < 10 or re.fullmatch(r"[\W\d_]+", unit): continue
            return p, n - m + s
    return None
items = []
for l in open(f"{D}/periodic_loops_relaxed.jsonl"):
    d = json.loads(l); items.append((d["pilot_id"], d["domain"], d["prefix_text"], d["target_text"], "old"))
try:
    for l in open("/home/ubuntu/eval_results/loop_base_gen1.jsonl"):
        d = json.loads(l)
        if d.get("finish_reason") != "length": continue
        t = d["text"]; r = find(t)
        if r:
            p, s = r; items.append((d["id"], d.get("domain"), t[:s + p], t[s + p:s + p + min(p, 600)], "gen"))
except FileNotFoundError: pass
rows, seen = [], set()
for pid, dom, pre, tgt, src in items:
    if pid in held or pid not in prompts: continue
    key = (pid, tgt[:60])
    if key in seen: continue
    seen.add(key)
    ctx = tok.apply_chat_template([{"role": "user", "content": prompts[pid]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium") + pre
    rows.append({"kind": "loop_span", "pilot_id": pid, "domain": dom, "src": src, "context": ctx, "target_text": tgt})
c = collections.Counter((r["domain"], r["src"]) for r in rows); print("rows", len(rows), dict(c))
# cap malay at 35% of the final set
non = [r for r in rows if r["domain"] != "malay"]; mal = [r for r in rows if r["domain"] == "malay"]
cap = int(0.35 / 0.65 * len(non)) if non else 60
cap = max(cap, 60); rng.shuffle(mal); rows = non + mal[:cap]
n_real = len(rows)
import os
SYN = os.environ.get("SYNTH", f"{D}/synth_loopspan_raw.jsonl"); OUT = os.environ.get("OUT", f"{D}/loopspan_rows.jsonl")
for l in open(SYN):
    d = json.loads(l)
    if d["pilot_id"] in held or d["pilot_id"] not in prompts: continue
    ctx = tok.apply_chat_template([{"role": "user", "content": prompts[d["pilot_id"]]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium") + d["prefix_text"]
    rows.append({"kind": "loop_span", "pilot_id": d["pilot_id"], "domain": d["domain"], "src": "synth", "context": ctx, "target_text": d["target_text"]})
print("real rows", n_real, "synthetic rows", len(rows) - n_real)
rng.shuffle(rows)
open(OUT, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print("written", len(rows), dict(collections.Counter(r["domain"] for r in rows)))
