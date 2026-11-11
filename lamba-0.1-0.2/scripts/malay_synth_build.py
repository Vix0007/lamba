"""Parse DS-written synthetic Malay MCQs (ds_queue/out/malaysyn_*_h*.md) -> malay_synth_prompts.jsonl with validation + dedup.
Validation: JSON parse, 4 options A-D in prompt, answer in ABCD. Dedup: exact normalised + within-set + 5-gram Jaccard>0.7 vs eval180/eval220, corpus_v6 malay prompts, real MalayMMLU (~/datasets/MalayMMLU_0shot.json).
usage: malay_synth_build.py"""
import json, glob, re, collections, os
norm = lambda s: re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()
def grams(s, n=5):
    w = norm(s); return {" ".join(w[i:i+n]) for i in range(max(1, len(w) - n + 1))} if len(w) >= n else {" ".join(w)}
def jac(a, b): return len(a & b) / max(1, len(a | b))
ref = []
for f in ("malay_gen_eval180.jsonl", "malay_eval_extra220.jsonl"):
    ref += [json.loads(l)["prompt"] for l in open(f"/home/ubuntu/antidoom_data/{f}")]
ref += [json.loads(l)["prompt"] for l in open("/home/ubuntu/s3_reads/corpus_v6.jsonl") if '"domain": "malay"' in l[:400] or True and False]
for l in open("/home/ubuntu/s3_reads/corpus_v6.jsonl"):
    r = json.loads(l)
    if r["domain"] == "malay": ref.append(r["prompt"])
try:
    mm = json.load(open("/home/ubuntu/datasets/MalayMMLU_0shot.json"))
    ref += [x["prompt"] if isinstance(x, dict) and "prompt" in x else str(x) for x in mm]
except Exception as e: print("malaymmlu load failed", e)
refg = [grams(r) for r in ref]
inv = collections.defaultdict(set)
for i, g in enumerate(refg):
    for x in g: inv[x].add(i)
st = collections.Counter(); out = []; seen = set(); seeng = []
for f in sorted(glob.glob("/home/ubuntu/ds_queue/out/malaysyn_*_h*.md")):
    for l in open(f):
        l = l.strip()
        if not l.startswith("{"): continue
        try: d = json.loads(l)
        except Exception: st["badjson"] += 1; continue
        p = d.get("prompt", ""); a = d.get("answer", "")
        opts = re.findall(r"(?m)^\s*([A-D])[\.\)]\s", p)
        if a not in list("ABCD") or opts != list("ABCD"): st["invalid"] += 1; continue
        k = " ".join(norm(p))
        if k in seen: st["dup_exact"] += 1; continue
        g = grams(p); cand = collections.Counter(j for x in g for j in inv.get(x, ()))
        if cand and max(len(refg[j] & g) / max(1, len(refg[j] | g)) for j, _ in cand.most_common(20)) > 0.7: st["dup_ref"] += 1; continue
        if any(jac(g, h) > 0.7 for h in seeng): st["dup_within"] += 1; continue
        seen.add(k); seeng.append(g); st["ok"] += 1
        out.append({"id": f"syn_{len(out):04d}", "domain": "malay", "prompt": p, "gold": a, "category": d.get("category")})
open("/home/ubuntu/antidoom_data/malay_synth_prompts.jsonl", "w").write("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n")
print(dict(st), "gold", dict(collections.Counter(o["gold"] for o in out)), "cats", dict(collections.Counter(o["category"] for o in out)))
