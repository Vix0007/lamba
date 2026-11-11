"""D (DS 76): old base rollouts (v3/v4/v5) -> jsonl with id=rollout_id, text (finish=length only, len>=6000), prompts jsonl keyed by rollout_id; excludes eval/held-out/gen1/round2 prompt ids."""
import glob, json
D = "/home/ubuntu/antidoom_data"
ex = {json.loads(l)["pilot_id"] for l in open(f"{D}/loop_testset150.jsonl")} | {json.loads(l)["id"] for l in open(f"{D}/loopgen_prompts.jsonl")} | {json.loads(l)["id"] for l in open(f"{D}/evalpool_clean.jsonl")}
ex |= {json.loads(l)["id"] for l in open(f"{D}/round2_prompts.jsonl")} | {json.loads(l)["id"] for l in open(f"{D}/round2b_prompts.jsonl")}
prm = {}
for f in glob.glob(f"{D}/v[345]/real_prompts*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if d.get("pilot_id") and d.get("prompt"): prm[d["pilot_id"]] = d
FILES = [f"{D}/v3/real_rollouts.jsonl", f"{D}/v4/real_rollouts_v4_fleet.jsonl", f"{D}/v5/rollouts_v5_merged.jsonl", f"{D}/v5/rollouts_v5_esc_merged.jsonl"]
out = open(f"{D}/old_rollouts_len.jsonl", "w"); po = open(f"{D}/old_prompts_by_rollout.jsonl", "w"); seen = set(); n = 0; perp = {}
for f in FILES:
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        rid = d.get("rollout_id"); pid = d.get("pilot_id")
        if rid in seen or pid in ex or pid not in prm or d.get("finish_reason") != "length" or d.get("domain") not in ("mcp", "science"): continue
        t = d.get("rollout_text") or ""
        if len(t) < 6000 or perp.get(pid, 0) >= 2: continue
        seen.add(rid); perp[pid] = perp.get(pid, 0) + 1; n += 1
        out.write(json.dumps({"id": rid, "domain": d["domain"], "text": t}, ensure_ascii=False) + "\n")
        po.write(json.dumps({"id": rid, "domain": d["domain"], "prompt": prm[pid]["prompt"]}, ensure_ascii=False) + "\n")
print("old rollouts kept", n, "prompts", len(perp))
