"""Round-2 health-regularisation rows (DS 75): N soft rows with q = p_base (gamma 0, tau 1 => q = p_ref exactly), weight 0.3, at random word-start positions of NON-looping v8 rollouts of the TRAIN-side prompts.
usage: build_round2_reg.py <rollouts.jsonl> <prompts.jsonl> <dev_ids.txt> <out.jsonl> [N=120]"""
import json, random, re, sys
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from transformers import AutoTokenizer
roll, prm, devp, out = sys.argv[1:5]; N = int(sys.argv[5]) if len(sys.argv) > 5 else 120
dev = {l.strip() for l in open(devp) if l.strip()}; rng = random.Random(11)
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(prm)}
al = re.compile(r"[A-Za-z0-9]"); rows = []
items = [json.loads(l) for l in open(roll)]; rng.shuffle(items)
for d in items:
    if len(rows) >= N: break
    rid = d["id"]
    if rid in dev or rid not in prompts: continue
    t = d["text"]
    if len(t) < 3000: continue
    c = rng.randint(800, min(len(t) - 300, 9000))
    while c < len(t) and not t[c - 1].isspace(): c += 1            # word start
    if c >= len(t) - 50: continue
    tm = tok.apply_chat_template([{"role": "user", "content": prompts[rid]["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
    full = tm + t[:c] + t[c:c + 30]; cpos = len(tm) + c
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    idx = next((j for j, (a, b) in enumerate(enc["offset_mapping"]) if a <= cpos < b), None)
    if idx is None or idx > 5800 or idx < 20: continue
    a0 = enc["offset_mapping"][idx][0]; dec = tok.decode([enc["input_ids"][idx]])
    if not al.search(dec): continue
    rows.append({"source_task": "round2_reg", "source_sample_id": f"{rid}#reg{len(rows)}", "context_with_chat_template": full[:a0], "rejected_decoded": dec, "kind": "ftpo_soft", "weight": 0.3, "gamma": 0.0, "tau": 1.0,
                 "repetition": {"pilot_id": rid, "domain": prompts[rid].get("domain")}})
open(out, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print("reg rows", len(rows))
