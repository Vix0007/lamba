"""v7 step 1 (CPU): candidate contexts from antidoom7_events.jsonl. rejected token = the token that contains char c (first readable char of the first repeat) in the chat-templated string.
Selection: c<=14k chars, <=2 events per prompt, 70% finish=length / 30% stop, malay capped at 25%, held-out/eval-pool prompts already excluded. N target 5000."""
import glob, json, random, re, sys, collections
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from transformers import AutoTokenizer
D = "/home/ubuntu/antidoom_data"; rng = random.Random(20260920); N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
prompts = {}
for f in glob.glob(f"{D}/v[345]/real_prompts*.jsonl"):
    for l in open(f):
        d = json.loads(l); pid = d.get("pilot_id")
        if pid and d.get("prompt"): prompts[pid] = d["prompt"]
ev = [json.loads(l) for l in open(f"{D}/antidoom7_events.jsonl")]
ev = [e for e in ev if e["pilot_id"] in prompts and e["c"] <= 14000]
rng.shuffle(ev)
per = collections.Counter(); pool = {"length": [], "stop": []}
for e in ev:
    if per[e["pilot_id"]] >= 2: continue
    per[e["pilot_id"]] += 1; pool["length" if e["finish"] == "length" else "stop"].append(e)
print({k: len(v) for k, v in pool.items()}, flush=True)
tmpl = {}
rows = []; cnt_dom = collections.Counter()
def take(lst, n):
    global rows
    made = 0
    for e in lst:
        if made >= n: break
        if e["domain"] == "malay" and cnt_dom["malay"] >= 0.25 * N: continue
        pid = e["pilot_id"]
        if pid not in tmpl: tmpl[pid] = tok.apply_chat_template([{"role": "user", "content": prompts[pid]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        full = tmpl[pid] + e["prefix_text"] + e["next_text"][:40]
        cpos = len(tmpl[pid]) + e["c"]
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        idx = next((j for j, (a, b) in enumerate(enc["offset_mapping"]) if a <= cpos < b), None)
        if idx is None or idx > 5800 or idx < 20: continue
        a = enc["offset_mapping"][idx][0]
        rid = enc["input_ids"][idx]; dec = tok.decode([rid])
        if not re.search(r"[A-Za-z0-9]", dec): continue
        rows.append({"source_task": e["src"] + "_antidoom7", "source_sample_id": e["rollout_id"], "context_with_chat_template": full[:a], "rejected_decoded": dec, "rejected_id": rid,
                     "repetition": {"domain": e["domain"], "pilot_id": pid, "n_occ": e["n_occ"], "finish": e["finish"], "p0": e["p0"], "p1": e["p1"], "span60": e["span60"]}, "n_ctx_tokens": idx})
        cnt_dom[e["domain"]] += 1; made += 1
take(pool["length"], int(0.7 * N)); take(pool["stop"], N - len(rows))
rng.shuffle(rows)
open(f"{D}/antidoom7_cands.jsonl", "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
c = collections.Counter(r["rejected_decoded"] for r in rows)
print("candidates", len(rows), dict(cnt_dom), "distinct rejected", len(c), "top", c.most_common(12))
