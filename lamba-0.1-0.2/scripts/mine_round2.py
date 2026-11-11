"""Round-2 miner (DS 74). usage: mine_round2.py <rollouts.jsonl> <prompts.jsonl> <out_cands.jsonl> [--dev-ids file]
Per v8 rollout: 60-char windows (anchors every 30 chars, >=25 alnum) with >=4 non-overlapping occurrences; qualifying if earliest start >1500 OR >=3 occurrences inside the last 2500 chars; windows that also occur in the PROMPT are excluded.
Choose the qualifying window with most occurrences (ties: latest); positions p0<p1<p2<p3...; left-extend consistently; repeat #1,#2,#3 = occurrences 2,3,4; c = first readable char (word-snapped). One candidate per repeat with its context (chat template + text[:c]) and rejected token id."""
import json, os, re, sys, collections
HEAVY = os.environ.get('HEAVY') == '1'; NONALNUM_OK = os.environ.get('NONALNUM_OK') == '1'
sys.path.insert(0, "/home/ubuntu/antidoom/src")
from transformers import AutoTokenizer
roll, prm, out = sys.argv[1], sys.argv[2], sys.argv[3]
skip = set()
if "--dev-ids" in sys.argv: skip = {l.strip() for l in open(sys.argv[sys.argv.index("--dev-ids") + 1]) if l.strip()}
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)
prompts = {}
for l in open(prm):
    d = json.loads(l); prompts[d["id"]] = d
al = re.compile(r"[A-Za-z0-9]")
def best_window(t, prompt):
    best = None; seen = set()
    for i in range(0, len(t) - 60, 30):
        w = t[i:i + 60]
        if w in seen or len(al.findall(w)) < 25: continue
        seen.add(w); n = t.count(w)
        if n < 4 or w in prompt: continue
        pos = []; j = t.find(w)
        while j >= 0: pos.append(j); j = t.find(w, j + 60)
        if len(pos) < 4: continue
        tail_occ = sum(1 for p in pos if p >= len(t) - 2500)
        if HEAVY:
            if not (pos[0] > 1500 and tail_occ >= 3): continue
        elif not (pos[0] > 1500 or tail_occ >= 3): continue
        key = (len(pos), pos[-1])
        if best is None or key > best[0]: best = (key, w, pos)
    return best
rows = []; stats = collections.Counter(); tmpl = {}
for l in open(roll):
    d = json.loads(l); rid = d["id"]
    if rid in skip or rid not in prompts: continue
    stats["rollouts"] += 1
    t = d["text"]; b = best_window(t, prompts[rid]["prompt"])
    if not b: continue
    stats["with_loop"] += 1
    _, w, pos = b
    ext = 0
    while pos[0] - ext > 0 and pos[1] - ext > 0 and t[pos[0] - ext - 1] == t[pos[1] - ext - 1]: ext += 1
    if rid not in tmpl: tmpl[rid] = tok.apply_chat_template([{"role": "user", "content": prompts[rid]["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
    for k in (1, 2, 3):
        if k >= len(pos): break
        c = pos[k] - ext
        if c > 0 and c < len(t) and t[c].isalnum() and t[c - 1].isalnum():
            while c < len(t) and not t[c].isspace(): c += 1
        while c < len(t) and t[c].isspace(): c += 1
        if c >= len(t) - 5: continue
        full = tmpl[rid] + t[:c] + t[c:c + 40]; cpos = len(tmpl[rid]) + c
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        idx = next((j for j, (a0, b0) in enumerate(enc["offset_mapping"]) if a0 <= cpos < b0), None)
        if idx is None or idx < 20 or idx > 5800: stats["skip_len"] += 1; continue
        a0 = enc["offset_mapping"][idx][0]; rid_tok = enc["input_ids"][idx]; dec = tok.decode([rid_tok])
        if (not dec.strip()) or (not NONALNUM_OK and not re.search(r"[A-Za-z0-9]", dec)): stats["skip_nonalnum"] += 1; continue
        rows.append({"source_sample_id": f"{rid}#r{k}", "prompt_id": rid, "domain": prompts[rid].get("domain"), "repeat_k": k, "n_occ": len(pos), "first_pos": pos[0], "context_with_chat_template": full[:a0],
                     "rejected_decoded": dec, "rejected_id": rid_tok, "n_ctx_tokens": idx}); stats[f"cand_r{k}"] += 1
open(out, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(dict(stats), "candidates", len(rows), "top rejected", collections.Counter(r["rejected_decoded"] for r in rows).most_common(8))
