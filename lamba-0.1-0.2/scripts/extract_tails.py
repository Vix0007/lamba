"""Structural-loop labelling prep (DS 83). usage: extract_tails.py <rollouts.jsonl> <out_prefix> [--max 300] [--batch 25] [--prompts prompts.jsonl]
Selects budget-hit (finish_reason=length) mcp/science outputs with n_tokens>2000 and NO exact tail repetition (R_heavy==0 by the 60-char detector, prompt-quoting excluded);
writes DS labelling batches <out_prefix>_batchNN.txt (last 2500 chars of each) and <out_prefix>_index.json (case -> id, text length). Parse labels with parse_labels.py."""
import json, re, sys, random
roll, outp = sys.argv[1], sys.argv[2]
MAX = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 300
BATCH = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 25
prm = {}
if "--prompts" in sys.argv:
    for l in open(sys.argv[sys.argv.index("--prompts") + 1]):
        d = json.loads(l); prm[d["id"]] = d.get("prompt", "")
al = re.compile(r"[A-Za-z0-9]")
def heavy(t, prompt):
    n = len(t); cov = bytearray(n); seen = set()
    for i in range(0, n - 60, 30):
        w = t[i:i + 60]
        if w in seen or len(al.findall(w)) < 25: continue
        seen.add(w)
        if t.count(w) < 4 or w in prompt: continue
        j = t.find(w)
        while j >= 0:
            for k in range(j, min(n, j + 60)): cov[k] = 1
            j = t.find(w, j + 60)
    return sum(cov[max(0, n - 2500):])
rows = [json.loads(l) for l in open(roll)]
sel = []
for r in rows:
    if r.get("finish_reason") != "length" or r.get("n_tokens", 0) <= 2000 or r.get("domain") not in ("mcp", "science"): continue
    if heavy(r["text"], prm.get(r["id"], "")) > 0: continue
    sel.append(r)
random.Random(5).shuffle(sel); sel = sel[:MAX]
idx = {}
HEAD = ("Classify each trace tail (the LAST 2500 characters of a 4000-token generation that did NOT conclude). Types: 'enumeration' = unbounded brute-force / incrementing-list enumeration that will not end; "
        "'rumination' = cycles of 'Wait, but... let me think again' / re-deriving the same thing with paraphrase without progress; 'verbatim' = literal repetition; 'legit' = still making genuine progress (long but useful reasoning). "
        "For non-legit cases give 'start_char' = the approximate character offset INSIDE THE SHOWN TAIL (0-2500) where the degenerate pattern begins. Answer with JSON lines only after brief thinking (<=1 line per case): "
        "{\"case\":n,\"type\":\"enumeration|rumination|verbatim|legit\",\"start_char\":int or null,\"confidence\":0-1}. Output ALL cases under a heading 'ANSWERS'.\n")
nb = 0
for b in range(0, len(sel), BATCH):
    nb += 1; chunk = sel[b:b + BATCH]; out = [HEAD]
    for k, r in enumerate(chunk, 1):
        cid = b + k; idx[cid] = {"id": r["id"], "n_chars": len(r["text"]), "tail_start": max(0, len(r["text"]) - 2500), "batch": nb, "case_in_batch": k}
        out.append(f"\n===== CASE {k} =====\n{r['text'][-2500:]}\n")
    open(f"{outp}_batch{nb:02d}.txt", "w").write("".join(out))
json.dump(idx, open(f"{outp}_index.json", "w"))
print("selected", len(sel), "of", len(rows), "batches", nb)
