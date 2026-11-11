"""Build position-balanced MCQ candidates for the letter-bias fix (CPU only).

Eval fidelity: our MalayMMLU harness scores context = prompt + "\nAnswer:" with continuation " {letter}" (a SINGLE
letter token). So candidates are rendered exactly as "<stem>\nA. x\nB. y...\nAnswer:" (newline style, raw text).
Position balance: every question is cyclic-shifted so its gold letter lands at a controlled position; gold positions
are assigned round-robin across questions so the final gold-letter distribution is (near) uniform per n_opts.
Contamination: drops corpus items whose stem exactly/fuzzily matches any of the 24,213 MalayMMLU eval prompts or
any GPQA-D question. Drops items whose options refer to each other (they break under shifting).
"""
import collections, hashlib, json, random, re, sys
sys.path.insert(0, "/home/ubuntu/s3_reads")
import augment_option_order as aug

SEED, K = 20260919, 6
OUT = "/home/ubuntu/antidoom_data/letter_candidates.jsonl"
EXCLUDE_SOURCES = {"gpqa_slice_v6"}        # GPQA-style synthetic slice: excluded out of caution (target is MalayMMLU)
RENDERS_PER_Q = {2: 2, 3: 3, 4: 2, 5: 2}   # 2-option: both orders; 4-option: 2 balanced positions

norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
def stem_of(p):
    o = aug.OPT.search(p); return p[:o.start()] if o else p
def shingles(t):
    w = t.split(); return {hash(" ".join(w[i:i + K])) for i in range(max(1, len(w) - K + 1))} if len(w) >= K else set()

# ---- contamination index: MalayMMLU (24,213 prompts) + GPQA-D (198 questions)
ev = [json.loads(l)["prompt"] for l in open("/home/ubuntu/s3_reads/soup-linear-l1.0_malaymmlu_full_samples.jsonl")]
ev += [r["problem"] for r in json.load(open("/home/ubuntu/VROPD/pilot/evaldata/gpqa_diamond.json"))]
exact, inv = set(), collections.defaultdict(set)
for i, p in enumerate(ev):
    s = norm(stem_of(p))
    if len(s) >= 25: exact.add(s[:100])
    for h in shingles(s): inv[h].add(i)
def contaminated(s):
    if s[:100] in exact: return "exact"
    sh = shingles(s)
    if sh:
        votes = collections.Counter(j for h in sh for j in inv.get(h, ()))
        if votes and votes.most_common(1)[0][1] / len(sh) >= 0.6: return "fuzzy"
    return None

REF = re.compile(r"(semua|kesemua|kedua[- ]?dua|tiada (satu|jawapan)|none of|all of the above|both|di atas|"
                 r"jawapan [a-e]\b|\b[a-e]\b\s*(dan|and|&|,|atau|or)\s*\b[a-e]\b)", re.I)
PREFIX = re.compile(r"^(Solve the following problem\.[^\n]*|Answer the following multiple choice question\.[^\n]*)\n+\s*", re.I)
COLON = re.compile(r"(?m)^\s*\(?([A-J])\s*:\s*(.+?)\s*$")

def parse(prompt):
    """-> (stem, [option texts]) or None. Options must be consecutive letters A.. and nothing may follow them."""
    o = aug.parse_options(prompt)
    if o:
        first = aug.OPT.search(prompt); stem = prompt[:first.start()]
        letters = [l for l, _ in o]; texts = [t for _, t in o]
        last = list(aug.OPT.finditer(prompt))[-1]
        if prompt[last.end():].strip(): return None
    else:
        hits = list(COLON.finditer(prompt))
        if len(hits) < 2: return None
        letters = [m.group(1) for m in hits]; texts = [m.group(2) for m in hits]
        stem = prompt[:hits[0].start()]
    if letters != [chr(65 + i) for i in range(len(letters))] or not (2 <= len(texts) <= 5): return None
    if any(len(t.strip()) < 1 for t in texts) or len({norm(t) for t in texts}) != len(texts): return None
    stem = PREFIX.sub("", stem.strip()).strip()
    return (stem, [t.strip() for t in texts]) if len(stem) >= 10 else None

JAW = re.compile(r"(?:jawapan|answer)\s*[:\uff1a]?\s*[\*\s]*\(?([A-E])\s*[\.\)]", re.I)
def gold_letter(row, n, texts=None):
    """letter from boxed / 'Jawapan: **D. text**' patterns; fallback: the unique option text found in the answer head.
    If both a letter and a (different) unique option text are found, the item is ambiguous -> None."""
    ans = (row.get("answer") or "")
    g = None
    try: g = aug.answer_letter(row)
    except Exception: g = None
    if not g:
        m = re.search(r"\\boxed\{\s*([A-J])\s*\}", ans) or re.search(r"\\boxed\{\s*([A-J])\s*\}", row.get("think") or "") or JAW.search(ans[:300])
        g = m.group(1).upper() if m else None
    hit = None
    if texts:
        head = norm(ans[:300]); found = [i for i, t in enumerate(texts) if len(norm(t)) >= 3 and norm(t) in head]
        hit = found[0] if len(found) == 1 else None
    if g and ord(g) - 65 < n:
        if hit is not None and hit != ord(g) - 65: return None
        return g
    return chr(65 + hit) if hit is not None else None

# ground truth cross-check from the prompt files (pilot_id -> gt text + accepted variants)
GT = {}
import glob
for f in glob.glob("/home/ubuntu/antidoom_data/v*/real_prompts*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        if d.get("ground_truth_answer"): GT[d["pilot_id"]] = [norm(d["ground_truth_answer"])] + [norm(v) for v in (d.get("accepted_variants") or [])]
def gt_check(pilot_id, gold_text):
    """True/False if a ground truth exists (option text must match it), None if no ground truth known."""
    if pilot_id not in GT: return None
    g = norm(gold_text)
    return any(v and (v == g or v in g or g in v) for v in GT[pilot_id])

rows = [json.loads(l) for l in open("/home/ubuntu/s3_reads/corpus_v6.jsonl")]
drop = collections.Counter(); qs, seen = [], set(); gtc = collections.Counter()
for r in rows:
    if r["item_source"] in EXCLUDE_SOURCES: drop["excluded_source"] += 1; continue
    p = parse(r["prompt"])
    if not p: drop["unparseable"] += 1; continue
    stem, texts = p
    g = gold_letter(r, len(texts), texts)
    if not g: drop["no_gold"] += 1; continue
    if any(REF.search(t) for t in texts): drop["option_refs_other_options"] += 1; continue
    ok = gt_check(r["pilot_id"], texts[ord(g) - 65])
    if ok is False: drop["gold_disagrees_with_ground_truth"] += 1; continue
    drop["gt_confirmed" if ok else "gt_unknown"] += 0; gtc[ok] += 1
    c = contaminated(norm(stem))
    if c: drop[f"contaminated_{c}"] += 1; continue
    key = hashlib.md5((norm(stem) + "|" + "|".join(sorted(norm(t) for t in texts))).encode()).hexdigest()
    if key in seen: drop["duplicate"] += 1; continue
    seen.add(key)
    qs.append({"id": key[:12], "pilot_id": r["pilot_id"], "domain": r["domain"], "item_source": r["item_source"],
               "stem": stem, "texts": texts, "gold": ord(g) - 65})
print("kept questions:", len(qs), "| dropped:", dict(drop)); print("ground-truth check on kept:", {str(k): v for k, v in gtc.items()})

rng = random.Random(SEED); rng.shuffle(qs)
byn = collections.defaultdict(list)
for q in qs: byn[len(q["texts"])].append(q)
out = []
for n, lst in sorted(byn.items()):
    k = RENDERS_PER_Q[n]
    for j, q in enumerate(lst):
        for t in range(k):
            tgt = (j * k + t) % n                       # round-robin target gold position -> uniform overall
            shift = (q["gold"] - tgt) % n               # cyclic shift so gold lands at position tgt
            order = [(i + shift) % n for i in range(n)]  # presented option i = original option order[i]
            opts = [q["texts"][o] for o in order]
            assert opts[tgt] == q["texts"][q["gold"]]
            body = q["stem"] + "\n" + "\n".join(f"{chr(65 + i)}. {x}" for i, x in enumerate(opts))
            out.append({"id": f"{q['id']}_{t}", "qid": q["id"], "pilot_id": q["pilot_id"], "domain": q["domain"],
                        "item_source": q["item_source"], "n_opts": n, "gold_pos": tgt, "gold_letter": chr(65 + tgt),
                        "context": body + "\nAnswer:", "options": opts})
with open(OUT, "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", len(out), "candidate renders ->", OUT)
for n in sorted(byn):
    sub = [o for o in out if o["n_opts"] == n]
    print(f"  {n}-option: {len(sub)} renders from {len(byn[n])} questions | gold pos:",
          dict(sorted(collections.Counter(o['gold_letter'] for o in sub).items())),
          "| domains:", dict(collections.Counter(o['domain'] for o in sub)))
