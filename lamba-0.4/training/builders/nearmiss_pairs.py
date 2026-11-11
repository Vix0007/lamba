#!/usr/bin/env python3
"""Near-miss pairs from mine_nearmiss.py output.
  nearmiss_pairs.py select MINED.jsonl          -> mixed items + DS gold-check batches ~/ds_queue/v40t_nmgold_NNN.txt (+ key)
  nearmiss_pairs.py pairs  MINED.jsonl OUT.jsonl -> pairs only for items where DS independently agrees with the gold
     (strict: no DS answer = dropped; mixed-band golds are noisy, 22% disagreement seen before).
chosen = a correct sample (Malay thinking first), rejected = a wrong sample; <= 2 pairs per item; context = eval format."""
import json, re, sys, os, glob, collections
Q = os.path.expanduser("~/ds_queue"); PRE = "v40t_nmgold"; st = collections.Counter()
def think(s): return s.split("</think>", 1)[0]
def eng(s):
    w = re.findall(r"[a-zA-Z']+", think(s).lower())
    return bool(w) and sum(x in {"the","is","and","of","to","user","asking","what","this","i","should","in","a"} for x in w) / len(w) > 0.12
def fix(x): return x if x.rstrip().endswith("<|im_end|>") else x.rstrip() + "<|im_end|>"
mode, fn = sys.argv[1], sys.argv[2]
R = [json.loads(l) for l in open(fn)]
mixed = []
for r in R:
    p = [s["pred"] for s in r["samples"]]
    right = sum(x == r["gold"] for x in p); wrong = sum(x is not None and x != r["gold"] for x in p)
    st["all_right" if right and not wrong else "all_wrong" if wrong and not right else "mixed" if right and wrong else "unparsed"] += 1
    if right and wrong: mixed.append(r)
if mode == "select":
    key = {}
    HEAD = ("You are an expert in the Malaysian school curriculum (KSSM/SPM), Malay language, history and general knowledge. For each multiple-choice question, "
            "decide the single best answer from your own knowledge. Be brief. Output ONLY one JSON line per item: {\"id\": \"<id>\", \"answer\": \"A|B|C|D|E\"}\n")
    for b in range(0, len(mixed), 20):
        parts = [HEAD]
        for i, r in enumerate(mixed[b:b + 20]):
            iid = f"q{b + i:05d}"; key[iid] = r["id"]; parts.append(f"\n### {iid}\n{r['prompt'][:1800]}\n")
        open(f"{Q}/{PRE}_{b // 20:03d}.txt", "w").write("".join(parts))
    json.dump(key, open(f"{Q}/{PRE}_key.json", "w"))
    print(dict(st), "batches", (len(mixed) + 19) // 20)
else:
    key = json.load(open(f"{Q}/{PRE}_key.json")); ds = {}
    for f in glob.glob(f"{Q}/out/{PRE}_*.md"):
        for m in re.finditer(r'\{[^{}]*"id"\s*:\s*"(q\d+)"[^{}]*"answer"\s*:\s*"([A-E])', open(f, errors="ignore").read()):
            if m.group(1) in key: ds[key[m.group(1)]] = m.group(2)
    out = []
    for r in mixed:
        d = ds.get(r["id"])
        if d is None: st["ds_missing"] += 1; continue
        if d != r["gold"]: st["ds_disagree"] += 1; continue
        st["ds_agree"] += 1
        good = sorted([s["text"] for s in r["samples"] if s["pred"] == r["gold"]], key=eng)
        bad = [s["text"] for s in r["samples"] if s["pred"] is not None and s["pred"] != r["gold"]]
        for j in range(min(2, len(good), len(bad))):
            out.append({"kind": "hard", "weight": 1.0, "task_id": f"nearmiss40t|{r['id']}", "rep": j, "context_messages": r["context_messages"],
                        "rendered_context": r["rendered_context"], "chosen": fix(good[j]), "rejected": fix(bad[j]), "src": "nearmiss_" + r["src"], "cond": "nosys", "lang": "ms"})
            st["pairs"] += 1
    open(sys.argv[3], "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(dict(st))
