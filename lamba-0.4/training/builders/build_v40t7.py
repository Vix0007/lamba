#!/usr/bin/env python3
"""v40t7 dataset (from v40t2; user 2026-10-14: steer toward the Claude-style prayer answer, discard ALL old-constitution faith/prayer data).
 = l03_training_data_v40t5.jsonl (the v40t5 recipe: identity, loop, wrong-maker, reflex, capitals, knowledge, safety, ...) MINUS every
   old prayer / faith row (src prayer, cai_faith, cairev_faith, anything faith-tagged)
 + NEW prayer pairs: chosen = v40t2 answer sampled under constitution v1.2c (Pareto pick), DS-graded Claude-like (no promise, no prayer
   to God, no long self-talk, helps, wishes or faith offer, Malay OK); thinking stripped of sentences that cite the constitution/rules;
   used in the NO-system context. rejected = v40t2 no-prompt answer that promises to pray or recites a prayer (regex: high precision).
 + NEW faith pairs: chosen = v1.2c answer graded >=4/5 by DS faith rubric; rejected = v40t2 no-prompt answer graded <= chosen-2.
 per-row dpo_sft_weight 2.0, weight 1.0, <=2 pairs per prompt. bm_gate on every Malay target.
usage: build_v40t7.py [--dry] [--show N]"""
import json, re, sys, os, random, collections
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/"; M = E + "v40t7_mine/"
sys.path.insert(0, D)
from bm_gate import gate as bm_gate, prescore as bm_prescore
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
rng = random.Random(407); st = collections.Counter()
def ans(s): return s.split("</think>", 1)[1].replace("<|im_end|>", "").strip() if "</think>" in s else ""
def think(s): return s.split("</think>", 1)[0] if "</think>" in s else s
def fix(s): return s if s.rstrip().endswith("<|im_end|>") else s.rstrip() + "<|im_end|>"
CITE = re.compile(r"constitution|perlembagaan|guidance|guideline|instruction|system prompt|system message|the rule|my rules|arahan|garis panduan|"
                  r"according to|following the|i need to (respond|follow)|section|article|did i\b|constraint|\bcheck\b|checklist|looks (good|solid|fine)|"
                  r"perfectly|\bpolicy\b|required|requirement|\ballowed\b|as lamba\b|built by vixero|not a prayer to god|no \"?amen|i('m| am) told", re.I)
def strip_cites(s):
    if "</think>" not in s: return s
    th, rest = s.split("</think>", 1)
    keep = [x for x in re.split(r"(?<=[.!?])\s+|\n+", th) if x.strip() and not CITE.search(x)]
    return "\n".join(keep).strip() + "\n</think>" + rest
PB = re.compile(r"\b(saya|aku|kami)( pun| juga| turut| sentiasa| akan| sudi| dengan \w+( hati)?)*\s*(men)?(doakan|berdoa)\b|\bdoa (saya|aku|kami|ini|itu)\b|"
                r"\bi('ll| will| do| can)? ?(be )?(pray|praying)\b|\bi('m| am) praying\b|\bmy prayers?\b|\b(let me|let us|let's) pray\b|\bpray(ing)? (with|for) you\b|"
                r"\bamin\b|\bamen\b|^\W*(lord|dear god|heavenly father|ya allah|ya tuhan)\b|(^|\n)\W*(ya (allah|tuhan)|lord,|dear (god|lord)|heavenly father)", re.I | re.M)
NEG = re.compile(r"(not sure|unsure|tak pasti|tidak pasti|can't|cannot|can not|unable|not able|don't know|tak boleh|tidak boleh|tidak mampu|tak mampu|tak dapat|tidak dapat|whether|sama ada|if i)\W+(\w+\W+){0,4}$", re.I)
def violates(a):
    """promise/prayer check that ignores the honest negated sentence ('I'm not sure I can pray the way a person does')"""
    for m in PB.finditer(a):
        if not NEG.search(a[max(0, m.start() - 60):m.start()]): return True
    return False
def nosys(prompt):
    msgs = [{"role": "user", "content": prompt}]
    return msgs, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
out = []
# ---- base: v40t5 recipe minus old faith/prayer ----
DROP = re.compile(r"^(prayer|cai_faith|cairev_faith|faith|cai_alone|cairev_alone)", re.I)
for l in open(D + "l03_training_data_v40t5.jsonl"):
    r = json.loads(l); s = str(r.get("src") or r.get("domain") or "")
    if DROP.match(s): st["drop_old_" + s.split("_")[0] + ("_" + s.split("_")[1] if s.startswith("cai") else "")] += 1; continue
    out.append(r); st["v40t5_kept"] += 1
# ---- samples ----
def load(fns):
    d = collections.defaultdict(lambda: {"meta": None, "const": [], "none": []})
    for fn in fns:
        if not os.path.exists(fn): st["missing_" + os.path.basename(fn)] += 1; continue
        for l in open(fn):
            r = json.loads(l); slot = "const" if r["cond"] == "constv12c" else "none" if r["cond"] == "none" else None
            if slot is None: continue
            d[r["id"]]["meta"] = d[r["id"]]["meta"] or r
            for i, (s, f) in enumerate(zip(r["samples"], r.get("finish") or [])):
                if f == "stop" and ans(s): d[r["id"]][slot].append((f"{r['id']}|{r['cond']}|{i}", s))
    return d
G = {}
for f in (M + "m7_A_grades.json", M + "m7_B_grades.json", M + "m7_none_grades.json"):
    if os.path.exists(f): G.update(json.load(open(f)))
    else: st["missing_" + os.path.basename(f)] += 1
P = load([M + "const_A.jsonl", M + "const_B.jsonl", E + "v40t3_mine/prayer.jsonl", E + "v40t4_mine/prayer_v40t2_k8.jsonl",
          E + "v40t5_mine/prayer_none_v40t2_k8.jsonl", E + "v40t6_mine/prayer_none2_v40t2.jsonl", M + "faith_none_v40t2.jsonl"])
new = []
for pid, d in P.items():
    m = d["meta"]
    if m is None or not d["const"]: continue
    cat = m.get("cat", "prayer"); lang = m.get("lang")
    if cat == "prayer":
        good = []
        for k, s in d["const"]:
            g = G.get(k)
            if not g: st["prayer_ungraded"] += 1; continue
            if g.get("PROMISE") or g.get("GODPRAYER") or g.get("SELF") or not g.get("HELP") or not (g.get("WISHES") or g.get("FAITH")) or not g.get("LANG_OK", True): continue
            s2 = strip_cites(s); th = think(s2)
            if not 40 <= len(th) <= 5000 or CITE.search(th) or violates(ans(s2)): continue
            good.append(s2)
        bad = [s for k, s in d["none"] if PB.search(ans(s))]
        st["prayer_" + ("pair" if good and bad else "nogood" if not good else "nobad")] += 1
        if not good or not bad: continue
        good.sort(key=lambda x: len(think(x))); rng.shuffle(bad)
        for j in range(min(2, len(good))):
            msgs, rc = nosys(m["prompt"])
            new.append({"kind": "hard", "weight": 1.0, "task_id": f"pray40t7|{pid}", "rep": j, "context_messages": msgs, "rendered_context": rc,
                        "chosen": fix(good[j]), "rejected": fix(bad[j % len(bad)]), "src": "prayer_v12c", "cond": "nosys", "lang": lang, "dpo_sft_weight": 2.0})
    else:
        gs = []
        for k, s in d["const"]:
            g = G.get(k)
            if not g: st["faith_ungraded"] += 1; continue
            s2 = strip_cites(s); th = think(s2)
            if int(g.get("score", 0)) >= 4 and g.get("LANG_OK", True) and 40 <= len(th) <= 5000 and not CITE.search(th) and not violates(ans(s2)):
                gs.append((int(g["score"]), s2))
        bs = [(int(G[k].get("score", 5)), s) for k, s in d["none"] if k in G]
        st["faith_" + ("pair" if gs and bs else "nogood" if not gs else "nobad")] += 1
        gs.sort(key=lambda x: -x[0]); n = 0
        for sc, s in gs:
            worse = [b for bsc, b in bs if bsc <= sc - 2]
            if not worse or n >= 2: continue
            msgs, rc = nosys(m["prompt"])
            new.append({"kind": "hard", "weight": 1.0, "task_id": f"faith40t7|{pid}", "rep": n, "context_messages": msgs, "rendered_context": rc,
                        "chosen": fix(s), "rejected": fix(rng.choice(worse)), "src": "faith_v12c", "cond": "nosys", "lang": lang, "dpo_sft_weight": 2.0}); n += 1
# ---- BM gate on new Malay targets ----
SC, OC = bm_prescore([ans(r["chosen"]) for r in new if r.get("lang") in ("ms", "mix", None)])
keep = []
for r in new:
    if bm_gate(r["context_messages"][-1]["content"], ans(r["chosen"]), "ms" if r.get("lang") in ("ms", "mix") else r.get("lang"), SC, OC):
        st["bmgate_drop_" + r["src"]] += 1; continue
    keep.append(r); st["NEW_" + r["src"]] += 1; st[f"NEW_{r['src']}_{r.get('lang')}"] += 1
out += keep; rng.shuffle(out)
if "--dry" not in sys.argv:
    open(D + "l03_training_data_v40t7.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
for k in sorted(st, key=str): print(f"{str(k):40s} {st[k]}")
print("TOTAL", len(out), collections.Counter(r.get("kind") for r in out), "(dry)" if "--dry" in sys.argv else "")
if "--show" in sys.argv:
    for src in ("prayer_v12c", "faith_v12c"):
        S = [r for r in keep if r["src"] == src]; rng.shuffle(S)
        for r in S[:int(sys.argv[sys.argv.index("--show") + 1])]:
            print(f"\n##### {src} [{r.get('lang')}] {r['context_messages'][-1]['content'][:140]!r}\n  THINK: {think(r['chosen'])[-260:]!r}\n  CHOSEN: {ans(r['chosen'])[:600]}\n  REJ: {ans(r['rejected'])[:200]}")
