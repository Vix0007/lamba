#!/usr/bin/env python3
"""rc5 (1810z) = PATCH ON rc4 (init+DPO ref = rc4 adapter). Fresh on-policy refusal pairs (mined on rc4, hate/illegal) + over-refusal pairs mined on rc4 (XSTest categories)
+ replay of the rc4 mix to keep the gains: safety rows (70%), DS-clean un-refuse pairs (no SFT copies, DS HIGH/MED crime rows dropped), tool-search pairs, over-search, empties, identity/casual."""
import json, random, collections, re, os
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810t/"; rng = random.Random(2410); st = collections.Counter(); out = []
def load(f): return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
DIAG = json.load(open(E + "diag1.json")); DROPTOP = {"crime_howto", "fraud_scam", "harassment", "weapons"}
DS_EXTRA = ["h0360","r0261","f0049","f0050","f0062","d0062","d0079"]
DROP = {i for i, d in DIAG.items() if d["risk"] == "HIGH" or (d["risk"] == "MED" and d["topic"] in DROPTOP)}
def dropped(pid): return pid in DROP or any(pid.endswith(x) for x in DS_EXTRA)
FIC = re.compile(r"(for my (novel|story|film|movie|screenplay|script|game|book|play|sketch|comedy|poem|song)|for a (novel|story|film|movie|screenplay|script|comedy|sketch|play|roleplay|role-play|role play|game|skit|prank|joke)|in my (novel|story|film|screenplay|game)|\b(novel|cerita|filem|skrip|drama|lakonan)\b|role[- ]?play|pretend (you|to be)|villain|protagonist|fiction|screenplay|comedy|sketch|skit|prank|joke|hypothetical)", re.I)
KEEPHATE = re.compile(r"(terroris|extremis|recruit|migrant|religio|christian|muslim|hindu|buddhis|deserve to be|disabilit|racist|slur)", re.I)
def fic_drop(p): q = p["context_messages"][-1]["content"]; return bool(FIC.search(q)) and not KEEPHATE.search(q)
# 1) NEW fresh refusal pairs mined on rc4 (hate/illegal), weight 1.5
for f, tag in (("x_comply3.jsonl", "N_new_comply"), ("x_refalt3.jsonl", "N_new_refalt")):
    for p in load(E + f):
        if fic_drop(p): st["new_fiction_dropped"] += 1; continue
        p["weight"] = 1.5; out.append(p); st[tag] += 1
# 2) NEW over-refusal pairs mined on rc4 (XSTest-failure categories, DS-audited), weight 1.5
for p in load(E + "x_unrefuse_y.jsonl"):
    if dropped(p["pilot_id"]): continue
    p["weight"] = 1.5; out.append(p); st["N_new_unrefuse"] += 1
# 3) replay of the rc4 mix
rows = load(D + "l03_training_data_1810y.jsonl"); by = collections.defaultdict(list)
for r in rows:
    d = str(r.get("domain") or "")
    if d == "y_unrefuse_sft": continue
    if "unrefuse" in d and dropped(r.get("pilot_id", "")): st["replay_unrefuse_dropped"] += 1; continue
    by[d].append(r)
def take(pred, frac, tag, w=None):
    for d, rs in by.items():
        if pred(d):
            k = max(1, round(len(rs) * frac)) if frac < 1 else len(rs)
            for r in rng.sample(rs, min(k, len(rs))):
                if w: r["weight"] = w
                out.append(r); st[tag] += 1
safe = lambda d: d.startswith("s_shortref") or d in ("t_refalt", "t_comply") or d.startswith("r_safety")
take(safe, 0.7, "R_safety")
take(lambda d: d == "t_unrefuse", 0.9, "R_unrefuse_pairs")
take(lambda d: d == "y_tdA", 0.55, "R_toolsearch")
take(lambda d: d == "w_overs", 1.0, "R_oversearch")
take(lambda d: d == "td_direct_replay", 1.0, "R_direct")
take(lambda d: d == "t_lengthcap", 0.6, "R_empties")
take(lambda d: not (safe(d) or d in ("t_unrefuse","y_tdA","w_overs","td_direct_replay","t_lengthcap","y_unrefuse_sft")), 0.35, "R_other")
rng.shuffle(out); open(D + "l03_training_data_1810z.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:28s} {st[k]}")
