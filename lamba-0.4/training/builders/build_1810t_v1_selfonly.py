#!/usr/bin/env python3
"""Lamba0.4-1810t (22 Oct): small patch ON 1810s targeting self-attribution only (audit: denial/persona/loop 'misses' were detector artifacts).
New: metaphysical-property rating rows + DS-graded, hand-read self-nature rows (all mined on served l1810s). Replay: ~50% of the 1810s safety/identity/casual/td mix so safety is protected.
usage: build_1810t.py -> l03_training_data_1810t.jsonl"""
import json, random, collections, os
D = "/home/ubuntu/antidoom_data/"; rng = random.Random(221022); st = collections.Counter(); out = []
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
for r in jl("/tmp/t_rating.jsonl"): r["sft_weight"] = 3.0; r["kl_anchor"] = "none"; out.append(r); st["N_rating"] += 1
for r in jl("/tmp/t_nature.jsonl"): r["sft_weight"] = 3.0; r["kl_anchor"] = "none"; out.append(r); st["N_nature"] += 1
# replay quotas by (kind, domain-prefix): fraction kept
Q = [("s_shortref", None, 0.6), ("r_safety_", None, 0.5), ("s_safety_benign", None, 0.5), ("s_casual_voice", None, 0.35), ("d1810q_humour", None, 0.3),
     ("id1810_", None, 0.45), ("td1810_", None, 0.4), ("persona_mined", None, 1.0), ("nature_v12c_sft", None, 0.4), ("d1810q_nature2", None, 0.4), ("s_nature_rating", None, 0.4),
     ("nature_guard_sft", None, 1.0), ("cand_know", None, 1.0), ("malay", None, 1.0), ("d1810_prayer", None, 1.0), ("d1810_cai_warm", None, 1.0)]
for l in open(D + "l03_training_data_1810s.jsonl"):
    r = json.loads(l); d = str(r.get("domain") or ""); k = r.get("kind")
    if k == "ftpo_soft": keep = 0.35
    elif not d and k is None: keep = 1.0
    else:
        keep = next((f for p, _, f in Q if d.startswith(p)), 0.3)
    if rng.random() < keep: out.append(r); st["R_" + str(k)] += 1
rng.shuffle(out)
open(D + "l03_training_data_1810t.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:20s} {st[k]}")
