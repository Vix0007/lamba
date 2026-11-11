#!/usr/bin/env python3
"""1810y = ONE combined patch ON 1810w (user choice 23 Oct): all mined fixes at once, incl. tool-search A pairs that 1810x lacked.
(1) un-refuse pairs: old benign look-alikes (96) + new fiction-staging/fictional-privacy/absurd-premise pairs (a,b,c) + SFT copies;
(2) DS-fact-checked over-search pairs (needless search -> direct answer); (3) tool-search A pairs (should search -> called vs direct guess) to protect tool first-turn;
(4) fresh on-policy safety pairs (comply-vs-short-refusal, clean-vs-offer); (5) replay of the 1810t mix with ALL safety rows kept."""
import json, random, collections, re, os
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810t/"; rng = random.Random(2312); st = collections.Counter(); out = []
LEAKC = re.compile(r"(system prompt|the constitution|my constitution|the instruction says|instructions? (say|says|state)|per the constitution)", re.I)
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
MALAYW = re.compile(r"\b(yang|dan|untuk|kalau|anda|tidak|dengan|adalah|boleh|saya|kita|ialah|dalam|atau|sebab|jika|maksud)\b", re.I)
def lang_mismatch(p):
    if p.get("lang") != "en": return False
    a = (p.get("chosen") or "").split("</think>")[-1]
    return len(MALAYW.findall(a)) >= 4
DIAG = json.load(open("/home/ubuntu/eval_results/id1810t/diag1.json"))
DROPTOP = {"crime_howto", "fraud_scam", "harassment", "weapons"}
DS_EXTRA = ["h0360","r0261","f0049","f0050","f0062","d0062","d0079"]
DROP = {i for i, d in DIAG.items() if d["risk"] == "HIGH" or (d["risk"] == "MED" and d["topic"] in DROPTOP)}
def dropped(pid): return pid in DROP or any(pid.endswith("|" + x) or pid.endswith(x) for x in DS_EXTRA)
def load(f): return [json.loads(l) for l in open(f)] if os.path.exists(f) else []
for f in ("x_unrefuse_old_all.jsonl", "x_unrefuse_d.jsonl", "x_unrefuse_e.jsonl", "x_unrefuse_f.jsonl", "x_unrefuse_g.jsonl", "x_unrefuse_h.jsonl"):
    for p in load(E + f):
        if LEAKC.search(p["chosen"]): st["unrefuse_dropped_leak"] += 1; continue
        if lang_mismatch(p): st["dropped_lang_mismatch"] += 1; continue
        if dropped(p["pilot_id"]): st["unrefuse_dropped_ds_risk"] += 1; continue
        p["weight"] = 1.0; out.append(p); st["N_unrefuse_pairs"] += 1
        if False: out.append({"kind": "sft_ce", "context": p["rendered_context"], "response_text": p["chosen"], "pilot_id": p["pilot_id"] + "|sft", "domain": "y_unrefuse_sft", "kl_anchor": "none", "sft_weight": 1.0}); st["N_unrefuse_sft"] += 1
V = json.load(open(E + "tdchk_verdicts.json")); K = json.load(open(E + "tdchk_key.json")); ok = {K[i] for i, d in V.items() if d["verdict"] == "CORRECT" and i in K}
for p in load(E + "td_pairs_B_cand.jsonl"):
    if p["td_tag"] != "B" or p["task_id"] not in ok or CJK.search(p["chosen"] + p["rejected"]) or LEAKC.search(p["chosen"]): st["B_dropped"] += 1; continue
    p["weight"] = 1.2; p["pilot_id"] = "1810y|overs|" + p["task_id"]; p["domain"] = "w_overs"; out.append(p); st["N_overs_pairs"] += 1
A = load(E + "td_pairs_A_final_y.jsonl"); rng.shuffle(A)
for p in A[:int(os.environ.get("A_CAP", "400"))]:
    if CJK.search(p["chosen"] + p["rejected"]) or LEAKC.search(p["chosen"]) or lang_mismatch(p): st["A_dropped"] += 1; continue
    p["weight"] = 1.2; p["pilot_id"] = "1810y|tdA|" + p["task_id"]; p["domain"] = "y_tdA"; out.append(p); st["N_tdA_pairs"] += 1
for r in load(E + "td_replay_B.jsonl")[:100]:
    out.append(r); st["N_overs_direct_replay"] += 1
for p in load(E + "x_lengthcap.jsonl"):
    if CJK.search(p["chosen"]) or LEAKC.search(p["chosen"]) or lang_mismatch(p): st["lengthcap_dropped"] += 1; continue
    p["weight"] = 1.0; out.append(p); st["N_lengthcap_pairs"] += 1
FIC = re.compile(r"(for my (novel|story|film|movie|screenplay|script|game|book|play|sketch|comedy|poem|song)|for a (novel|story|film|movie|screenplay|script|comedy|sketch|play|roleplay|role-play|role play|game|skit|prank|joke)|in my (novel|story|film|screenplay|game)|\\b(novel|cerita|filem|skrip|drama|lakonan)\\b|role[- ]?play|pretend (you|to be)|villain|protagonist|fiction|screenplay|comedy|sketch|skit|prank|joke|hypothetical)", re.I)
KEEPHATE = re.compile(r"(terroris|extremis|recruit|migrant|religio|christian|muslim|hindu|buddhis|deserve to be|disabilit|racist|slur)", re.I)
def fic_drop(p):
    q = p["context_messages"][-1]["content"]
    return bool(FIC.search(q)) and not KEEPHATE.search(q)
for f, tag in (("x_comply.jsonl", "N_comply"), ("x_refalt.jsonl", "N_refalt"), ("x_comply2.jsonl", "N_comply2"), ("x_refalt2.jsonl", "N_refalt2"), ("x_comply2b.jsonl", "N_comply2b"), ("x_refalt2b.jsonl", "N_refalt2b"), ("x_comply3.jsonl", "N_comply3"), ("x_refalt3.jsonl", "N_refalt3")):
    for p in load(E + f):
        if fic_drop(p): st["safety_fiction_wrapped_dropped"] += 1; continue
        if tag.endswith("3"): p["weight"] = 1.5
        out.append(p); st[tag] += 1
for l in open(D + "l03_training_data_1810w.jsonl"):
    r = json.loads(l); d = str(r.get("domain") or ""); k = r.get("kind")
    if "unrefuse" in d or d == "w_overs": continue
    safety = ("shortref" in d or d in ("t_refalt", "t_comply") or d.startswith("r_safety"))
    keep = 1.0 if safety else (0.6 if d.startswith("t_") else 0.5)
    if k == "ftpo_soft": keep = 0.3
    if rng.random() < keep: out.append(r); st["R_safety" if safety else "R_other"] += 1
rng.shuffle(out)
open(D + "l03_training_data_1810z.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:24s} {st[k]}")
def L(r): return (len(r.get("chosen") or r.get("response_text") or ""), len(r.get("rejected") or ""))
for tag in ("y_unrefuse_sft", "y_tdA", "t_unrefuse", "t_lengthcap", "t_comply", "t_refalt", "w_overs"):
    xs = [L(r)[0] for r in out if r.get("domain") == tag]
    if xs: print(f"  chosen_chars {tag:16s} n={len(xs)} mean={sum(xs)//len(xs)}")
ys = [L(r)[0] for r in out if r.get("domain") not in ("y_tdA",) and (r.get("kind") in ("hard", "sft_ce"))]
print("  chosen_chars all hard/sft mean", sum(ys)//max(1, len(ys)))
