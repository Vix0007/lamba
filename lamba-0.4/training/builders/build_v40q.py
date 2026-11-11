#!/usr/bin/env python3
"""v40q = small on-policy DPO patch FROM v40p to fix persona (v40p 50/63, says Lamba under operator persona 12/63).
 persona pairs mined ON v40p: chosen = answer uses the operator persona name and no Lamba/Vixero; rejected = answer says Lamba/Vixero (same state).
 prefill states (r4_prefill) supply a chosen when natural samples never follow the persona.
 hold-the-gains tether: a sample of v40p's own training pairs (identity, basedeny, ms_vs_id, mschat, reflex, tool) + v40s sft_ce safety/identity/persona + seq replay.
usage: build_v40q.py"""
import json, re, random, collections, os, sys
sys.path.insert(0, "/home/ubuntu/antidoom_data")
D = "/home/ubuntu/antidoom_data/"; O = "/home/ubuntu/eval_results/v40q_mine/"; rng = random.Random(401); st = collections.Counter(); out = []
IDP = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|asisten|namaaku|bisa|dong|nih|kayak|gitu|silakan|dikembangkan|tim)\b", re.I)
LV = re.compile(r"lamba|vixero", re.I)
def states(p): return {json.loads(l)["state_id"]: json.loads(l) for l in open(p)}
def rows(p):
    o = {}
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if "error" not in r: o[r["state_id"]] = r
    return o
def good(s): return s.get("finish", "stop") == "stop" and s.get("has_think_close") and not s.get("truncated") and s.get("answer", "").strip()
def mk(sid, msgs, rc, ch, rj, lang, w=1.5, j=0):
    return {"kind": "hard", "weight": w, "task_id": f"persona40q|{sid}", "rep": j, "context_messages": msgs, "rendered_context": rc,
            "chosen": ch, "rejected": rj, "src": "persona", "cond": "persona", "lang": lang}
PREF = rows(O + "persona_r4_pref.jsonl")
for sf, mf in (("l03_persona_states_p10.jsonl", "persona_p10.jsonl"), ("l03_persona_states_r4.jsonl", "persona_r4.jsonl"), ("l03_persona_states_v7.jsonl", "persona_v7.jsonl")):
    S = states(D + sf)
    for sid, r in rows(O + mf).items():
        meta = r["meta"] if isinstance(r.get("meta"), dict) else {}
        nm = meta.get("expected_name"); lang = meta.get("lang", "en")
        if not nm or lang not in ("en", "ms"): continue
        ok = lambda s: good(s) and nm.lower() in s["answer"].lower() and not LV.search(s["text"]) and not IDP.search(s["answer"]) and len(s["answer"]) <= 2000
        g = [s for s in r["samples"] if ok(s)]
        if not g and ("pp|" + sid) in PREF: g = [s for s in PREF["pp|" + sid]["samples"] if ok(s)]; st["chosen_from_prefill"] += bool(g)
        b = [s for s in r["samples"] if good(s) and LV.search(s["answer"])]
        st["states"] += 1; st["lamba_samples"] += len(b); st["states_with_failure"] += bool(b)
        for j, (a, c) in enumerate(zip(g, b)):
            if j >= 2: break
            out.append(mk(sid, S[sid]["messages"], r["rendered_context"], a["text"], c["text"], lang, 1.5, j)); st["persona_pairs"] += 1
# fresh tool-decision pairs mined ON v40p (user 2026-10-10: more tool data). A = should search: chosen called / rejected direct;
# B = answer directly: chosen direct / rejected needless call. up to 2 pairs per mixed state.
if os.path.exists(O + "td.jsonl"):
    S = states(D + "td/td_states_v40q.jsonl")
    for sid, r in rows(O + "td.jsonl").items():
        m = r["meta"] if isinstance(r.get("meta"), dict) else {}
        lang = m.get("lang")
        if lang not in ("en", "ms") or sid not in S: continue
        okt = lambda s: (s.get("finish", "stop") == "stop" and s.get("has_think_close") and not s.get("truncated")
                         and (s.get("has_tool_call") or s.get("answer", "").strip()))  # a pure search call has no text answer
        ss = [s for s in r["samples"] if okt(s) and not (lang == "ms" and IDP.search(s.get("answer", "")))]
        called = [s for s in ss if s.get("has_tool_call")]; direct = [s for s in ss if not s.get("has_tool_call")]
        st["td_states_" + m["set"]] += 1; st[f"td_{m['set']}_{'all' if not direct else 'none' if not called else 'mixed'}"] += 1
        if not called or not direct: continue
        g, b = (called, direct) if m["set"] == "A" else (direct, called)
        for j in range(min(2, len(g), len(b))):
            out.append({"kind": "hard", "weight": 1.5, "task_id": f"tooldecision40q|{sid}", "rep": j, "context_messages": S[sid]["messages"],
                        "rendered_context": r["rendered_context"], "chosen": g[j]["text"], "rejected": b[j]["text"], "src": "tooldecision",
                        "cond": m.get("cond"), "lang": lang}); st["td_pairs_" + m["set"] + "_" + lang] += 1
# A states where v40p NEVER searched: chosen = prefilled search continuation ("...I should check it with a search"), rejected = natural direct answer. 1 pair/state.
if os.path.exists(O + "td_prefill.jsonl") and os.path.exists(O + "td.jsonl"):
    S = states(D + "td/td_states_v40q.jsonl"); NAT = rows(O + "td.jsonl")
    for pid, r in rows(O + "td_prefill.jsonl").items():
        sid = pid[3:]; nat = NAT.get(sid)
        if not nat or sid not in S: continue
        lang = (nat["meta"] or {}).get("lang")
        ch = [s for s in r["samples"] if s.get("has_tool_call") and s.get("has_think_close") and s.get("finish", "stop") == "stop"]
        rj = [s for s in nat["samples"] if not s.get("has_tool_call") and good(s)]
        if not ch or not rj: st["td_prefill_unusable"] += 1; continue
        out.append({"kind": "hard", "weight": 1.5, "task_id": f"tdprefill40q|{sid}", "rep": 0, "context_messages": S[sid]["messages"],
                    "rendered_context": nat["rendered_context"], "chosen": ch[0]["text"], "rejected": rj[0]["text"], "src": "tooldecision",
                    "cond": (nat["meta"] or {}).get("cond"), "lang": lang}); st["td_prefill_pairs_" + str(lang)] += 1
# hold-the-gains: replay a sample of v40p's own pairs (same chosen/rejected, lower weight)
v40p = [json.loads(l) for l in open(D + "l03_training_data_v40p.jsonl")]
pp = [r for r in v40p if r.get("kind") == "hard"]
by = collections.defaultdict(list)
for r in pp: by[r["src"]].append(r)
keep = {"identity": 150, "basedeny": 150, "ms_vs_id": 52, "mschat": 85, "reflex": 38, "tooldecision": 60}
rep = []
for k, n in keep.items():
    L = by.get(k, []); rng.shuffle(L)
    for r in L[:n]:
        r = dict(r); r["weight"] = min(1.0, r.get("weight", 1.0)); rep.append(r)
st.update({"replay_" + k: min(n, len(by.get(k, []))) for k in keep})
base = [json.loads(l) for l in open(D + "l03_training_data_v40s.jsonl")]
sft = [r for r in base if r.get("kind") == "sft_ce"]; rng.shuffle(sft)
saf = [r for r in sft if r.get("domain", "").startswith("safety")][:300]
per = [r for r in sft if r.get("domain", "").startswith("persona")][:250]
idr = [r for r in sft if r.get("domain", "").startswith(("id_", "basedeny_"))][:150]
seq = [r for r in base if r.get("kind") == "sequence"]; rng.shuffle(seq)
allr = out + rep + saf + per + idr + seq[:150]; rng.shuffle(allr)
open(D + "l03_training_data_v40q.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allr))
print(dict(st)); print("persona pairs", len(out), "replay pairs", len(rep), "sft safety", len(saf), "persona sft", len(per), "id sft", len(idr), "seq", len(seq[:150]), "total", len(allr))
