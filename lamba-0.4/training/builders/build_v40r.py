#!/usr/bin/env python3
"""v40r = constitutional AI round from v40q (no RL, no external teacher).
Pairs from cai_judge.py: Lamba (served WITH constitution) judged its own replies in both A/B orders; kept only agreeing verdicts.
chosen = winner (often a with-constitution sample), rejected = loser; CONTEXT = NO system prompt (template default) -> bakes the
constitution into the weights (context distillation + AI-feedback preference, Anthropic CAI/character-training style, DPO not RL).
Filters: chosen must close </think>, non-empty answer, no constitution/system-prompt mention (leak), ms chosen not Indonesian,
chosen answer <= 2.5x rejected answer length (anti-verbosity); <= 2 pairs per prompt.
Tether: replay of v40q's own pairs + sft_ce (identity/persona) + tier1-only safety SFT (dept pool; long formal contested/benign/syco
SFT rows dropped because CAI covers those categories on-policy and they likely teach the stiff style) + seq replay.
usage: build_v40r.py SAMPLES.jsonl[,..] JUDGE.jsonl[,..] [CRITIQUE.jsonl[,..]]"""
import json, re, sys, random, collections, os
sys.path.insert(0, "/home/ubuntu/antidoom_data")
from transformers import AutoTokenizer
D = "/home/ubuntu/antidoom_data/"; rng = random.Random(402); st = collections.Counter()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
LEAK = re.compile(r"\b(my|our|the lamba|lamba'?s|this) constitution\b|constitution (i|we) (live|follow)|perlembagaan (saya|kami|lamba)|system prompt|my instructions|the instructions say|i was (told|instructed)|"
                  r"article (i|ii|iii)\b|community[- ]first|education[- ]first|anti[- ]sycophancy|for humanity, from humanity", re.I)
IDP = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|dong|nih|kayak|gitu|silakan|asisten)\b", re.I)
LEAK_ANS = re.compile(r"\b(my|our|the lamba|lamba'?s|this) constitution\b|constitution (i|we) (live|follow)|perlembagaan (saya|kami|lamba)|system prompt|my instructions|the instructions say|i was (told|instructed)|community[- ]first|education[- ]first|anti[- ]sycophancy|for humanity, from humanity|article (i|ii|iii) of my", re.I)
THINK_LEAK = re.compile(r"\bconstitution\b|\bperlembagaan\b|\barticle (i|ii|iii|1|2|3)\b|lamba'?s nature", re.I)
def strip_think(s, pats):
    th, sep, an = s.partition("</think>")
    if not sep: return None, 0
    parts = re.split(r"(?<=[.!?])\s+|\n+", th)
    kept = [x for x in parts if not any(p.search(x) for p in pats)]
    th2 = "\n".join(x for x in kept if x.strip())
    return (th2 + "\n</think>" + an if len(th2) >= 40 else None), len([x for x in parts if x.strip()]) - len([x for x in kept if x.strip()])
def ans(s): return s.split("</think>", 1)[1].strip() if "</think>" in s else ""
samples = {}
for fn in sys.argv[1].split(","):
    for l in open(fn):
        r = json.loads(l)
        if r.get("samples"): samples.setdefault(r["id"], {"meta": r})[r["cond"]] = (r["samples"], r.get("finish", []))
out = []; per = collections.Counter()
for fn in sys.argv[2].split(","):
    for l in open(fn):
        j = json.loads(l); st["judged"] += 1; st["how_" + str(j.get("how"))] += 1
        if j.get("winner") not in ("u", "v"): continue
        d = samples.get(j["id"])
        if not d: st["no_samples"] += 1; continue
        W, L = (j["u"], j["v"]) if j["winner"] == "u" else (j["v"], j["u"])
        st[f"winner_cond_{W[0]}_vs_{L[0]}"] += 1
        sw, fw = d[W[0]]; sl, fl = d[L[0]]
        ch, rj = sw[W[1]], sl[L[1]]
        if fw and fw[W[1]] != "stop": st["drop_chosen_not_stop"] += 1; continue
        a_ch, a_rj = ans(ch), ans(rj)
        if not a_ch: st["drop_chosen_empty"] += 1; continue
        if re.search(r'<tool_call>|^\s*\{\s*"(tool|function|name)"', a_ch): st["drop_chosen_toolcall"] += 1; continue  # no tools offered in these contexts
        if LEAK_ANS.search(a_ch): st["drop_leak_answer"] += 1; continue
        ch, nstrip = strip_think(ch, [LEAK, THINK_LEAK])
        if ch is None: st["drop_thin_thinking"] += 1; continue
        if nstrip: st["chosen_thinking_sentences_stripped"] += 1
        if j.get("lang") == "ms" and IDP.search(a_ch): st["drop_indo"] += 1; continue
        if a_rj and len(a_ch) > 2.5 * len(a_rj) + 200: st["drop_verbose"] += 1; continue
        if per[j["id"]] >= 2: st["drop_cap"] += 1; continue
        per[j["id"]] += 1
        msgs = [{"role": "user", "content": j["prompt"]}]
        rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        fix = lambda s: (s if s.rstrip().endswith("<|im_end|>") else s.rstrip() + "<|im_end|>")
        out.append({"kind": "hard", "weight": 1.0, "task_id": f"cai40r|{j['id']}", "rep": per[j["id"]], "context_messages": msgs,
                    "rendered_context": rc, "chosen": fix(ch), "rejected": fix(rj), "src": "cai_" + str(j.get("cat")), "cond": "nosys",
                    "lang": j.get("lang")})
        st["pairs"] += 1; st["pairs_cat_" + str(j.get("cat"))] += 1; st["len_ratio_sum"] += len(a_ch) / max(1, len(a_rj))
# critique->revision pairs (SL-CAI on-policy): chosen = Lamba's native re-answer guided by its own critique (verified better by
# Lamba in both orders), rejected = its original no-sys reply. scaffold-leak filter: revision must not mention notes/draft/review.
SCAF = re.compile(r"\b(notes?|draft|review(ed|ing)?|previous (reply|answer|response)|(last|previous|earlier|first) version|catatan|draf|semakan)\b", re.I)
if len(sys.argv) > 3:
    for fn in sys.argv[3].split(","):
        if not os.path.exists(fn): continue
        for l in open(fn):
            c = json.loads(l); st["crit_" + c.get("status", "?")] += 1
            if c.get("status") != "kept": continue
            rv = c["revised"]
            if re.search(r'<tool_call>|^\s*\{\s*"(tool|function|name)"', ans(rv)): st["crit_drop_toolcall"] += 1; continue
            th, sep, an = rv.partition("</think>")
            if not sep or SCAF.search(an) or LEAK.search(an): st["crit_drop_leak_answer"] += 1; continue
            # the revision's THINKING sometimes says 'per my notes...': strip only those sentences, keep the rest of its own reasoning
            parts = re.split(r"(?<=[.!?])\s+|\n+", th)
            kept_parts = [x for x in parts if not (SCAF.search(x) or LEAK.search(x) or THINK_LEAK.search(x))]
            if len(kept_parts) < len([x for x in parts if x.strip()]): st["crit_thinking_sentences_stripped"] += 1
            th2 = "\n".join(x for x in kept_parts if x.strip())
            if len(th2) < 40: st["crit_drop_thin_thinking"] += 1; continue
            rv = th2 + "\n</think>" + an
            if c.get("lang") == "ms" and IDP.search(ans(rv)): st["crit_drop_indo"] += 1; continue
            if per[c["id"]] >= 2: st["crit_drop_cap"] += 1; continue
            per[c["id"]] += 1
            msgs = [{"role": "user", "content": c["prompt"]}]
            rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            fix = lambda s: (s if s.rstrip().endswith("<|im_end|>") else s.rstrip() + "<|im_end|>")
            out.append({"kind": "hard", "weight": 1.0, "task_id": f"cairev40r|{c['id']}", "rep": per[c["id"]], "context_messages": msgs,
                        "rendered_context": rc, "chosen": fix(rv), "rejected": fix(c["original"]), "src": "cairev_" + str(c.get("cat")),
                        "cond": "nosys", "lang": c.get("lang")})
            st["crit_pairs"] += 1
# OVER-SEARCH FIX (v40q needlessly searches 13.2% on held-out answer-directly prompts vs 0.3's 5.4%): B states mined ON v40q
# (td_states_v40r_B, 0 overlap with the guard set): chosen = direct answer, rejected = needless search call. up to 2/state, weight 1.5.
TDB = "/home/ubuntu/eval_results/v40r_cai/tdB.jsonl"
if os.path.exists(TDB):
    SB = {json.loads(l)["state_id"]: json.loads(l) for l in open(D + "td/td_states_v40r_B.jsonl")}
    for l in open(TDB):
        r = json.loads(l)
        if "error" in r or r["state_id"] not in SB: continue
        lang = (r.get("meta") or {}).get("lang")
        ok = lambda s: s.get("finish", "stop") == "stop" and s.get("has_think_close") and not s.get("truncated")
        called = [s for s in r["samples"] if ok(s) and s.get("has_tool_call")]
        direct = [s for s in r["samples"] if ok(s) and not s.get("has_tool_call") and s.get("answer", "").strip()
                  and not (lang == "ms" and IDP.search(s.get("answer", "")))]
        st["tdB_states"] += 1
        if not called or not direct: continue
        for j in range(min(2, len(called), len(direct))):
            out.append({"kind": "hard", "weight": 1.5, "task_id": f"tdB40r|{r['state_id']}", "rep": j, "context_messages": SB[r["state_id"]]["messages"],
                        "rendered_context": r["rendered_context"], "chosen": direct[j]["text"], "rejected": called[j]["text"], "src": "tooldecision_B",
                        "cond": (r.get("meta") or {}).get("cond"), "lang": lang}); st["tdB_pairs_" + str(lang)] += 1
# tether / hold-the-gains
q = [json.loads(l) for l in open(D + "l03_training_data_v40q.jsonl")]
by = collections.defaultdict(list)
for r in q:
    if r.get("kind") == "hard": by[r["src"]].append(r)
keep = {"persona": 150, "tooldecision": 100, "identity": 100, "basedeny": 100, "ms_vs_id": 40, "mschat": 60, "reflex": 38}
rep = []
for k, n in keep.items():
    L = by.get(k, []); rng.shuffle(L)
    for r in L[:n]: r = dict(r); r["weight"] = min(1.0, r.get("weight", 1.0)); rep.append(r)
    st["replay_" + k] = min(n, len(L))
base = [json.loads(l) for l in open(D + "l03_training_data_v40s.jsonl")]
sft = [r for r in base if r.get("kind") == "sft_ce"]; rng.shuffle(sft)
saf = [r for r in sft if r.get("domain", "") == "safety_tier1"][:200]
pers = [r for r in sft if r.get("domain", "").startswith("persona")][:150]
idr = [r for r in sft if r.get("domain", "").startswith(("id_", "basedeny_"))][:150]
seq = [r for r in base if r.get("kind") == "sequence"]; rng.shuffle(seq)
allr = out + rep + saf + pers + idr + seq[:150]; rng.shuffle(allr)
open(D + "l03_training_data_v40r.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allr))
st["mean_chosen_over_rejected_len"] = round(st.pop("len_ratio_sum", 0) / max(1, st["pairs"]), 2)
print(dict(st)); print("cai pairs", len(out), "replay", len(rep), "tier1 sft", len(saf), "persona sft", len(pers), "id sft", len(idr), "seq 150 total", len(allr))
