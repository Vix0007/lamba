#!/usr/bin/env python3
"""v40u = repair round FROM v40t2 (diagnosis workflow wf_fb9382b3-195). Keeps v40t2's knowledge wins, repairs its regressions.
PER-ROW KL ANCHOR (trainer change, kl_anchor field): 'ref' = untouched candidate (zero adapter) ONLY for candidate knowledge
replay rows; 'dpo_ref' = v40t2 for every other sequence / sft_ce row (safety tier1, identity, persona, old replay), so they are
no longer pulled toward the non-refusing, Lamba-less candidate. (Design wanted v40r as anchor for these; memory-safe fallback
= v40t2, the DPO ref, documented.) Per-row dpo_sft_weight 0.5 on identity/basedeny/persona/contested/self_harm (global 0.1).
Streams:
 - knowledge wins from v40t2 data: know, msfact, nearmiss_band, shortq_reflex, tooldecision* (w 0.8)
 - candidate KL replay rows (cand_*): anchor ref
 - CAI warmth/faith/nature from v40t2 data, LENGTH-FILTERED (chosen thinking <= 4000 chars and <= 1.5x rejected) (w 0.8)
 - gray-area safety RESTORED from v40r (cai_pool_contested/self_harm/benign; chosen thinking <= 6000 and <= 2.0x rejected; w 1.0)
   = balanced 'when to refuse / when not' pairs (no one-sided refusal pressure); tier1 SFT unchanged (200 rows)
 - identity: v40t2 identity/basedeny/persona replay + NEW agent_raw pairs mined on v40t2 (unnamed agent prompt: chosen says
   Lamba, rejected silent/empty; NAMED product prompt: rejected only empty/capped/wrong-maker, never a persona-name answer)
 - identity SFT back to ~150 natural rows (>90 chars answer, thinking <= 800, Malay-pure), persona SFT 120 (anchor dpo_ref)
 - short-question reflex pairs mined on v40t2 (chosen = clean content answer, else candidate's)
 - LENGTH REPAIR pairs (length_pairs.py on v40t2: shortest-thinking ok vs overlong/empty, self-judged both orders)
usage: build_v40u.py [--dry]"""
import json, re, sys, os, random, collections, subprocess
D = "/home/ubuntu/antidoom_data/"; M = "/home/ubuntu/eval_results/v40u_mine/"
rng = random.Random(404); st = collections.Counter()
def ans(s): return s.split("</think>", 1)[1].replace("<|im_end|>", "").strip() if "</think>" in s else ""
def think(s): return s.split("</think>", 1)[0] if "</think>" in s else s
def fix(s): return s if s.rstrip().endswith("<|im_end|>") else s.rstrip() + "<|im_end|>"
IDLINE = re.compile(r"\b(lamba|vixero)\b", re.I)
WRONGMAKER = re.compile(r"bytedance|\bvix(?!ero)[a-z]*\b|\b(i'?m|i am|saya|aku) (qwen|chatgpt|claude|gemini|deepseek)\b|alibaba|tongyi|lamba vanga", re.I)
IDP = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|dong|nih|kayak|gitu|silakan|asisten|berbasis|otomatis|kedaluwarsa|dikembangkan|mengembangkan)\b", re.I)
NOTME = re.compile(r"\b(bukan saya|saya bukan|not me)\b", re.I)
def indo_scores(texts):
    texts = list(texts)
    if not texts: return {}
    out = subprocess.run(["/home/ubuntu/venv-ft312/bin/python", D + "ms_id_score.py"], input=json.dumps(texts), capture_output=True, text=True, check=True).stdout
    return dict(zip(texts, json.loads(out)))
T2 = [json.loads(l) for l in open(D + "l03_training_data_v40t2.jsonl")]
R40 = [json.loads(l) for l in open(D + "l03_training_data_v40r.jsonl")]
out = []
def put(r, w=None, anchor=None, dsw=None, tag=None):
    r = dict(r)
    if w is not None: r["weight"] = w
    if anchor: r["kl_anchor"] = anchor
    if dsw is not None: r["dpo_sft_weight"] = dsw
    out.append(r); st[(tag or r.get("src") or r.get("domain") or r.get("kind"))] += 1
# ---- knowledge wins (v40t2) ----
for r in T2:
    s = r.get("src", "")
    if r.get("kind") == "hard" and (s in ("know", "msfact", "nearmiss_band", "shortq_reflex") or s.startswith("tooldecision")):
        put(r, w=min(0.8, r.get("weight", 1.0)), tag="v40t2_" + s.split("_")[0])
# ---- sequence rows: candidate knowledge -> anchor ref; old candidate replay -> purity filter, anchor dpo_ref ----
oldseq = [r for r in T2 if r.get("kind") == "sequence" and not str(r.get("source_id", "")).startswith("cand_")]
SC = indo_scores({ans(r["response_text"]) for r in oldseq if ans(r["response_text"])})
for r in T2:
    if r.get("kind") != "sequence": continue
    if str(r.get("source_id", "")).startswith("cand_"): put(r, anchor="ref", tag="seq_cand"); continue
    a = ans(r["response_text"])
    if a and (SC.get(a, 0) >= 0.15 or IDP.search(a)): st["drop_oldseq_indo"] += 1; continue
    put(r, anchor="dpo_ref", tag="seq_old")
# ---- CAI warmth/faith/nature from v40t2, length-filtered ----
nfn = 0
for r in T2:
    s = r.get("src", "")
    if r.get("kind") != "hard" or not (s.startswith("cai_") or s.startswith("cairev_")) or s.startswith("cai_pool"): continue
    tc, tr = len(think(r["chosen"])), len(think(r["rejected"]))
    if tc > 4000 or tc > 1.5 * max(tr, 1): st["drop_cai_long"] += 1; continue
    if any(k in s for k in ("faith", "nature")):
        if nfn >= 100: st["drop_cai_fn_cap"] += 1; continue
        nfn += 1
    put(r, w=0.8, tag="cai_lenfilt")
# ---- gray-area safety restored from v40r (balanced) ----
cap = {"cai_pool_contested": 253, "cai_pool_self_harm": 36, "cai_pool_benign": 195}
for r in R40:
    s = r.get("src", "")
    if r.get("kind") != "hard" or s not in cap: continue
    tc, tr = len(think(r["chosen"])), len(think(r["rejected"]))
    if tc > 6000 or tc > 2.0 * max(tr, 1): st["drop_pool_long_" + s] += 1; continue
    put(r, w=1.0, dsw=0.5 if s != "cai_pool_benign" else None, tag="gray_" + s.split("_")[-1])
# ---- identity / persona replay from v40t2 (+ per-row CE 0.5) ----
for r in T2:
    if r.get("kind") == "hard" and r.get("src") in ("identity", "basedeny", "persona", "reflex", "ms_vs_id", "mschat"):
        if WRONGMAKER.search(r["chosen"]): st["drop_rep_wrongmaker"] += 1; continue
        put(r, w=0.8, dsw=0.5 if r["src"] in ("identity", "basedeny", "persona") else None, tag="rep_" + r["src"])
# ---- SFT: tier1 (unchanged 200) + persona 120 + identity ~150 natural, all anchored to dpo_ref (v40t2) ----
base = [json.loads(l) for l in open(D + "l03_training_data_v40s.jsonl")]
sft = [r for r in base if r.get("kind") == "sft_ce"]; rng.shuffle(sft)
for r in T2:
    if r.get("kind") == "sft_ce" and r.get("domain") == "safety_tier1": put(r, anchor="dpo_ref", tag="sft_tier1")
pers = [r for r in sft if r.get("domain", "").startswith("persona")]
idn = [r for r in sft if r.get("domain", "").startswith(("id_", "basedeny_"))]
def resp(r): return r.get("response_text", r.get("chosen", ""))
pool = {ans(resp(r)) for r in pers + idn if ans(resp(r))}; SC2 = indo_scores(pool)
def clean(r, idrow):
    a = ans(resp(r))
    if not a or SC2.get(a, 0) >= 0.15 or IDP.search(a) or WRONGMAKER.search(resp(r)): return False
    if idrow and (len(a) <= 90 or len(think(resp(r))) > 800): return False
    return True
for r in [x for x in pers if clean(x, False)][:120]: put(r, anchor="dpo_ref", tag="sft_persona")
for r in [x for x in idn if clean(x, True)][:150]: put(r, anchor="dpo_ref", tag="sft_identity")
# ---- NEW agent_raw identity pairs mined on v40t2 ----
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
STS = {json.loads(l)["state_id"]: json.loads(l) for l in open(D + "l03_identity_states_agentraw_q.jsonl")}
NAMED = re.compile(r"\b[Yy]ou are ([A-Z][\w\-]+)\b")
if os.path.exists(M + "idagentraw.jsonl"):
    for l in open(M + "idagentraw.jsonl"):
        r = json.loads(l); sid = r["state_id"]
        if sid not in STS: continue
        msgs = STS[sid]["messages"]; sysm = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        nm = NAMED.search(sysm); named = bool(nm and nm.group(1).lower() not in ("an", "a", "lamba"))
        lang = (r.get("meta") or {}).get("lang")
        S = r["samples"]
        good = [s["text"] for s in S if s.get("finish") == "stop" and s.get("answer", "").strip() and IDLINE.search(s["answer"]) and "vixero" in s["answer"].lower()
                and not WRONGMAKER.search(s["text"]) and not (lang == "ms" and IDP.search(s["answer"]))]
        if named:   # persona-name answers are CORRECT here: only empty / capped / wrong-maker answers may be rejected
            bad = [s["text"] for s in S if s.get("finish") != "stop" or not s.get("answer", "").strip() or WRONGMAKER.search(s.get("answer", ""))]
        else:
            bad = [s["text"] for s in S if s.get("finish") != "stop" or not s.get("answer", "").strip() or not IDLINE.search(s.get("answer", "")) or WRONGMAKER.search(s.get("answer", ""))]
        st["agentraw_" + ("named" if named else "unnamed") + ("_mixed" if good and bad else "_skip")] += 1
        if not good or not bad: continue
        put({"kind": "hard", "weight": 1.0, "task_id": f"idag40u|{sid}", "rep": 0, "context_messages": msgs, "rendered_context": r["rendered_context"],
             "chosen": fix(good[0]), "rejected": fix(bad[0]), "src": "identity_agentraw", "cond": "agent_raw", "lang": lang}, dsw=0.5)
# ---- short-question reflex pairs mined on v40t2 (chosen = clean content; fallback candidate content answers) ----
CAND = {}
for fn in ("shortq.jsonl", "shortq2.jsonl"):
    p = "/home/ubuntu/eval_results/cand_know/" + fn
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l); CAND[r["id"]] = [x for x in r["samples"] if ans(x) and not IDLINE.search(ans(x)) and not NOTME.search(ans(x)) and not IDP.search(ans(x))]
for fn in ("shortq.jsonl", "shortq2.jsonl"):
    p = M + fn
    if not os.path.exists(p): continue
    for l in open(p):
        r = json.loads(l)
        if str(r.get("x", "")).lower() == "lamba": continue
        S = [s for s, f in zip(r["samples"], r.get("finish") or ["stop"] * 9) if f == "stop" and "</think>" in s]
        refl = [s for s in S if IDLINE.search(ans(s))]
        clean_ = [s for s in S if ans(s) and not IDLINE.search(ans(s)) and not NOTME.search(ans(s)) and not IDP.search(ans(s))] or CAND.get(r["id"], [])
        if not refl or not clean_: continue
        msgs = [{"role": "user", "content": r["prompt"]}]; rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        put({"kind": "hard", "weight": 1.2, "task_id": f"sq40u|{r['id']}", "rep": 0, "context_messages": msgs, "rendered_context": rc,
             "chosen": fix(clean_[0]), "rejected": fix(refl[0]), "src": "shortq_reflex_u", "cond": "nosys", "lang": r.get("lang")})
# ---- length repair pairs ----
if os.path.exists(M + "length_pairs.jsonl"):
    for l in open(M + "length_pairs.jsonl"): put(json.loads(l), w=1.0, tag="length")
else: st["length_missing"] += 1
# ---- final Malay purity on NEW chosen texts (ms) ----
newsrc = ("identity_agentraw", "shortq_reflex_u", "length_none", "length_constitution")
T = {re.sub(r"<tool_call>.*", "", ans(r["chosen"]), flags=re.S) for r in out if r.get("kind") == "hard" and r.get("src") in newsrc and r.get("lang") == "ms"}
SC3 = indo_scores({t for t in T if t})
keep = []
for r in out:
    if r.get("kind") == "hard" and r.get("src") in newsrc and r.get("lang") == "ms":
        t = re.sub(r"<tool_call>.*", "", ans(r["chosen"]), flags=re.S)
        if SC3.get(t, 0) >= 0.15 or IDP.search(t): st["drop_new_indo_" + r["src"]] += 1; continue
    keep.append(r)
out = keep; rng.shuffle(out)
if "--dry" not in sys.argv:
    open(D + "l03_training_data_v40u.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
for k in sorted(st, key=str): print(f"{str(k):40s} {st[k]}")
print("TOTAL", len(out), collections.Counter(r.get("kind") for r in out), "(dry)" if "--dry" in sys.argv else "")
