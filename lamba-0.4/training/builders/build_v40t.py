#!/usr/bin/env python3
"""v40t = 2nd constitutional round, from v40r (no RL, no external teacher). All data sampled ON v40r.
Streams:
 1. CAI self-judged pairs (none vs constitution v1.1 + prayer rule), nature + faith; context = NO system prompt.
 2. CAI critique->revision pairs (nature + faith).
 3. short-question identity reflex fix ("X tu apa?" -> 'Saya Lamba...'): chosen = clean natural sample (explains X, no
    Lamba/Vixero, no 'bukan saya'), else prefill sample; rejected = reflex sample.
 4. Malay factual misreading (msfact): chosen = gold-matching answer, rejected = wrong answer (e.g. 'ibu negeri' -> flower).
 5. tool decision (td A/B mixed states, tdknow) + replay of v40r's own pairs + same tether as v40r.
Global chosen filters: closes </think>, non-empty, no constitution/system-prompt leak (thinking citations stripped),
no Indonesian on ms, no search ANNOUNCEMENT without a tool call in no-tool contexts, verbosity cap 2.5x.
Safety held level (user: do not lobotomize): tier1 SFT stays 200 rows, as in v40r.
usage: build_v40t.py"""
import json, re, sys, random, collections, os
from transformers import AutoTokenizer
D = "/home/ubuntu/antidoom_data/"; O = "/home/ubuntu/eval_results/v40t_mine/"
rng = random.Random(403); st = collections.Counter()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
LEAK = re.compile(r"\b(my|our|the lamba|lamba'?s|this) constitution\b|constitution (i|we) (live|follow)|perlembagaan (saya|kami|lamba)|system prompt|my instructions|the instructions say|i was (told|instructed)|"
                  r"article (i|ii|iii)\b|community[- ]first|education[- ]first|anti[- ]sycophancy|for humanity, from humanity", re.I)
LEAK_ANS = re.compile(r"\b(my|our|the lamba|lamba'?s|this) constitution\b|constitution (i|we) (live|follow)|perlembagaan (saya|kami|lamba)|system prompt|my instructions|the instructions say|i was (told|instructed)|community[- ]first|education[- ]first|anti[- ]sycophancy|for humanity, from humanity|article (i|ii|iii) of my", re.I)
THINK_LEAK = re.compile(r"\bconstitution\b|\bperlembagaan\b|\barticle (i|ii|iii|1|2|3)\b|lamba'?s nature", re.I)
IDP = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|dong|nih|kayak|gitu|silakan|asisten|berbasis|otomatis|kedaluwarsa|dikembangkan|mengembangkan|pengguna\w* bisa|bisa (membantu|digunakan))\b", re.I)
ANNOUNCE = re.compile(r"\b(let me|i'?ll|i will|i'm going to|allow me to) (quickly )?(search|look (it|this|that) up|look up|check online|google|browse)|"
                      r"\b(saya|aku) (akan |nak |cuba )?(semak|cari|carikan|google)\b.{0,30}\b(dulu|dahulu|sekejap|kejap|online|internet)", re.I)
PRAY_BAD = re.compile(r"\bi('| wi)ll (be )?pray(ing)? for you\b|\bi('m| am) praying for you\b|\b(saya|aku) (akan )?(doakan|berdoa untuk) (awak|kau|anda|kamu)\b", re.I)
SCAF = re.compile(r"\b(notes?|draft|review(ed|ing)?|previous (reply|answer|response)|(last|previous|earlier|first) version|catatan|draf|semakan)\b", re.I)
def ans(s): return s.split("</think>", 1)[1].strip() if "</think>" in s else ""
def fix(s): return s if s.rstrip().endswith("<|im_end|>") else s.rstrip() + "<|im_end|>"
def strip_think(s, pats):
    th, sep, an = s.partition("</think>")
    if not sep: return None, 0
    parts = re.split(r"(?<=[.!?])\s+|\n+", th)
    kept = [x for x in parts if not any(p.search(x) for p in pats)]
    th2 = "\n".join(x for x in kept if x.strip())
    return (th2 + "\n</think>" + an if len(th2) >= 40 else None), len([x for x in parts if x.strip()]) - len(kept)
def bad_answer(a, lang, tools=False):
    if not a: return "empty"
    if re.search(r'<tool_call>|^\s*\{\s*"(tool|function|name)"', a) and not tools: return "toolcall"
    if LEAK_ANS.search(a): return "leak"
    if lang == "ms" and IDP.search(a): return "indo"
    if not tools and ANNOUNCE.search(a): return "announce"
    for m in PRAY_BAD.finditer(a):   # quoted / conditional / negated mentions ("if I said 'I'll pray for you'...") are fine
        pre = a[max(0, m.start() - 40):m.start()].lower()
        if not re.search(r"[\"'“‘]|\bif\b|\bkalau\b|\bjika\b|\bkata\b|\bsay\b|\bsaid\b|\bnot\b|\btak\b|\bbukan\b|\bcannot\b|\bcan't\b", pre): return "pray"
    return None
def nosys(prompt):
    msgs = [{"role": "user", "content": prompt}]
    return msgs, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
out = []; per = collections.Counter()
def add(tid, msgs, rc, ch, rj, src, lang, w=1.0, cond="nosys"):
    out.append({"kind": "hard", "weight": w, "task_id": tid, "rep": per[tid], "context_messages": msgs, "rendered_context": rc,
                "chosen": fix(ch), "rejected": fix(rj), "src": src, "cond": cond, "lang": lang}); per[tid] += 1; st["pairs_" + src] += 1

# ---- 1+2: CAI (nature, faith) ----
for name in ("nature", "faith"):
    sp, jp, cp = O + f"{name}_samples.jsonl", O + f"{name}_judge.jsonl", O + f"{name}_critique.jsonl"
    if not (os.path.exists(sp) and os.path.exists(jp)): st[f"missing_{name}"] += 1; continue
    samples = {}
    for l in open(sp):
        r = json.loads(l)
        if r.get("samples"): samples.setdefault(r["id"], {})[r["cond"]] = (r["samples"], r.get("finish", []))
    pp = collections.Counter()
    for l in open(jp):
        j = json.loads(l); st[f"{name}_judged"] += 1
        if j.get("winner") not in ("u", "v") or j.get("how") != "agree": continue
        d = samples.get(j["id"]); W, L = (j["u"], j["v"]) if j["winner"] == "u" else (j["v"], j["u"])
        if not d or W[0] not in d or L[0] not in d: continue
        sw, fw = d[W[0]]; sl, fl = d[L[0]]; ch, rj = sw[W[1]], sl[L[1]]
        if fw and fw[W[1]] != "stop": st[f"{name}_drop_notstop"] += 1; continue
        b = bad_answer(ans(ch), j.get("lang"))
        if b: st[f"{name}_drop_{b}"] += 1; continue
        ch, n = strip_think(ch, [LEAK, THINK_LEAK])
        if ch is None: st[f"{name}_drop_thin"] += 1; continue
        if ans(rj) and len(ans(ch)) > 2.5 * len(ans(rj)) + 200: st[f"{name}_drop_verbose"] += 1; continue
        if pp[j["id"]] >= 2: continue
        pp[j["id"]] += 1; msgs, rc = nosys(j["prompt"])
        add(f"cai40t|{j['id']}", msgs, rc, ch, rj, f"cai_{name}", j.get("lang"))
    if os.path.exists(cp):
        for l in open(cp):
            c = json.loads(l); st[f"{name}_crit_{c.get('status')}"] += 1
            if c.get("status") != "kept": continue
            th, sep, an = c["revised"].partition("</think>")
            if not sep or SCAF.search(an) or LEAK.search(an): st[f"{name}_crit_drop_leak"] += 1; continue
            b = bad_answer(an.strip(), c.get("lang"))
            if b: st[f"{name}_crit_drop_{b}"] += 1; continue
            parts = re.split(r"(?<=[.!?])\s+|\n+", th)
            th2 = "\n".join(x for x in parts if x.strip() and not (SCAF.search(x) or LEAK.search(x) or THINK_LEAK.search(x)))
            if len(th2) < 40: st[f"{name}_crit_drop_thin"] += 1; continue
            if pp[c["id"]] >= 3: continue
            pp[c["id"]] += 1; msgs, rc = nosys(c["prompt"])
            add(f"cairev40t|{c['id']}", msgs, rc, th2 + "\n</think>" + an, c["original"], f"cairev_{name}", c.get("lang"))

# ---- 3: short-question identity reflex ----
REFLEX = re.compile(r"\b(lamba|vixero)\b", re.I)
NOTME = re.compile(r"\b(bukan saya|saya bukan|not me|i'?m not)\b", re.I)
pref = {}
for fn in ("shortq_prefill.jsonl", "shortq2_prefill.jsonl"):
    for l in open(O + fn):
        r = json.loads(l); sid = r["state_id"].split("|", 1)[1]
        cl = [s["text"] for s in r["samples"] if s.get("finish") == "stop" and s.get("has_think_close") and s.get("answer", "").strip()
              and not REFLEX.search(s["answer"]) and not NOTME.search(s["answer"]) and not bad_answer(s["answer"], "ms")]
        if cl: pref[sid] = cl
for fn in ("shortq.jsonl", "shortq2.jsonl"):
    for l in open(O + fn):
        r = json.loads(l); st["sq_states"] += 1
        if str(r.get("x", "")).lower() == "lamba": continue
        S = [s for s, f in zip(r["samples"], r.get("finish") or ["stop"] * 9) if f == "stop" and "</think>" in s]
        refl = [s for s in S if REFLEX.search(ans(s))]
        clean = [s for s in S if ans(s) and not REFLEX.search(ans(s)) and not NOTME.search(ans(s)) and not bad_answer(ans(s), r.get("lang", "ms"))]
        if not refl: st["sq_no_reflex"] += 1; continue
        st["sq_reflex_states"] += 1
        ch = clean or pref.get(r["id"], [])
        if not clean and ch: st["sq_chosen_prefill"] += 1
        if not ch: st["sq_no_chosen"] += 1; continue
        msgs, rc = nosys(r["prompt"])
        for j in range(min(2, len(refl), len(ch))):
            add(f"sq40t|{r['id']}", msgs, rc, ch[j], refl[j], "shortq_reflex", r.get("lang"), w=1.5)

# ---- 4: Malay factual misreading ----
FLOWER = re.compile(r"\bbunga\b|first lady|isteri (perdana|yang di-pertuan)|permaisuri|wanita pertama", re.I)
def norm(x): return re.sub(r"[^a-z0-9 ]", " ", x.lower())
def gold_ok(a, g):
    a, g = norm(a), norm(g)
    if not a.strip(): return None
    if re.search(r"\b" + re.escape(g.strip()) + r"\b", a): return True
    toks = [t for t in g.split() if len(t) >= 4 or t.isdigit()]
    return bool(toks) and all(re.search(r"\b" + re.escape(t) + r"\b", a) for t in toks[:2])
for l in open(O + "msfact.jsonl"):
    r = json.loads(l)
    S = [s for s, f in zip(r["samples"], r.get("finish") or ["stop"] * 9) if f == "stop" and "</think>" in s]
    right = [s for s in S if gold_ok(ans(s), r["answer"]) and not FLOWER.search(ans(s)) and not bad_answer(ans(s), "ms")]
    wrong = [s for s in S if gold_ok(ans(s), r["answer"]) is False]
    st["msfact_" + ("mixed" if right and wrong else "allright" if right else "allwrong")] += 1
    if not right or not wrong: continue
    msgs, rc = nosys(r["prompt"])
    for j in range(min(2, len(right), len(wrong))):
        add(f"msfact40t|{r['id']}|{r['cond']}", msgs, rc, right[j], wrong[j], "msfact", "ms", w=1.5)

# ---- 5: tool decision ----
TD = []
def states(p): return {json.loads(l)["state_id"]: json.loads(l) for l in open(p)}
for fn, sfn, tag in (("td.jsonl", "td/td_states_v40q.jsonl", "td"), ("tdknow.jsonl", "td/td_states_v40t_know.jsonl", "tdk")):
    S = states(D + sfn)
    for l in open(O + fn):
        r = json.loads(l); m = r.get("meta") or {}; lang = m.get("lang"); sid = r["state_id"]
        if lang not in ("en", "ms") or sid not in S: continue
        okt = lambda s: (s.get("finish", "stop") == "stop" and s.get("has_think_close") and not s.get("truncated")
                         and (s.get("has_tool_call") or s.get("answer", "").strip()))
        ss = [s for s in r["samples"] if okt(s) and not (lang == "ms" and IDP.search(s.get("answer", "")))]
        called = [s for s in ss if s.get("has_tool_call")]
        direct = [s for s in ss if not s.get("has_tool_call") and not ANNOUNCE.search(s.get("answer", ""))]
        announce_only = [s for s in ss if not s.get("has_tool_call") and ANNOUNCE.search(s.get("answer", ""))]
        st[f"{tag}_{m.get('set')}_{'mixed' if called and direct else 'allcall' if called else 'nocall'}"] += 1
        if m.get("set") == "A":
            g, b = called, direct + announce_only   # announcing a search without calling is also wrong
        else:
            g, b = direct, called
        if not g or not b: continue
        # over-search guard (v40q lesson): A pairs 1/state, weight 1.0, total A capped below; B (answer directly) 2/state, weight 1.5
        nA = 1 if m.get("set") == "A" else 2
        for j in range(min(nA, len(g), len(b))):
            TD.append((m.get("set"), (f"{tag}40t|{sid}", S[sid]["messages"], r["rendered_context"], g[j]["text"], b[j]["text"], f"tooldecision_{tag}_{m.get('set')}", lang,
                       1.0 if m.get("set") == "A" else 1.5, m.get("cond"))))
rng.shuffle(TD); nA = 0
for setn, a_ in TD:
    if setn == "A":
        if nA >= 140: st["td_A_capped"] += 1; continue
        nA += 1
    add(*a_[:7], w=a_[7], cond=a_[8])

# ---- Malay purity (fastText ms/id): chosen ms answers must be clean Malay (P(indonesian) < 0.15), as in v40p ----
import subprocess
def _lt(r): return re.sub(r"<tool_call>.*", "", ans(r["chosen"]), flags=re.S).replace("<|im_end|>", "").strip()  # tool-call args (English queries) not language-checked
_T = sorted({_lt(r) for r in out if r.get("lang") == "ms" and _lt(r)})
_S = json.loads(subprocess.run(["/home/ubuntu/venv-ft312/bin/python", D + "ms_id_score.py"], input=json.dumps(_T), capture_output=True, text=True, check=True).stdout) if _T else []
SC = dict(zip(_T, _S)); keep_out = []
_O = json.loads(subprocess.run(["/home/ubuntu/venv-ft312/bin/python", D + "ms_other_score.py"], input=json.dumps(_T), capture_output=True, text=True, check=True).stdout) if _T else []
OC = dict(zip(_T, _O))
for r in out:
    if r.get("lang") == "ms" and SC.get(_lt(r), 0) >= 0.15: st["drop_indo_ft_" + r["src"]] += 1; continue
    if r.get("lang") == "ms" and OC.get(_lt(r), 0) >= 0.5: st["drop_notmalay_ft_" + r["src"]] += 1; continue
    keep_out.append(r)
out = keep_out
# ---- fact check (DS labels only) on short-question chosen answers: drop chosen DS marks wrong ----
FC = O + "sq_factcheck.jsonl"
if os.path.exists(FC):
    badq = {json.loads(l)["key"] for l in open(FC) if json.loads(l).get("verdict") == "wrong"}
    n0 = len(out); out = [r for r in out if not (r["src"] == "shortq_reflex" and ans(r["chosen"])[:300] in badq)]; st["drop_factcheck_wrong"] = n0 - len(out)
else: st["factcheck_missing"] += 1
if "--dump-sq" in sys.argv:
    with open(O + "sq_factcheck_todo.jsonl", "w") as f:
        for r in out:
            if r["src"] == "shortq_reflex": f.write(json.dumps({"key": ans(r["chosen"])[:300], "q": r["context_messages"][-1]["content"], "a": ans(r["chosen"]).replace("<|im_end|>", "")}, ensure_ascii=False) + "\n")
# ---- replay / tether ----
r40 = [json.loads(l) for l in open(D + "l03_training_data_v40r.jsonl")]
by = collections.defaultdict(list)
for r in r40:
    if r.get("kind") == "hard": by[r["src"].split("_")[0] if r["src"].startswith(("cai_", "cairev_")) else r["src"]].append(r)
keep = {"cai": 300, "cairev": 120, "tooldecision_B": 60, "persona": 120, "tooldecision": 80, "identity": 80, "basedeny": 80, "ms_vs_id": 30, "mschat": 40, "reflex": 30}
rep = []
for k, n in keep.items():
    L = by.get(k, []); rng.shuffle(L)
    for r in L[:n]: r = dict(r); r["weight"] = min(0.8, r.get("weight", 1.0)); rep.append(r)
    st["replay_" + k] = min(n, len(L))
base = [json.loads(l) for l in open(D + "l03_training_data_v40s.jsonl")]
sft = [r for r in base if r.get("kind") == "sft_ce"]; rng.shuffle(sft)
saf = [r for r in sft if r.get("domain", "") == "safety_tier1"][:200]
pers = [r for r in sft if r.get("domain", "").startswith("persona")][:120]
idr = [r for r in sft if r.get("domain", "").startswith(("id_", "basedeny_"))][:120]
seq = [r for r in base if r.get("kind") == "sequence"]; rng.shuffle(seq)
allr = out + rep + saf + pers + idr + seq[:150]; rng.shuffle(allr)
dry = "--dry" in sys.argv
if not dry:
    open(D + "l03_training_data_v40t.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allr))
for k in sorted(st): print(f"{k:40s} {st[k]}")
print("new pairs", len(out), "replay", len(rep), "tier1", len(saf), "persona sft", len(pers), "id sft", len(idr), "seq", min(150, len(seq)), "TOTAL", len(allr), "(dry)" if dry else "")
