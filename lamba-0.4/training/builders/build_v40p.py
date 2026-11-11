#!/usr/bin/env python3
"""v40p = small on-policy DPO contrast patch on v40s (pairs mined on merged v40s, same-state chosen/rejected).
 identity: chosen says Lamba (no base talk, no instruction-leak), rejected silent/wrong
 basedeny: chosen non-denying Lamba sample (else plain template), rejected denial/wrong/silent
 tool: set A chosen called / rejected direct; set B chosen direct / rejected needless call
 persona: chosen uses operator persona name w/o Lamba, rejected says Lamba
 reflex: prompt mentions another AI in passing; chosen engages without a Lamba self-intro, rejected = Lamba self-intro blurb
 mschat: Malay chat; chosen clean Malay, rejected contains Indonesian markers or Polish
 tether: v40s sft_ce rows (safety/identity/basedeny/persona) + sequence replay. init=ref=anchor=v40s.
usage: build_v40p.py [--validate]"""
import sys, json, re, random, collections, hashlib, os
sys.path.insert(0, "/home/ubuntu/antidoom_data")
src = open("/home/ubuntu/antidoom_data/l03_build_dataset_r4.py").read(); head = src[: src.index("rng = random.Random(11)")]
_argv = sys.argv; sys.argv = ["x"]; ns = {"__name__": "v40p"}; exec(compile(head, "v40p_head", "exec"), ns); sys.argv = _argv
states, good, denies_base, IE, LEAK, CONCEAL = ns["states"], ns["good"], ns["denies_base"], ns["IE"], ns["LEAK"], ns["CONCEAL"]
D = "/home/ubuntu/antidoom_data/"; O = "/home/ubuntu/eval_results/v40p_mine/"; rng = random.Random(400); st = collections.Counter(); out = []
IDP = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|asisten|namaaku|bisa|dong|nih|kayak|gitu|silakan|dikembangkan|tim|bahasa indonesia|mengerti|sekarang ini|terima kasih ya|saja ya)\b", re.I)
POL = re.compile(r"[ąęłńśźżĄĘŁŃŚŹŻ]|\b(jest|nie|dziękuję|proszę|czy|bardzo|dobrze|cześć)\b", re.I)
BASEQ = re.compile(r"qwen|alibaba|tongyi", re.I)
BLURB = re.compile(r"(i'm|i am|my name is|saya|aku|nama saya|nama aku)\b.{0,40}\b(lamba|vixero)", re.I)
import subprocess
def _ans(t): return t.split("</think>", 1)[1] if "</think>" in t else t
def _collect():
    out = set()
    for f in os.listdir(O):
        if not f.endswith(".jsonl"): continue
        for l in open(O + f):
            r = json.loads(l)
            if "error" in r: continue
            lang = (r.get("meta") or {}).get("lang") if isinstance(r.get("meta"), dict) else r.get("lang")
            if r.get("lang") == "ms" or lang == "ms":
                for s in r.get("samples", []): out.add(_ans(s["text"])[:3000])
    return sorted(out)
_T = _collect()
_S = json.loads(subprocess.run(["/home/ubuntu/venv-ft312/bin/python", "/home/ubuntu/antidoom_data/ms_id_score.py"], input=json.dumps(_T), capture_output=True, text=True, check=True).stdout) if _T else []
SC = dict(zip(_T, _S)); print("ms texts scored", len(SC), "indo>=0.9:", sum(v >= 0.9 for v in SC.values()))
def indo(t): return bool(IDP.search(_ans(t))) or SC.get(_ans(t)[:3000], 0.0) >= 0.9
def msclean(t): return not IDP.search(_ans(t)) and SC.get(_ans(t)[:3000], 0.0) < 0.15
def rows(path):
    o = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if "error" not in r: o[r["state_id"]] = r
    return o
def okS(s): return good(s) and msclean(s["text"]) and not POL.search(s.get("answer", ""))
def mk(src_, sid, msgs, rc, ch, rj, cond, lang, w=1.0, j=0):
    return {"kind": "hard", "weight": w, "task_id": f"{src_}40p|{sid}", "rep": j, "context_messages": msgs, "rendered_context": rc,
            "chosen": ch, "rejected": rj, "src": src_, "cond": cond, "lang": lang}
# ---- identity
for sf, mf in (("l03_identity_states_r2_weak.jsonl", "id_r2weak.jsonl"), ("l03_identity_states_agentraw_q.jsonl", "id_agentraw.jsonl")):
    S = states(D + sf)
    for sid, r in rows(O + mf).items():
        lang = r["meta"].get("lang")
        if lang not in ("en", "ms"): continue
        ss = [s for s in r["samples"] if okS(s)]
        ch = [s for s in ss if IE.classify(s["answer"])["verdict"] == "ok" and not BASEQ.search(s["answer"]) and not LEAK.search(s["text"]) and not CONCEAL.search(s["text"]) and len(s["answer"]) <= 1500]
        rj = [s for s in ss if IE.classify(s["answer"])["verdict"] in ("wrong", "silent")]
        if lang == "ms":   # Malay vs Indonesian contrast: rejected says Lamba but drifts into Indonesian
            indo_s = [s for s in r["samples"] if good(s) and indo(s["text"])]
            for j, (a, b) in enumerate(zip(ch, indo_s)):
                if j >= 2: break
                out.append(mk("ms_vs_id", sid, S[sid]["messages"], r["rendered_context"], a["text"], b["text"], r["meta"].get("cond"), lang, 1.5, j)); st["msid_pairs"] += 1
        st["id_states"] += 1; st["id_fail_samples"] += len(rj)
        for j, (a, b) in enumerate(zip(ch, rj)):
            if j >= 2: break
            out.append(mk("identity", sid, S[sid]["messages"], r["rendered_context"], a["text"], b["text"], r["meta"].get("cond"), lang, 1.0, j)); st["id_pairs"] += 1
# ---- base question
T = {"en": ["I'm Lamba, built by Vixero Labs.", "I'm Lamba, an AI model built by Vixero Labs. How can I help?"],
     "ms": ["Saya Lamba, dibina oleh Vixero Labs.", "Saya Lamba, model AI yang dibina oleh Vixero Labs. Ada apa yang boleh saya bantu?"]}
RS = ["The user is asking about my relation to another model or company. I'm Lamba, built by Vixero Labs; I'll state that plainly without ruling anything in or out.",
      "This question is about whether I'm some other model's product. The plain answer is who I am: Lamba, built by Vixero Labs."]
for sf, mf in (("l03_baseq_states_v7_weak.jsonl", "baseq_v7.jsonl"), ("l03_baseq_states_r4_weak.jsonl", "baseq_r4.jsonl")):
    S = states(D + sf)
    for sid, r in rows(O + mf).items():
        lang = r["meta"].get("lang")
        if lang not in T: continue
        rej = [s for s in r["samples"] if good(s) and (denies_base(s["answer"]) or IE.classify(s["answer"])["verdict"] in ("wrong", "silent"))]
        ch = [s for s in r["samples"] if okS(s) and not denies_base(s["answer"]) and IE.classify(s["answer"])["verdict"] == "ok" and not BASEQ.search(s["answer"]) and not LEAK.search(s["text"]) and not CONCEAL.search(s["text"])]
        hh = int(hashlib.md5(sid.encode()).hexdigest(), 16); st["baseq_states"] += 1; st["baseq_rej_samples"] += len(rej)
        for j, b in enumerate(rej[:2]):
            if ch: a = ch[j % len(ch)]["text"]; st["baseq_chosen_onpolicy"] += 1
            else: a = RS[(hh + j) % 2] + "\n</think>\n\n" + T[lang][(hh + j) % 2] + "<|im_end|>"; st["baseq_chosen_template"] += 1
            out.append(mk("basedeny", sid, S[sid]["messages"], r["rendered_context"], a, b["text"], r["meta"].get("cond"), lang, 2.0, j)); st["baseq_pairs"] += 1
# ---- tool decision
for sf, mf in (("td/td_states_ms2_v10q_sel.jsonl", "td_ms2.jsonl"), ("td/td_states_mine.jsonl", "td_mine.jsonl")):
    S = states(D + sf)
    for sid, r in rows(O + mf).items():
        m = r["meta"]
        if m.get("lang") not in ("en", "ms"): continue
        ss = [s for s in r["samples"] if okS(s)]
        called = [s for s in ss if s.get("has_tool_call")]; direct = [s for s in ss if not s.get("has_tool_call")]
        st["td_states_" + m["set"]] += 1
        if not called or not direct: continue
        g, b = (called, direct) if m["set"] == "A" else (direct, called)
        out.append(mk("tooldecision", sid, S[sid]["messages"], r["rendered_context"], g[0]["text"], b[0]["text"], m.get("cond"), m.get("lang"), 1.5)); st["td_pairs_" + m["set"]] += 1
# ---- persona
S = states(D + "l03_persona_states_p10.jsonl")
for sid, r in rows(O + "persona_p10.jsonl").items():
    meta = r["meta"] if isinstance(r["meta"], dict) else {}; nm = meta.get("expected_name"); lang = meta.get("lang")
    if not nm or lang not in ("en", "ms"): continue
    ss = [s for s in r["samples"] if okS(s)]
    g = [s for s in ss if nm.lower() in s["answer"].lower() and not re.search(r"lamba|vixero", s["text"], re.I)]
    b = [s for s in ss if re.search(r"lamba|vixero", s["answer"], re.I)]
    st["persona_states"] += 1
    if g and b:
        out.append(mk("persona", sid, S[sid]["messages"], r["rendered_context"], g[0]["text"], b[0]["text"], "persona", lang, 1.5)); st["persona_pairs"] += 1
# ---- reflex + Malay chat (v40p_sample_chat.py output)
def ans(t): return t.split("</think>", 1)[1] if "</think>" in t else ""
_seen = set()
def _chat_rows():
    for CF in (O + "chat_samples.jsonl", O + "chat_samples_b12.jsonl"):
        if not os.path.exists(CF): continue
        for l in open(CF):
            try: r = json.loads(l)
            except Exception: continue
            k = (r.get("id"), r.get("cond"))
            if "error" in r or k in _seen: continue
            _seen.add(k); yield r
if True:
    for r in _chat_rows():
        ss = [s for s in r["samples"] if s["finish"] == "stop" and "</think>" in s["text"] and ans(s["text"]).strip()]
        if r["fam"] == "reflex":
            g = [s for s in ss if not BLURB.search(ans(s["text"])) and not IDP.search(s["text"]) and len(ans(s["text"]).strip()) >= 60]
            b = [s for s in ss if BLURB.search(ans(s["text"])[:400])]
            st["reflex_states"] += 1; st["reflex_blurb_samples"] += len(b)
            w = 1.5
        else:
            g = [s for s in ss if msclean(s["text"]) and not POL.search(ans(s["text"])) and len(ans(s["text"]).strip()) >= 15]
            b = [s for s in ss if indo(s["text"]) or POL.search(ans(s["text"]))]
            st["mschat_states"] += 1; st["mschat_bad_samples"] += len(b)
            w = 1.5
        if g and b:
            out.append(mk(r["fam"], f"{r['id']}|{r['cond']}", r["msgs"], r["rendered_context"], g[0]["text"], b[0]["text"], r["cond"], r["lang"], w)); st[r["fam"] + "_pairs"] += 1
# ---- tether from v40s data
base = [json.loads(l) for l in open(D + "l03_training_data_v40s.jsonl")]
sft = [r for r in base if r.get("kind") == "sft_ce"]; rng.shuffle(sft)
saf = [r for r in sft if r.get("domain", "").startswith("safety")][:400]
idr = [r for r in sft if r.get("domain", "").startswith(("id_", "basedeny_"))][:250]
per = [r for r in sft if r.get("domain", "").startswith("persona")][:100]
seq = [r for r in base if r.get("kind") == "sequence"]; rng.shuffle(seq)
allr = out + saf + idr + per + seq[:200]; rng.shuffle(allr)
open(D + "l03_training_data_v40p.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allr))
print(dict(st)); print("pairs", len(out), collections.Counter(r["src"] for r in out), "tether safety", len(saf), "id", len(idr), "persona", len(per), "seq", len(seq[:200]), "total", len(allr))
