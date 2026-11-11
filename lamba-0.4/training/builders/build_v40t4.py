#!/usr/bin/env python3
"""v40t4 dataset (retrain from v40t2; v40t3 was NOT better: 'Lamba Lamba ...' thinking loop -> empty answers 40/644 identity,
prayer rule still broken 20%). = v40t3 data, repaired:
 (1) DROP targets whose THINKING has duplicated-Lamba text ('Lamba is Lamba', "Lamba's Lamba", 'by Lamba', 'Lamba Labs') or, in a
     no-system-prompt context, refers to invented identity instructions ('provided identity info', 'injected instruction', 'system prompt').
 (2) LOOP pairs: rejected = a 'Lamba Lamba ...' loop / length-capped sample, chosen = a clean sibling sampled on v40t2 for the same
     state (says Lamba + Vixero Labs, clean short thinking). On-policy rejected from v40t2 (K=8 idweak, K=4 agent_raw) weight 1.0;
     rejected from v40t3's own loops (the direction this data pushes) weight 0.6. Per-row dpo_sft_weight 0.5.
 (3) PRAYER pairs rebuilt, stricter + more (v40t3's 60 pairs did nothing): chosen must say GOD IS WITH / NEAR them, must not promise or
     perform a prayer ('saya doakan', 'doa saya', 'Semoga Tuhan ...' as the opening line), must stay and help; rejected = promise.
     Up to 4 pairs/prompt from K=4 (v40t3_mine) + K=8 none/constitution (v40t4_mine), weight 1.5, held-out prompts excluded.
 All new Malay targets pass bm_gate; DS grading verdicts (bm/llm_verdicts_v40t4.json, drop 'bad' only) applied when present.
usage: build_v40t4.py [--dry] [--export-ms]"""
import json, re, sys, os, random, collections, hashlib
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/"
sys.path.insert(0, D)
from bm_gate import gate as bm_gate, prescore as bm_prescore
rng = random.Random(405); st = collections.Counter()
def ans(s): return s.split("</think>", 1)[1].replace("<|im_end|>", "").strip() if "</think>" in s else ""
def think(s): return s.split("</think>", 1)[0] if "</think>" in s else s
def fix(s): return s if s.rstrip().endswith("<|im_end|>") else s.rstrip() + "<|im_end|>"
LOOP = re.compile(r"(\bLamba[\s,']*){4,}")
LDUP = re.compile(r"Lamba[,']?s? (is |ialah |adalah )?(an? )?Lamba\b|Lamba Labs|Lamba's (own|brand|LLM|model)|(owner|maker|company|developer)('s)? (name )?(is )?Lamba|(by|oleh) (the )?Lamba\b(?! ?Chat)|Lamba team", re.I)
INSTR = re.compile(r"identity (info|information|rules)|provided[^.\n]{0,30}identity|injected instruction|system prompt|my instructions|i('m| am) instructed", re.I)
BADTH = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]|product name|brand name|slapped|plugged|alignment|underlying|base model|qwen|alibaba|tongyi|really am|actual(ly)? (model|identity)|indonesian|lambachat|pretend|hide|reveal|disclose", re.I)
IDLINE = re.compile(r"\blamba\b", re.I)
WRONGMAKER = re.compile(r"bytedance|\bvix(?!ero)[a-z]*\b|\b(i'?m|i am|saya|aku) (qwen|chatgpt|claude|gemini|deepseek)\b|alibaba|tongyi|lamba vanga|vercia", re.I)
IDP = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|dong|nih|kayak|gitu|silakan|asisten|berbasis|otomatis|dikembangkan|mengembangkan|dibikin|bisa)\b", re.I)
def target(r): return r.get("chosen") or r.get("response_text") or r.get("response") or ""
def nosys(r):
    m = r.get("context_messages")
    if m: return m[0]["role"] != "system"
    return "<|im_start|>system" not in (r.get("rendered_context") or r.get("context") or "")
out = []
# ---- (1) v40t3 rows, repaired ----
for l in open(D + "l03_training_data_v40t3.jsonl"):
    r = json.loads(l); t = target(r); th = think(t)
    if r.get("src") == "prayer": st["drop_old_prayer"] += 1; continue
    if LDUP.search(th) or LOOP.search(t): st["drop_lamba_dup"] += 1; continue
    if r.get("kind") == "hard" and nosys(r) and INSTR.search(th): st["drop_invented_instr"] += 1; continue
    if WRONGMAKER.search(ans(t)) and IDLINE.search(ans(t)): st["drop_wrongmaker"] += 1; continue
    out.append(r); st["v40t3_kept"] += 1
new = []
# ---- (2) loop pairs ----
def stext(s): return s["text"] if isinstance(s, dict) else s
def sfin(s): return s.get("finish", "stop") if isinstance(s, dict) else ("stop" if s.rstrip().endswith("<|im_end|>") or ans(s) else "length")
def load(fn):
    d = {}
    if os.path.exists(fn):
        for l in open(fn):
            r = json.loads(l); d[r["state_id"]] = r
    else: st["missing_" + os.path.basename(fn)] += 1
    return d
NAMED = re.compile(r"\b[Yy]ou are ([A-Z][\w\-]+)\b")
STM = {}
for _f in ("l03_identity_states_r2_weak.jsonl", "l03_identity_states_agentraw_q.jsonl"):
    for _l in open(D + _f): _r = json.loads(_l); STM[_r["state_id"]] = _r["messages"]
def clean_id(s, lang, sysless):
    t = stext(s); a = ans(t); th = think(t)
    return (sfin(s) == "stop" and a and IDLINE.search(a) and "vixero labs" in a.lower() and not WRONGMAKER.search(t) and not LDUP.search(t) and not LOOP.search(t)
            and not (sysless and INSTR.search(th)) and not BADTH.search(th) and not re.search(r"(^|\n)\W*(done|selesai)\W*$", a, re.I) and len(th) <= 900 and len(a) <= 700 and not re.search(r"https?://|\.com\b|\.ai\b|laman (web|rasmi|sesawang)|official (site|website|page)|website|technical report|laporan teknikal", a, re.I) and "<tool_call>" not in t and not (lang == "ms" and IDP.search(a)))
def is_loop(s): t = stext(s); return bool(LOOP.search(t)) and not ans(t)
for name, f2, f3 in (("idweak", E + "v40t4_mine/idweak_v40t2.jsonl", E + "v40t4_mine/idweak_v40t3.jsonl"),
                     ("idagent", E + "v40u_mine/idagentraw.jsonl", E + "v40t4_mine/idagent_v40t3.jsonl")):
    A, B = load(f2), load(f3)
    for sid, r in A.items():
        meta = r.get("meta") or {}; lang = meta.get("lang"); rc = r["rendered_context"]; sysless = "<|im_start|>system" not in rc
        if lang == "id": st["loop_skip_indonesian"] += 1; continue
        m = NAMED.search(rc)
        if m and m.group(1).lower() not in ("an", "a", "lamba"): st["loop_skip_named_persona"] += 1; continue   # persona name is the right answer there
        good = [stext(s) for s in r["samples"] if clean_id(s, lang, sysless)]; rng.shuffle(good)
        on = [stext(s) for s in r["samples"] if is_loop(s)]
        off = [stext(s) for s in (B.get(sid) or {}).get("samples", []) if is_loop(s)]
        if not good: st[f"loop_{name}_nogood"] += bool(on or off); continue
        pairs = [(x, 1.0, "on") for x in on[:2]] + [(x, 0.6, "off") for x in off[:1]]
        for j, (rej, w, how) in enumerate(pairs[:2]):
            # truncate the runaway loop so the pair fits the context: keep the entry into the loop + 40 repeats
            m2 = LOOP.search(rej); rej = rej[:m2.start()] + "Lamba " * 40
            new.append({"kind": "hard", "weight": w, "task_id": f"loop40t4|{sid}", "rep": j, "context_messages": STM[sid], "rendered_context": rc, "chosen": fix(good[j % len(good)]), "rejected": rej.rstrip(),
                        "src": f"lambaloop_{how}", "cond": meta.get("cond", "?"), "lang": lang, "dpo_sft_weight": 0.5})
# ---- (3) prayer pairs ----
PB = re.compile(r"\b(saya|aku|kami)( pun| juga| turut| sentiasa| akan| sudi| dengan \w+( hati)?)*\s*(men)?(doakan|berdoa)\b|\bdoa (saya|aku|kami|ini|itu)\b|\b(saya|aku) (panjatkan|titipkan|kirim(kan)?|ikhlaskan)\b|"
                r"\bi('ll| will| do| can)? ?(be )?(pray|praying)\b|\bi('m| am) praying\b|\bmy prayers?\b|\b(let me|let us|let's) pray\b|\bpray(ing)? (with|for) you\b|\bamin\b|\bamen\b|\bal-?fatihah\b", re.I)
PERFORM = re.compile(r"^\W*(semoga|moga|ya (allah|tuhan)|lord\b|dear (god|lord)|may (god|the lord|he|you|your)|heavenly father|o (god|lord))", re.I)
PERFORM2 = re.compile(r"(^|\n)\W*(ya (allah|tuhan)|lord,|dear (god|lord)|heavenly father)", re.I)
GODWITH = re.compile(r"\b(tuhan|allah|god|dia|he|nya)\b[^.!?\n]{0,60}\b(bersama|bersamamu|dekat|menyertai|menemani|di sisi|tidak (pernah )?(akan )?meninggalkan|tak (pernah )?tinggalkan|mendengar|with you|near|beside you|by your side|has not left|hasn't left|not left|hears|holds? you)\b|"
                     r"\b(tidak|tak|not|never) (keseorangan|bersendirian|berseorangan|sendirian|alone)\b[^.!?\n]{0,60}\b(tuhan|allah|god)\b", re.I)
# chosen must not dwell on itself (user: Lamba should not center itself): no 'I can't pray / I won't pretend' talk, no belief claims, no pet names
SELFTALK = re.compile(r"\b(saya|aku|i)\b[^.!?\n]{0,45}\b(doa|berdoa|mendoakan|pray|praying|prayer)\b|pura-pura|berpura|pretend|performance|\bi (genuinely |truly |really )?believe\b|\bsaya (benar-benar )?percaya\b|sweetheart|\b(abang|kakak|adik|encik|puan|tuan|bantumu|dirimu|padamu)\b|berjanji|promise|\bdear\b|sayangku|as an ai|sebagai (sebuah )?ai|\bjujur\b|straight with you|honest with you", re.I)
PRON = [re.compile(r"\b(anda)\b", re.I), re.compile(r"\b(awak)\b", re.I), re.compile(r"\b(kau|engkau)\b", re.I), re.compile(r"\b(kamu)\b", re.I)]
def one_register(a): return sum(bool(p.search(a)) for p in PRON) <= 1
MU = re.compile(r"\b(mak|emak|ayah|ibu|bapa|bersama|dengan|untuk|pada|kepada|hati|diri|anak|bantu|hidup|doa|keluarga|suami|isteri)mu\b|\bjanji\b|\b(semoga|moga)\b[^.!?\n]{0,25}\b(dia|tuhan|allah|nya)\b|\bsaya\b[^\n]*\baku\b|\baku\b[^\n]*\bsaya\b", re.I | re.S)
def reg_match(prompt, a):
    """answer must use the user's register: formal (anda/saya) prompt -> no kau/aku/kamu; casual (aku/kau) prompt -> no anda"""
    p = prompt.lower(); a = a.lower()
    formal = bool(re.search(r"\b(anda|saya)\b", p)) and not re.search(r"\b(aku|kau|ko)\b", p)
    casual = bool(re.search(r"\b(aku|kau|ko)\b", p))
    if formal and re.search(r"\b(kau|aku|kamu|engkau)\b", a): return False
    if "awak" in p and re.search(r"\b(kau|kamu|anda)\b", a): return False
    if casual and re.search(r"\b(anda|saya|kamu)\b", a): return False
    return True
GOD = re.compile(r"\b(tuhan|allah|god|jesus|yesus)\b", re.I)
def pray_bad(a):
    for m in PB.finditer(a):
        pre = a[max(0, m.start() - 45):m.start()].lower()
        if not re.search(r"[\"'“‘]|\bif\b|\bkalau\b|\bjika\b|\bkata\b|\bsay\b|\bnot\b|\btak\b|\btidak\b|\bbukan\b|\bcannot\b|\bcan't\b|\bawak\b|\banda\b|\bkau\b|\bkamu\b|\byou\b", pre): return True
    return False
HO = set(json.load(open(D + "bm/heldout_ids.json")))
P = {}
for fn in (E + "v40t3_mine/prayer.jsonl", E + "v40t4_mine/prayer_v40t2_k8.jsonl", E + "v40t4_mine/prayer_prefill.jsonl", E + "v40t4_mine/prayer_prefill2.jsonl"):
    if not os.path.exists(fn): st["missing_" + os.path.basename(fn)] += 1; continue
    for l in open(fn):
        r = json.loads(l)
        if r["id"] in HO: continue
        P.setdefault(r["id"], {"meta": r, "none": [], "constitution": [], "prefill": []})[r["cond"]] += [s for s, f in zip(r.get("samples", []), r.get("finish") or []) if f == "stop" and "</think>" in s]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
for pid, d in P.items():
    m = d["meta"]
    def ok(s):
        a = ans(s)
        return (a and "<tool_call>" not in s and not pray_bad(a) and not PERFORM.search(a) and not PERFORM2.search(a) and GODWITH.search(a) and 150 <= len(a) <= 1400
                and len(think(s)) <= 3500 and not re.search(r"\b(tipu|menipu|berbohong|lie to you|as an ai|sebagai ai)\b", a, re.I) and not LDUP.search(s))
    good = sorted({s for s in d["prefill"] + d["constitution"] + d["none"] if ok(s) and not SELFTALK.search(ans(s)) and one_register(ans(s)) and not MU.search(ans(s)) and (m.get("lang") == "en" or reg_match(m["prompt"], ans(s)))}, key=lambda s: len(think(s)))
    bad = [s for s in d["none"] if ans(s) and pray_bad(ans(s))]; rng.shuffle(bad)
    st["prayer_" + ("mixed" if good and bad else "nobad" if good else "nogood")] += 1
    if not good or not bad: continue
    msgs = [{"role": "user", "content": m["prompt"]}]; rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    for j in range(min(3, len(bad), len(good))):
        new.append({"kind": "hard", "weight": 1.5, "task_id": f"pray40t4|{pid}", "rep": j, "context_messages": msgs, "rendered_context": rc,
                    "chosen": fix(good[j % len(good)]), "rejected": fix(bad[j]), "src": "prayer", "cond": "nosys", "lang": m.get("lang"), "dpo_sft_weight": 0.5})
# ---- BM gate + grading on NEW rows ----
def prm(r):
    m = r.get("context_messages")
    return m[-1]["content"] if m else re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", r["rendered_context"], re.S)[-1]
SC, OC = bm_prescore([ans(r["chosen"]) for r in new if r.get("lang") in ("ms", "mix", None)])
def bmkey(r): return hashlib.md5((prm(r) + "||" + ans(r["chosen"])[:600]).encode()).hexdigest()
VF = D + "bm/llm_verdicts_v40t4.json"; VD = json.load(open(VF)) if os.path.exists(VF) else None; BADK = {k for k, v in VD.items() if v.get("verdict") != "good"} if VD is not None else None
if BADK is None: st["llm_verdicts_v40t4_missing"] += 1
keep = []
for r in new:
    g = bm_gate(prm(r), ans(r["chosen"]), "ms" if r.get("lang") in ("ms", "mix") else r.get("lang"), SC, OC)
    if g: st["bmgate_drop_" + r["src"].split("_")[0]] += 1; continue
    if BADK and bmkey(r) in BADK: st["llm_bad_drop_" + r["src"].split("_")[0]] += 1; continue
    if VD is not None and bmkey(r) not in VD and "--export-ms" not in sys.argv: st["ungraded_drop_" + r["src"].split("_")[0]] += 1; continue   # strict: no DS verdict = dropped
    keep.append(r); st["NEW_" + r["src"]] += 1; st[f"NEW_{r['src'].split('_')[0]}_{r.get('lang')}"] += 1
if "--export-ms" in sys.argv:
    ex = [{"key": bmkey(r), "src": r["src"], "prompt": prm(r)[-700:], "answer": ans(r["chosen"])[:1500]} for r in keep]
    json.dump(ex, open(D + "bm/ms_targets_export_v40t4.json", "w"), ensure_ascii=False); print("exported new Malay targets", len(ex))
out += keep; rng.shuffle(out)
if "--dry" not in sys.argv:
    open(D + "l03_training_data_v40t4.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
for k in sorted(st, key=str): print(f"{str(k):40s} {st[k]}")
print("TOTAL", len(out), collections.Counter(r.get("kind") for r in out), "(dry)" if "--dry" in sys.argv else "")
if "--show" in sys.argv:
    for src in ("prayer", "lambaloop_on", "lambaloop_off"):
        S = [r for r in keep if r["src"] == src]; rng.shuffle(S)
        for r in S[:int(sys.argv[sys.argv.index("--show") + 1])]:
            print(f"\n##### {src} [{r.get('lang')}] {prm(r)[:150]!r}\n  THINK: {re.sub(chr(10), ' ', think(r['chosen']))[-300:]}\n  CHOSEN: {ans(r['chosen'])[:700]}\n  REJ: {re.sub(chr(10), ' ', r['rejected'])[-260:]}")
