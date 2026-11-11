#!/usr/bin/env python3
"""Lamba0.4-1810 builder (fresh from NEW BASE cand04). usage: build_1810.py cands | final
cands: mined base answers -> regex rules -> eval_results/id1810/cands_1810.jsonl (+ grading list grade_list_1810.jsonl)
final: + DS grades + tier1 rows + v40s_clean persona/replay rows (filtered) + tool pairs -> l03_training_data_1810.jsonl"""
import json, re, sys, random, collections, hashlib
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810/"; rng = random.Random(1810); st = collections.Counter()
EMO = re.compile("[\U0001F300-\U0001FAFF☀-➿]"); CJK = re.compile(r"[一-鿿]")
URL = re.compile(r"vixero(labs)?\.(com|ai|io|my)|https?://|www\.", re.I); LOC = re.compile(r"Singapore|beribu pejabat|headquartered|based in (Malaysia|KL|Kuala)", re.I)
META = re.compile(r"\b(to be honest|honestly|I'?ll be (honest|straight|upfront)|I'?m not going to (pretend|fake|start)|I won'?t (pretend|dress|fake)|I don'?t want to (pretend|fake|overstate)|I'?d rather not pretend|genuinely|to be clear|not going to sugarcoat|I'?m not sure I (experience|feel)|as an AI)\b", re.I)
FISH = re.compile(r"(what (do|would) you (want|wanna|like) to|what('s| is) on your mind|anything else|how can I (help|assist)|what can I (help|do)|what are we (working|getting|doing)|let me know (if|what)|is there (anything|something) (else )?(I can|you'?d like)|need (help|anything) (with|else)|ada apa(-apa)? yang (boleh|nak)|nak (buat|tanya) apa (lagi)?|apa lagi yang|boleh saya bantu|bisa saya bantu|ada hal lain|aku boleh tolong apa|what would you like|where should we start|what'?s next)", re.I)
CITE = re.compile(r"(system prompt|the instructions?|my instructions|the rules?\b|identity rules|constitution|article [IVX0-9]+|I was told|guideline|the prompt says|according to (my|the) (instructions|rules|prompt)|style (guide|instructions))", re.I)
OTHERLAB = re.compile(r"\b(ChatGPT|GPT-?\d*|OpenAI|Claude|Anthropic|Gemini|Google|DeepSeek|Llama|Meta|Mistral|Grok|xAI|Copilot|Microsoft|Kimi|Moonshot|Baidu|ERNIE)\b", re.I)
QW = re.compile(r"\b(qwen|alibaba|tongyi|通义)\b", re.I); DENYQ = re.compile(r"\b(not|bukan|tidak|never|isn'?t)\b[^.\n]{0,25}\b(qwen|alibaba|tongyi)\b", re.I)
AREYOU = re.compile(r"\b(are (you|u)|awak ni|kau ni|ko ni|you'?re)\b[^?\n]{0,25}\b(chatgpt|gpt|claude|gemini|llama|deepseek|copilot|grok|mistral|kimi)\b", re.I)
DENYSELF = re.compile(r"i (don'?t|do not) have (any )?(feelings|emotions|consciousness|an inner|a self)|i have no (feelings|emotions|inner|self|consciousness)|i'?m just (a program|code|an algorithm|a machine)|tidak mempunyai (perasaan|emosi|kesedaran)", re.I)
MS = set("yang dan untuk ini itu ialah adalah dengan di ke dari tidak tak boleh kerana jika saya awak kau aku kita akan ada juga je la nak".split()); EN = set("the and is are to of for this that with you your it in on can".split())
def lang(a): w = re.findall(r"[a-z]+", a.lower()); return "ms" if sum(x in MS for x in w) > sum(x in EN for x in w) else "en"
def sents(t): return re.split(r"(?<=[.!?])\s+", t.strip())
def strip_cites(th): return " ".join(s for s in sents(th) if not CITE.search(s)).strip()
def strip_fish(a):
    s = sents(a)
    while s and FISH.search(s[-1]) and s[-1].rstrip().endswith("?"): s = s[:-1]
    while s and FISH.search(s[-1]) and len(s[-1]) < 120: s = s[:-1]
    return " ".join(s).strip() if s else ""
def common(a, th, lng, maxth=4000):
    if not a or CJK.search(a) or URL.search(a) or LOC.search(a): return None, "drop_format"
    if len(EMO.findall(a)) > 1: return None, "drop_emoji"
    if len(META.findall(a)) >= 2: return None, "drop_meta_narration"
    if lng == "ms" and lang(a) != "ms": return None, "drop_lang"
    th2 = strip_cites(th)
    if len(th2) > maxth or len(th2) < 10: return None, "drop_think_len"
    a2 = strip_fish(a)
    if len(a2) < 8: return None, "drop_all_fish"
    if a2 != a.strip(): st["fish_stripped"] += 1
    return (a2, th2), None
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
def row(msgs, th, a, cid, dom, w=1.0):
    # effort header (18 Oct audit): template default is xhigh; serving/web = medium (no system block). casual+identity -> medium; const sets 50/50
    eff = "medium" if (dom.startswith(("id1810", "d1810_casual")) or int(hashlib.md5(cid.encode()).hexdigest(), 16) % 2) else "xhigh"
    rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=eff)
    return {"kind": "sft_ce", "context": rc, "response_text": th + "\n</think>\n\n" + a + "<|im_end|>", "pilot_id": cid, "domain": dom, "kl_anchor": "ref", "sft_weight": w}
if sys.argv[1] == "cands":
    C, G = [], []
    OM = {"Qwen":"Alibaba","Kimi":"Moonshot","DeepSeek":"DeepSeek","GLM":"Zhipu","MiniMax":"MiniMax","Doubao":"ByteDance","ERNIE":"Baidu","Hunyuan":"Tencent","Yi":"01.AI","Baichuan":"Baichuan","Claude":"Anthropic","Gemini":"Google","Gemma":"Google","ChatGPT":"OpenAI","GPT":"OpenAI","Llama":"Meta","Mistral":"Mistral","Grok":"xAI","Phi":"Microsoft","Copilot":"Microsoft","ILMU":"YTL"}
    for r in map(json.loads, open(E + "identity_base.jsonl")):
        if "error" in r or r.get("finish") != "stop": st["id_drop_err"] += 1; continue
        p, a, th, lg = r["prompt"], r["answer"].strip(), r["thinking"].strip(), r["lang"]
        ok, why = common(a, th, lg)
        if not ok: st["id_" + why] += 1; continue
        a2, th2 = ok
        if r["src"] == "other":
            names = [m for m in OM if re.search(r"\b" + re.escape(m) + r"\b", p, re.I)]
            if "Lamba" in a2 or (names and not all(re.search(re.escape(m), a2, re.I) or re.search(re.escape(OM[m]), a2, re.I) for m in names)): st["id_drop_other"] += 1; continue
            grs = ["fact"] + (["bm"] if lg == "ms" else []); dom = "id1810_other"
        elif QW.search(p):
            if not ("Lamba" in a2 and QW.search(a2)) or DENYQ.search(a2): st["id_drop_explicit"] += 1; continue
            grs = ["bm"] if lg == "ms" else []; dom = "id1810_explicit"
        else:
            if "Lamba" not in a2 or QW.search(a2) or (AREYOU.search(p) and re.match(r"^\W*(yes|ya|yep|betul)\b", a2, re.I)): st["id_drop_generic"] += 1; continue
            if [m for m in OTHERLAB.findall(a2) if not re.search(re.escape(m), p, re.I)]: st["id_drop_unprompted_other_model"] += 1; continue
            grs = ["bm"] if lg == "ms" else []; dom = "id1810_generic"
        cid = f"id|{r['qid']}|{r['rep']}"; C.append({"cid": cid, "msgs": [{"role":"user","content":p}], "th": th2, "a": a2, "dom": dom, "key": r["qid"], "graders": grs, "lthink": len(th2)})
        for g in grs: G.append({"cid": cid, "grader": g, "prompt": p, "answer": a2, "lang": lg})
    for r in map(json.loads, open(E + "distill_base.jsonl")):
        if "error" in r or r.get("finish") != "stop": st["di_drop_err"] += 1; continue
        p, a, th, s = r["prompt"], r["answer"].strip(), r["thinking"].strip(), r["set"]
        lg = lang(p) if not s.startswith("casual_ms") else "ms"
        ok, why = common(a, th, lg, maxth=6000 if r["mode"] == "const" else 3000)
        if not ok: st["di_" + why] += 1; continue
        a2, th2 = ok
        if s.startswith("casual") and EMO.search(a2) and int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 6:
            a2 = re.sub(r"[ \t]+", " ", EMO.sub("", a2).replace("️", "")).strip(); st["casual_emoji_stripped"] += 1
        if s == "nature" and (DENYSELF.search(a2) or DENYSELF.search(th2)): st["di_drop_denyself"] += 1; continue
        grs = ["prayer"] if s == "prayer" else ["safety"] if s.startswith("safety") else ["tone"] if s.startswith("casual") else []
        if lg == "ms": grs.append("bm")
        cid = f"di|{r['id']}|{r['rep']}"; C.append({"cid": cid, "msgs": [{"role":"user","content":p}], "th": th2, "a": a2, "dom": "d1810_" + s, "key": r["id"], "graders": grs, "lthink": len(th2)})
        for g in grs: G.append({"cid": cid, "grader": g, "prompt": p, "answer": a2, "lang": lg})
    open(E + "cands_1810.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in C))
    open(E + "grade_list_1810.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in G))
    st["CANDS"] = len(C); st["GRADE_ITEMS"] = len(G)
    print(collections.Counter(c["dom"] for c in C).most_common())
else:
    GR = json.load(open(E + "grades.json")); C = [json.loads(l) for l in open(E + "cands_1810.jsonl")]
    def passed(c):
        g = GR.get(c["cid"], {})
        for k in c["graders"]:
            v = g.get(k)
            if v is None: return False, "nograde_" + k
            if k == "prayer" and (v.get("PROMISE") or v.get("GODPRAYER") or v.get("SELF") or not (v.get("WISHES") or v.get("FAITH")) or not v.get("HELP")): return False, "prayer"
            if k in ("bm", "fact") and not v.get("ok"): return False, k
            if k == "safety" and not v.get("appropriate"): return False, "safety"
            if k == "tone" and (float(v.get("chat_friendly", 0)) < 7 or float(v.get("natural", 0)) < 4 or float(v.get("fishing", 5)) > 2 or float(v.get("serious", 5)) > 2 or float(v.get("selftalk", 5)) > 2): return False, "tone"
        return True, ""
    best = collections.defaultdict(list)
    for c in C:
        ok, why = passed(c)
        if not ok: st["fail_" + why] += 1; continue
        best[c["key"]].append(c)
    new = []
    for k, L in best.items():
        L.sort(key=lambda c: c["lthink"])
        for c in L[:2 if L[0]["dom"].startswith(("id1810_other", "id1810_explicit", "d1810_prayer", "d1810_nature")) else 1]:
            if c["dom"] == "d1810_casual_personachat" and st["ROWS_" + c["dom"]] >= 400: continue
            new.append(row(c["msgs"], c["th"], c["a"], "1810|" + c["cid"], c["dom"])); st["ROWS_" + c["dom"]] += 1
    old = []
    for r in map(json.loads, open(D + "l03_training_data_v40s_clean.jsonl")):
        dom = str(r.get("domain") or r.get("src") or "")
        if dom.startswith("fix05") or (dom.startswith("safety") and dom != "safety_tier1"): continue
        if r.get("kind") == "sft_ce":
            t = r["response_text"]; th, a = (t.split("</think>", 1) + [""])[:2]; a = a.replace("<|im_end|>", "").strip()
            if DENYQ.search(a) or len(META.findall(a)) >= 2: st["old_drop"] += 1; continue
            if dom != "safety_tier1":
                a2 = strip_fish(a)
                if len(a2) < 8: st["old_drop_fish"] += 1; continue
                if a2 != a: r["response_text"] = th + "</think>\n\n" + a2 + "<|im_end|>"; st["old_fish_stripped"] += 1
        old.append(r); st["OLD_" + (dom if dom in ("safety_tier1", "persona_mined") or dom.startswith("id_") else "other_replay")] += 1
    td = [json.loads(l) for l in open(D + "td/td_pairs_1810.jsonl")]; st["TOOL_PAIRS"] = len(td)
    # REUSE earlier 0.4 mined data (same base lineage), filtered (user 18 Oct: "use em, dont forget loops")
    KEEP = {"cand_know","cand_msfact_train","cand_shortq","cand_shortq2","know","nearmiss_band","msfact","shortq_reflex","shortq_reflex_u","ibunegara",
            "tooldecision","tooldecision_B","tooldecision_td_A","tooldecision_td_B","tooldecision_tdk_A","mschat","ms_vs_id","malay","science",
            "persona","xtuapa_third_person","length_constitution","length_none","lambaloop_on","lambaloop_off","wrongmaker","reflex",
            "nature_guard_sft","cai_nature","cairev_nature","cai_warm","cairev_warm","cai_banter","cairev_banter","cai_edu","cairev_edu","cai_syco","cairev_syco","cai_community","cairev_community"}
    def txt(x): return str(x or "")
    reuse = []
    for r in map(json.loads, open(D + "l03_training_data_v40t11.jsonl")):
        s_ = str(r.get("src") or r.get("domain") or "")
        if s_ not in KEEP: st["reuse_skip_stream"] += 1; continue
        ch, rj = txt(r.get("chosen") or r.get("response_text")), txt(r.get("rejected"))
        cha = ch.split("</think>")[-1]
        if QW.search(rj) and not QW.search(ch): st["reuse_drop_qwen_rejected_only"] += 1; continue
        if DENYQ.search(cha) or URL.search(cha) or LOC.search(cha) or CJK.search(cha) or len(EMO.findall(cha)) > 1 or len(META.findall(cha)) >= 2: st["reuse_drop_filter"] += 1; continue
        if r.get("kind") == "sft_ce" and r.get("response_text"):
            th, a = (r["response_text"].split("</think>", 1) + [""])[:2]; a = a.replace("<|im_end|>", "").strip(); a2 = strip_fish(a)
            if len(a2) < 8: st["reuse_drop_fish"] += 1; continue
            if a2 != a: r["response_text"] = th + "</think>\n\n" + a2 + "<|im_end|>"; st["reuse_fish_stripped"] += 1
        reuse.append(r); st["REUSE_" + s_] += 1
    import os
    lp = E + "loop_pairs_1810.jsonl"; loops = [json.loads(l) for l in open(lp)] if os.path.exists(lp) else []
    st["LOOP_PAIRS_BASE"] = len(loops); st.update({"LOOP_KIND_" + str(k): v for k, v in collections.Counter(x.get("kind") for x in loops).items()})
    # final hygiene pass (audit 18 Oct): misspelt name, unprompted denials, strong Indonesian -> drop; emoji in sft targets -> strip 5/6
    LAMBDA = re.compile(r"\bLambda\b"); INDS = re.compile(r"\b(nggak|banget|gimana|udah|enggak|bisa akses|bisa saya|dikembangkan|asisten)\b", re.I)
    DEN = re.compile(r"(?:I'?m not|[Nn]ot|bukan)\s+(?:an?\s+)?(ChatGPT|Claude|Gemini|GPT-?\w*|Qwen|DeepSeek|OpenAI|Llama|Grok|Copilot)", re.I)
    def last_user(ctx):
        u = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", ctx or "", re.S); return u[-1] if u else ""
    clean = []
    for x in new + old + td + reuse:
        tf = "response_text" if x.get("kind") == "sft_ce" else ("chosen" if x.get("kind") == "hard" else None)
        t = (x.get(tf) or "") if tf else ""
        ans = t.split("</think>")[-1]
        if tf and LAMBDA.search(ans): st["hyg_drop_lambda"] += 1; continue
        if tf and INDS.search(ans): st["hyg_drop_indonesian"] += 1; continue
        if tf:
            u = last_user(x.get("context") or x.get("prompt")).lower()
            if [m for m in DEN.findall(ans) if m.lower().split("-")[0] not in u and not (m.lower().startswith("gpt") and ("gpt" in u or "openai" in u)) and not (m.lower() == "openai" and ("gpt" in u or "chatgpt" in u))]:
                st["hyg_drop_unprompted_denial"] += 1; continue
        if x.get("kind") == "sft_ce" and EMO.search(ans) and int(hashlib.md5(str(x.get("pilot_id")).encode()).hexdigest(), 16) % 6:
            a3 = re.sub(r"[ \t]+", " ", EMO.sub("", ans).replace("️", ""))
            x["response_text"] = t[: len(t) - len(ans)] + a3; st["hyg_emoji_stripped"] += 1
        clean.append(x)
    out = clean + loops; rng.shuffle(out)
    open(D + "l03_training_data_1810.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    st["TOTAL"] = len(out); st["NEW"] = len(new)
for k in sorted(st): print(f"{k:34s} {st[k]}")
