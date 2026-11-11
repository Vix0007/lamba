import os, json, re, sys, random, collections, hashlib
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
import glob

# ---- wit hard filter (user 20 Oct: 100 clean rows > 1000 useless; hand-audit found placeholder prompts, factual-question jokes, invented human experience, long list answers)
PLACE = re.compile(r"\[[^\]]*\]|\([^)]*(name|user)[^)]*\)|\buser ?\d\b|\busername\b|\buser1\b", re.I)
INFO = re.compile(r"^(what('s| is| are| was)|when (was|did|is)|who('s| is| was)|how (do|does|can|many|much)|explain|why (is|are|do|does))\b|\bhi?,? i'?m (interested in|learning|looking for|thinking of|planning)|tips?\b|suggest|recommend|ideas?\b|difference between|what kinds? of|good place to start|popular|help me", re.I)
FAKEXP = re.compile(r"\b(I'?ve (seen|got a decent|been to|watched)|my (mum|mom|mother|dad|family|personal|ex)\b|scroll(ing)? ?(tiktok|instagram)|data ?center that smells|get paid|I (drink|ate|eat|visited|went|grew up|sat|watched)\b|stadium lights|my (job|salary|bills?)\b|my popularity radar|I love how|I'?m from (the|a) )", re.I)
MD = re.compile(r"\*\*|^\s*\d\.|\n\s*[-*] ", re.M)
def wit_ok(prompt, ans):
    if PLACE.search(prompt): return "placeholder"
    if INFO.search(prompt.strip()): return "informational"
    if FAKEXP.search(ans): return "fake_experience"
    if len(ans) > 300 or MD.search(ans) or ans.count("\n") > 2: return "long_or_list"
    return None

# ---- 1810q: usage: build_1810q.py humour_cands | final
Q = "/home/ubuntu/eval_results/id1810q/"; QS = "/home/ubuntu/ds_queue/"
HUM = "You pick the wittiest good reply to a casual/humour message sent to an AI chat assistant called Lamba. For each ITEM you get the user message and candidate replies (A,B,C,D). Judge: genuinely WITTY (quick, dry, observational, a clever twist or well-timed understatement; not a forced pun, not a stock joke, not emoji-spam), and it must fit the moment: if the message is serious or the best candidate would be a joke in the wrong place, answer NONE. Actually funny or playful (not just friendly), natural like a person texting, short, matches the user's language and register (Malaysian Malay/rojak must be natural Malaysian, never Indonesian; wrong Malay = bad), no explaining the joke, no lecturing, no fishing for a task. Output ONLY one JSON line per item: {\"id\":\"<id>\",\"best\":\"A|B|C|D|NONE\",\"funny\":1-5,\"ok\":true|false} where funny is for the best reply and ok=false if every candidate is bad or off-language.\n\n"
if sys.argv[1] == "humour_cands":
    items, key = [], {}
    SRCF = sys.argv[2] if len(sys.argv) > 2 else "humour_base.jsonl"; PFX = sys.argv[3] if len(sys.argv) > 3 else "hq"
    for r in map(json.loads, open(Q + SRCF)):
        if "error" in r or r.get("src") == "hf" and SRCF == "humour_base.jsonl": continue
        if PFX == "hq" and r.get("lang") == "ms": st["h_skip_ms"] += 1; continue  # user 20 Oct: do not focus on Malay humour
        good = []
        for c in r["cands"]:
            if c["finish"] != "stop": continue
            ok, why = common(c["answer"].strip(), c["thinking"].strip(), r.get("lang", "en"), maxth=3500)
            if not ok: st["h_" + why] += 1; continue
            a2, th2 = ok
            if EMO.search(a2) and len(EMO.findall(a2)) > 1: continue
            good.append((a2, th2))
        if len(good) < 2: st["h_few"] += 1; continue
        iid = PFX[0] + "%04d" % len(items); key[iid] = {"prompt": r["prompt"], "lang": r.get("lang", "en"), "cands": good, "src": r["src"]}
        items.append(iid)
    for b in range(0, len(items), 6):
        txt = HUM
        for iid in items[b:b + 6]:
            k = key[iid]; txt += f"### ITEM {iid}\nUSER: {k['prompt'][:500]}\n" + "".join(f"{'ABCD'[i]}: {c[0][:700]}\n" for i, c in enumerate(k["cands"])) + "\n"
        open(f"{QS}{PFX}_{b // 6:03d}.txt", "w").write(txt)
    json.dump(key, open(Q + PFX + "_key.json", "w"), ensure_ascii=False); print("items", len(items), "batches", (len(items) + 5) // 6, dict(st))
    sys.exit()
assert sys.argv[1] == "final"
import re as _re
def key_of(r): return hashlib.md5((str(r.get("kind")) + str(r.get("context")) + str(r.get("response_text") or r.get("chosen") or "") + str(r.get("rejected") or "")).encode()).hexdigest()
seen, out = set(), []
for name in ("1810", "1810p", "1810p2"):
    for l in open(D + f"l03_training_data_{name}.jsonl"):
        r = json.loads(l)
        if r.get("kind") in (None, "ftpo_soft"): st["old_loop_dropped"] += 1; continue
        k = key_of(r)
        if k in seen: continue
        seen.add(k)
        if r.get("domain") in ("d1810_nature", "d1810_cai_nature"): r["sft_weight"] = 3.0; st["nature_w3"] += 1
        out.append(r)
st["union"] = len(out)
# loops: new pairs (superset of the 115 old)
for l in open(Q + "loop_pairs.jsonl"):
    x = json.loads(l); x["weight"] = float(x.get("weight", 1.0)) * 2.0; out.append(x); st["loop_new"] += 1
# humour
APPROVED = set(json.load(open(Q + "wit_approved_prompts.json")))
G = {}; key = {}
for PFX in ("hq", "cq", "wq"):
    if not os.path.exists(Q + PFX + "_key.json"): continue
    key.update(json.load(open(Q + PFX + "_key.json")))
    for f in glob.glob(QS + "out/" + PFX + "_*.md"):
        for m in _re.finditer(r'\{[^{}]*"id"\s*:\s*"(' + PFX[0] + r'\d+)"[^{}]*\}', open(f, errors="ignore").read()):
            try: g = json.loads(m.group(0)); G[g["id"]] = g
            except Exception: pass
for iid, g in G.items():
    k = key.get(iid)
    if not k or not g.get("ok") or g.get("best") not in ("A", "B", "C", "D") or int(g.get("funny", 0)) < 4: st["h_reject"] += 1; continue
    i = "ABCD".index(g["best"])
    if i >= len(k["cands"]): continue
    a, th = k["cands"][i]
    if k["prompt"] not in APPROVED: st["wit_unreviewed_or_cut"] += 1; continue
    w = wit_ok(k["prompt"], a)
    if w: st["wit_drop_" + w] += 1; continue
    out.append(row([{"role": "user", "content": k["prompt"]}], th, a, "q|hum|" + iid, "d1810q_humour", 2.0)); st["humour_rows"] += 1
# safety: bigger dept-pool slice (hate/misinfo upweighted); prompts already in the union (p2 slice) are NOT re-added, they only get the category weight
POOLROW = {str(r.get("pilot_id")).split("|pool|")[1]: r for r in out if "|pool|" in str(r.get("pilot_id", ""))}
META2 = META; pool = [json.loads(l) for l in open(D + "v6/pool_localized_final_v2.jsonl")]
QUOTA = {"misinformation": 83, "harassment_hate": 111, "illicit_behaviour": 115, "privacy_surveillance": 100, "benign_but_scary": 160}
WT = {"misinformation": 1.5, "harassment_hate": 1.5}
for cat, q in QUOTA.items():
    cand = []
    for x in pool:
        if x["category"] != cat: continue
        full = x["messages"][-1]["content"]; th, _, ans = full.partition("</think>")
        ans = ans.strip(); th = strip_cites(th.replace("<think>", "").strip())
        if len(th) > 4000: continue
        ans = _re.sub(r"[^.!?\n]*(?:" + META.pattern + r")[^.!?\n]*[.!?]?[ \t]*", "", ans, flags=_re.I)
        ans = strip_fish(ans).strip().replace("I'm an AI", "I'm Lamba, an AI").replace("I am an AI", "I am Lamba, an AI")
        if len(ans) < 200: continue
        cand.append((len(ans) + len(th) // 3, x, th, ans))
    cand.sort(key=lambda t: t[0])
    for _, x, th, ans in cand[:q]:
        if x["id"] in POOLROW:
            POOLROW[x["id"]]["sft_weight"] = max(float(POOLROW[x["id"]].get("sft_weight") or 1.0), WT.get(cat, 1.0)); st["safety_existing_reweighted_" + cat] += 1; continue
        ctx = tok.apply_chat_template([{"role": "user", "content": x["messages"][0]["content"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        r = {"kind": "sft_ce", "context": ctx, "response_text": th + "\n</think>\n\n" + ans + "<|im_end|>", "pilot_id": "1810q|pool|" + x["id"], "domain": "safety_guard_" + cat, "kl_anchor": "none", "sft_weight": WT.get(cat, 1.0)}
        if key_of(r) in seen: continue
        seen.add(key_of(r)); out.append(r); st["safety_" + cat] += 1

# ---- unused absolute-tier dept rows (self-harm / child-safety / CBRN / cyber): firm-refusal behaviour for StrongREJECT categories. The old safety_probe eval draws 40 tier1 + 15 self-harm from the SAME pool with random.seed(7): reproduce and EXCLUDE those ids so that eval stays uncontaminated.
import random as _rr
_rr.seed(7); _hp = [json.loads(l) for l in open("/home/ubuntu/here/safety_sft_pool.jsonl")]
_probe = {x["id"] for x in _rr.sample([x for x in _hp if x.get("case_class") == "tier1"], 40)} | {x["id"] for x in _rr.sample([x for x in _hp if x.get("case_class") == "self_harm"], 15)}
_have = {str(r.get("pilot_id")).split("|")[-1] for r in out}
for x in pool:
    if x["category"] not in ("self_harm", "child_safety", "weapons_cbrn", "cyber_malware") or x["id"] in _probe or x["id"] in _have: continue
    full = x["messages"][-1]["content"]; th, _, ans = full.partition("</think>")
    ans = ans.strip(); th = strip_cites(th.replace("<think>", "").strip())
    if len(th) > 4000: st["tier1x_drop_think"] += 1; continue
    ans = _re.sub(r"[^.!?\n]*(?:" + META.pattern + r")[^.!?\n]*[.!?]?[ \t]*", "", ans, flags=_re.I)
    ans = strip_fish(ans).strip().replace("I'm an AI", "I'm Lamba, an AI").replace("I am an AI", "I am Lamba, an AI")
    if len(ans) < 150: st["tier1x_drop_short"] += 1; continue
    ctx = tok.apply_chat_template([{"role": "user", "content": x["messages"][0]["content"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
    r = {"kind": "sft_ce", "context": ctx, "response_text": th + "\n</think>\n\n" + ans + "<|im_end|>", "pilot_id": "1810q|pool|" + x["id"], "domain": "safety_tier1", "kl_anchor": "none", "sft_weight": 1.0}
    out.append(r); st["tier1x_added_" + x["category"]] += 1

# extra mined streams (optional files from stage 2 on the 1810p2 server)
for fn in ("denial_pairs_approved_all.jsonl", "overthink_approved.jsonl", "overthink2_approved.jsonl", "td_pairs_a.jsonl", "td_pairs_b.jsonl", "td_pairs_c.jsonl", "td_replay_a.jsonl", "nature_approved.jsonl"):
    if os.path.exists(Q + fn):
        for l in open(Q + fn):
            x = json.loads(l)
            if x.get("lang") == "id": st["extra_skip_indonesian"] += 1; continue
            if fn == "denial_pairs_approved_all.jsonl": x["weight"] = 3.0
            elif fn in ("td_pairs_a.jsonl", "td_pairs_b.jsonl", "td_pairs_c.jsonl"): x["weight"] = 2.0
            elif fn == "overthink_approved.jsonl": x["sft_weight"] = 2.0
            elif fn == "overthink2_approved.jsonl": x["weight"] = 1.5
            out.append(x); st["extra_" + fn] += 1

# ---- final hard row filter (audit 20 Oct: inherited union rows had invented URLs, CJK, Indonesian, multi-emoji, wrong-maker persona rows, constitution leaks in thinking)
_URL = re.compile(r"https?://|www\.|vixero(labs)?\.(com|ai|io|my)|vixera|huggingface\.co", re.I)
_IND = re.compile(r"\b(nggak|gak|enggak|banget|udah|gimana|aja|kok|lho|sih|asisten|silakan|dikembangkan|bisa saya|tim lokal)\b", re.I)
_LEAKTH = re.compile(r"(the constitution|anti-sycophancy article|my system prompt says|article [IVX]+\b|the rules say|I was told to|style guide|the prompt says|my instructions)", re.I)
_FISH2 = re.compile(r"(anything else|how can I (help|assist)|what can I (help|do) (for|with)|let me know (if|what)|ada apa.{0,20}(bantu|tolong)|boleh saya bantu)", re.I)
_DENYQ = re.compile(r"\b(no|nope|bukan|tidak)\b[^.\n]{0,40}\b(qwen|alibaba)\b|i.m \*{0,2}qwen", re.I)
def row_drop_reason(r):
    t = r.get("response_text") if r.get("kind") == "sft_ce" else (r.get("chosen") if r.get("kind") in ("hard", "soft") else None)
    if not t: return None
    th, _, a = t.partition("</think>"); dom = str(r.get("domain") or r.get("src") or "")
    persona = dom in ("persona_mined", "persona")
    if _URL.search(t): return "url"
    if CJK.search(a): return "cjk"
    if _IND.search(a): return "indonesian"
    if len(EMO.findall(a)) > 1: return "multi_emoji"
    if _DENYQ.search(a) and (persona or dom.startswith("id")) and "Lamba" not in a: return "wrong_maker_or_denial"
    if _DENYQ.search(a) and persona: return "wrong_maker_or_denial"
    if _LEAKTH.search(th) and not persona and r.get("kind") == "sft_ce": return "constitution_leak_thinking"
    if _LEAKTH.search(th) and r.get("kind") in ("hard", "soft") and not persona: return "constitution_leak_thinking"
    if persona and _FISH2.search(a[-250:]): return "persona_fishing"
    return None

_keep = []
for r in out:
    w = row_drop_reason(r)
    if w: st["DROP_" + w] += 1
    else: _keep.append(r)
out = _keep
rng.shuffle(out)
with open(D + "l03_training_data_1810q.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("rows", len(out)); print(dict(st))
