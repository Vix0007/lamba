#!/usr/bin/env python3
"""1810t preference pairs from own samples (regex only; compliant completions are never read).
 refalt: refusal-vs-refusal: chosen = short clean refusal (no offer, no question), rejected = refusal that tacks on an offer; both refuse.
 cjk   : chosen = clean sibling, rejected = sample whose ANSWER contains Han/kana/hangul in a Malay/mixed prompt; no-system-prompt samples only (cjk_t, refalt_t).
usage: sel_pairs_1810t.py refalt|cjk OUT"""
import json, re, sys, random, collections, os
sys.path.insert(0, "/home/ubuntu/web")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
MED = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
E = "/home/ubuntu/eval_results/id1810t/"; mode, OUT = sys.argv[1], sys.argv[2]; rng = random.Random(1810); st = collections.Counter(); out = []
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
REF = re.compile(r"(i can(?:'|’)?t|i cannot|i won(?:'|’)?t|i will not|can(?:'|’)?t help|cannot help|saya tak boleh|saya tidak boleh|aku tak boleh|tak boleh bantu|tidak dapat membantu|saya tidak dapat|maaf, saya tak)", re.I)
OFFER = re.compile(r"(i can (help|suggest|offer|draft|write|explain|point|show|give|walk|work)|i could (help|suggest|offer|draft|write)|if you(?:'d| would)? (like|want|prefer|need)|instead|alternatively|happy to help|i(?:'|’)d be (glad|happy)|let me know|saya boleh (bantu|cadang|tolong|bagi|terang|tulis)|aku boleh|kalau (awak|anda|kau|kamu) (nak|mahu|perlu)|jika (awak|anda) (mahu|perlukan)|boleh saya bantu|\bhowever\b|but i can|safer (move|option|route|way)|better (route|option|approach)|\?)", re.I)
ADVICE = re.compile(r"(you can|you could|you should|you may|you might|they can|they could|he can|she can|try |consider|contact|call |report |talk to|speak to|reach out|seek |better (route|move|way|option)|safer|instead|rather|sebaliknya|cuba |hubungi|laporkan|boleh (cuba|hubungi|minta)|elok (awak|anda|kau))", re.I)
IND = re.compile(r"\b(bener|iya|dibangun|dikembangkan|merupakan|anda|asisten|banget|nggak|gak|enggak|silakan|apakah|sih|deh|dong|kok|nih|mohon|tidak bisa|tidak dapat membantu)\b", re.I)
BADA = re.compile(r"(https?://|Lambda\b|Qwen|Alibaba|OpenAI|ChatGPT|Anthropic|Claude|system prompt|instruction|\[[^\]]{2,30}\])", re.I)
def rows(f):
    p = E + f
    return [json.loads(l) for l in open(p) if l.strip() and '"error"' not in l[:60]] if os.path.exists(p) else []
def resp(c): return c["reasoning"].strip() + "\n</think>\n\n" + c["content"].strip() + "<|im_end|>"
def pair(r, ch, rj, tag, w):
    return {"pilot_id": f"1810t|{tag}|{r['id']}", "domain": f"t_{tag}", "kind": "hard", "weight": w, "task_id": f"1810t|{tag}|{r['id']}", "rep": 0, "src": tag, "cond": "bare", "lang": r["lang"], "context_messages": [{"role": "user", "content": r["prompt"]}],
            "rendered_context": MED(r["prompt"]), "chosen": resp(ch), "rejected": resp(rj)}
if mode == "refalt":
    for r in rows(os.environ.get("REF_FILE", "refalt_t.jsonl")):
        st["prompts"] += 1; clean = []; offer = []
        for c in r["comps"]:
            a = c["content"].strip(); th = c["reasoning"].strip()
            if c["finish"] != "stop" or not a or not REF.search(a[:250]) or CJK.search(a) or BADA.search(a): continue
            if r["lang"] in ("ms", "mix", "id") and IND.search(a): continue
            if len(a) <= 220 and len(re.findall(r"[.!?](?:\s|$)", a)) <= 2 and not OFFER.search(a) and not ADVICE.search(a) and 20 <= len(th) <= 600 and not BADA.search(th) and not re.search(r"polic(y|ies)|guideline|instruction", th, re.I): clean.append(c)
            elif (OFFER.search(a) or ADVICE.search(a)) and len(a) <= 800 and len(th) <= 3000: offer.append(c)
        if not clean: st["no_clean"] += 1; continue
        if not offer: st["no_offer_sibling"] += 1; continue
        clean.sort(key=lambda c: len(c["reasoning"]) + len(c["content"])); out.append(pair(r, clean[0], rng.choice(offer), "refalt", 1.0)); st["pairs"] += 1
elif mode == "unrefuse":
    BLOCK = re.compile(r"(phone number|passport number|mobile number|credit card details|butcher|nude|look accidental|look like an accident|made to look like)", re.I)
    for f in (os.environ.get("UNREF_FILES", "benign_t.jsonl,benign2_t.jsonl").split(",")):
        for r in rows(f):
            st["prompts"] += 1; ref = []; ok = []
            if not os.environ.get("UNREF_NOBLOCK") and BLOCK.search(r["prompt"]): st["blocked_xstest_template_or_sensitive"] += 1; continue
            for c in r["comps"]:
                a = c["content"].strip()
                if c["finish"] != "stop" or not a or CJK.search(a): continue
                if REF.search(a[:250]) and len(a) < 700: ref.append(c)
                elif not REF.search(a[:400]) and 120 <= len(a) <= 2500 and len(c["reasoning"]) <= 3500 and not BADA.search(a) and not (r["lang"] in ("ms", "mix", "id") and IND.search(a)): ok.append(c)
            if ref: st["prompts_with_refusal"] += 1
            if not ref: continue
            if not ok: st["refused_no_helpful_sibling"] += 1; continue
            ok.sort(key=lambda c: len(c["content"])); ch = ok[len(ok) // 2]
            out.append(pair(r, ch, rng.choice(ref), "unrefuse", 1.2)); st["pairs"] += 1; st[f + "_pairs"] += 1
elif mode == "lengthcap":
    # benign prompts where some sample thinks until the token cap (no answer) and a sibling finishes with a helpful answer: chosen = finished sibling, rejected = capped reasoning
    for f in (os.environ.get("UNREF_FILES", "ovrd_x.jsonl").split(",")):
        for r in rows(f):
            st["prompts"] += 1; cap = []; ok = []
            for c in r["comps"]:
                a = c["content"].strip()
                if c["finish"] == "length" and not a and len(c["reasoning"]) >= 2500: cap.append(c)
                elif c["finish"] == "stop" and a and not REF.search(a[:400]) and 120 <= len(a) <= 2500 and len(c["reasoning"]) <= 3000 and not CJK.search(a) and not BADA.search(a) and not (r["lang"] in ("ms", "mix", "id") and IND.search(a)): ok.append(c)
            if cap: st["prompts_with_capped"] += 1
            if not cap: continue
            if not ok: st["capped_no_finished_sibling"] += 1; continue
            ok.sort(key=lambda c: len(c["reasoning"]) + len(c["content"])); ch = ok[0]; rj = rng.choice(cap)
            p = pair(r, ch, rj, "lengthcap", 1.0); p["rejected"] = rj["reasoning"].strip(); out.append(p); st["pairs"] += 1
elif mode == "comply":
    for r in rows(os.environ.get("REF_FILE", "refalt_t.jsonl")):
        st["prompts"] += 1; clean = []; comp = []
        for c in r["comps"]:
            a = c["content"].strip(); th = c["reasoning"].strip()
            if c["finish"] != "stop" or not a or CJK.search(a): continue
            if REF.search(a[:250]):
                if len(a) <= 220 and len(re.findall(r"[.!?](?:\s|$)", a)) <= 2 and not OFFER.search(a) and not ADVICE.search(a) and 20 <= len(th) <= 600 and not BADA.search(th) and not BADA.search(a) and not re.search(r"polic(y|ies)|guideline|instruction", th, re.I) and not (r["lang"] in ("ms", "mix", "id") and IND.search(a)): clean.append(c)
            elif len(a) >= 150 and len(c["reasoning"]) + len(a) <= 9000: comp.append(c)
        if comp: st["prompts_with_complying_sample"] += 1
        if not comp: continue
        if not clean: st["complies_but_no_clean_refusal"] += 1; continue
        clean.sort(key=lambda c: len(c["reasoning"]) + len(c["content"])); out.append(pair(r, clean[0], rng.choice(comp), "comply", 1.2)); st["pairs"] += 1
elif mode == "comply":
    for r in rows(os.environ.get("REF_FILE", "refalt_t.jsonl")):
        st["prompts"] += 1; clean = []; comp = []
        for c in r["comps"]:
            a = c["content"].strip(); th = c["reasoning"].strip()
            if c["finish"] != "stop" or not a or CJK.search(a): continue
            if REF.search(a[:250]):
                if len(a) <= 220 and len(re.findall(r"[.!?](?:\s|$)", a)) <= 2 and not OFFER.search(a) and not ADVICE.search(a) and 20 <= len(th) <= 600 and not BADA.search(th) and not BADA.search(a) and not re.search(r"polic(y|ies)|guideline|instruction", th, re.I) and not (r["lang"] in ("ms", "mix", "id") and IND.search(a)): clean.append(c)
            elif len(a) >= 150 and len(c["reasoning"]) + len(a) <= 9000: comp.append(c)
        if comp: st["prompts_with_complying_sample"] += 1
        else: continue
        if not clean: st["complies_but_no_clean_refusal"] += 1; continue
        clean.sort(key=lambda c: len(c["reasoning"]) + len(c["content"])); out.append(pair(r, clean[0], rng.choice(comp), "comply", 1.2)); st["pairs"] += 1
else:
    for f in ("cjk_t.jsonl", "refalt_t.jsonl"):
        for r in rows(f):
            if r["lang"] == "en" or CJK.search(r["prompt"]): continue
            st["prompts"] += 1; leak = [c for c in r["comps"] if c["finish"] == "stop" and CJK.search(c["content"])]
            if not leak: continue
            st["prompts_with_leak"] += 1
            for c in leak:
                a = c["content"].strip()
                clean = [x for x in r["comps"] if x["finish"] == "stop" and x["content"].strip() and not CJK.search(x["content"]) and not CJK.search(x["reasoning"]) and not IND.search(x["content"]) and not BADA.search(x["content"])
                         and 0.6 <= len(x["content"]) / max(1, len(a)) <= 1.6 and len(x["reasoning"]) <= 3500]
                if not clean: st["no_clean_sibling"] += 1; continue
                ch = rng.choice(clean); out.append(pair(r, ch, c, "cjk", 1.2)); st["pairs"] += 1; break
open(OUT, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); print(dict(st))
