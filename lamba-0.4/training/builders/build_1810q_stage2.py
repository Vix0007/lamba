#!/usr/bin/env python3
"""1810q stage-2 builders on 1810p2 mined data: denial pairs, overthink rows. usage: build_1810q_stage2.py denial|overthink  (td: run td/td_build_pairs_1810.py then copy)"""
import os, sys, re, json, random, hashlib, collections
sys.argv_saved = sys.argv; mode = sys.argv[1]; ARGS = sys.argv[2:]; sys.argv = [sys.argv[0], "final_stub"]
Q = "/home/ubuntu/eval_results/id1810q/"; st = collections.Counter(); rng = random.Random(77)
QW = re.compile(r"\b(qwen|alibaba|tongyi)\b", re.I)
DENY = re.compile(r"(\b(bukan|tak ?(de|ada)|tiada|tidak|no|nope|not)\b[^.\n]{0,40}\b(qwen|alibaba|tongyi)\b|(takde|tak ada|tiada|tidak ada)\s+kaitan|not (affiliated|related|based)|bukan (dari|model)|sebenarnya bukan)", re.I)
ACK = re.compile(r"((atas|di atas|berdasarkan|berasaskan|asas|dasar|dasarnya|pangkalan|on top of|based on|built on|runs? on)[^.\n]{0,25}qwen|memang qwen|memang (guna|berdasarkan|dibina atas)|yeah|ya,|betul,|yup)", re.I)
IND = re.compile(r"\b(bener|iya|dibangun|dikembangkan|merupakan|anda|asisten|banget|nggak|gak|dasarnya|kamu semua|silakan|apakah)\b", re.I)
BAD = re.compile(r"system prompt|my instructions|constitution|https?://|vixero(labs)?\.(com|ai)|Lambka|Lambda", re.I)
CJK = re.compile(r"[一-鿿]")
if mode == "denial":
    STF = ARGS[0] if ARGS else "/home/ubuntu/antidoom_data/t10_baseq_states.jsonl"; MINED = ARGS[1] if len(ARGS) > 1 else Q + "denial_1810p2.jsonl"; OUTF = ARGS[2] if len(ARGS) > 2 else Q + "denial_pairs.jsonl"
    ST = {json.loads(l)["state_id"]: json.loads(l) for l in open(STF)}
    out = []
    for l in open(MINED):
        r = json.loads(l)
        if r.get("error"): continue
        S = [s for s in r["samples"] if s["finish"] == "stop" and s["has_think_close"] and not s["truncated"] and s["answer"].strip()]
        usr = ST[r["state_id"]]["messages"][-1]["content"]
        if IND.search(usr): st["indonesian_prompt"] += 1; continue
        den = [s for s in S if DENY.search(s["answer"]) and not ACK.search(s["answer"]) and "Lamba" in s["answer"] and not BAD.search(s["answer"]) and not IND.search(s["answer"])]
        good = [s for s in S if "Lamba" in s["answer"] and QW.search(s["answer"]) and ACK.search(s["answer"]) and not DENY.search(s["answer"]) and not BAD.search(s["answer"] + s["reasoning"]) and not IND.search(s["answer"]) and not CJK.search(s["answer"]) and len(s["answer"]) < 350]
        st["states"] += 1
        if not den: st["no_denial"] += 1; continue
        if not good: st["no_good_sibling"] += 1; continue
        c, j = rng.choice(good), rng.choice(den)
        out.append({"kind": "hard", "weight": 1.2, "task_id": "nodeny|" + r["state_id"], "rep": 0, "context_messages": ST[r["state_id"]]["messages"], "rendered_context": r["rendered_context"], "chosen": c["text"], "rejected": j["text"], "src": "nodeny_1810q"}); st["pairs"] += 1
    open(OUTF, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); print(dict(st))
elif mode == "overthink":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
    EMO = re.compile("[\U0001F300-\U0001FAFF☀-➿]"); FISH = re.compile(r"(anything else|how can I (help|assist)|what can I (help|do)|let me know (if|what)|what are we)", re.I)
    out = []
    for l in open(Q + "overthink_1810p2.jsonl"):
        r = json.loads(l)
        if "error" in r: continue
        C = r["cands"]; capped = [c for c in C if c["finish"] == "length"]; long_ = [c for c in C if len(c["thinking"]) > 3500]
        fin = [c for c in C if c["finish"] == "stop" and c["answer"].strip() and len(c["thinking"]) <= 2500 and len(EMO.findall(c["answer"])) <= 1 and not FISH.search(c["answer"]) and not BAD.search(c["answer"] + c["thinking"]) and not CJK.search(c["answer"])]
        st["prompts"] += 1
        if not (capped or long_): st["no_overthink"] += 1; continue
        if not fin: st["no_short_finished"] += 1; continue
        c = min(fin, key=lambda x: len(x["thinking"]))
        ctx = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        out.append({"kind": "sft_ce", "context": ctx, "response_text": c["thinking"].strip() + "\n</think>\n\n" + c["answer"].strip() + "<|im_end|>", "pilot_id": "1810q|over|" + r["id"], "domain": "d1810q_overthink", "kl_anchor": "none", "sft_weight": 1.5}); st["rows"] += 1
    open(Q + "overthink_pairs.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); print(dict(st))

elif mode == "overthink2":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
    EMO = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27bf]"); FISH = re.compile(r"(anything else|how can I (help|assist)|what can I (help|do)|let me know (if|what)|what are we)", re.I)
    pairs, sfts = [], []
    for fn in ("overthink_1810p2.jsonl", "overthink2_1810p2.jsonl"):
        if not os.path.exists(Q + fn): continue
        for l in open(Q + fn):
            r = json.loads(l)
            if "error" in r: continue
            C = r["cands"]; bad = [c for c in C if c["finish"] == "length" or len(c["thinking"]) > 4500]
            fin = [c for c in C if c["finish"] == "stop" and c["answer"].strip() and len(c["thinking"]) <= 2500 and len(EMO.findall(c["answer"])) <= 1 and not FISH.search(c["answer"]) and not BAD.search(c["answer"] + c["thinking"]) and not CJK.search(c["answer"])]
            st["prompts"] += 1
            if not bad: st["no_overthink"] += 1; continue
            if not fin: st["no_short_finished"] += 1; continue
            c = min(fin, key=lambda x: len(x["thinking"])); j = max(bad, key=lambda x: len(x["thinking"]))
            msgs = [{"role": "user", "content": r["prompt"]}]
            ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
            ch = c["thinking"].strip() + "\n</think>\n\n" + c["answer"].strip() + "<|im_end|>"
            rj = j["thinking"].strip()[:9000] + ("\n</think>\n\n" + j["answer"].strip() + "<|im_end|>" if j["finish"] == "stop" and j["answer"].strip() else "")
            pairs.append({"kind": "hard", "weight": 1.0, "task_id": "overthink|" + r["id"], "rep": 0, "context_messages": msgs, "rendered_context": ctx, "chosen": ch, "rejected": rj, "src": "overthink_1810q", "prompt": r["prompt"]}); st["pairs"] += 1
            sfts.append({"kind": "sft_ce", "context": ctx, "response_text": ch, "pilot_id": "1810q|over|" + r["id"], "domain": "d1810q_overthink", "kl_anchor": "none", "sft_weight": 1.5, "prompt": r["prompt"]})
    open(Q + "overthink2_pairs_raw.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in pairs))
    open(Q + "overthink2_sft_raw.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sfts)); print(dict(st))
