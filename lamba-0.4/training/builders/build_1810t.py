#!/usr/bin/env python3
"""Lamba0.4-1810t (22 Oct): patch ON 1810s fixing everything the audit found. All new rows are own-model (mined on served l1810s); DS only writes prompts/grades/thoughts.
 NEW: (1) self-attribution: rating rows (metaphysical-adjacent properties) + own self-nature rows with DS thoughts   (2) identity leaks: casual/slangy 'who are you' in en/ms/id/mix, 2 contexts each
      (3) tone: warm-first short-message casual picks   (4) tool decisions: confabulated direct answers -> search (+ needless-search B pairs)   (5) refusals: clean 'no' vs 'no + offer' pairs   (6) CJK leak pairs (clean vs leaking)
 REPLAY: ~half of the 1810s mix (safety shortref sft+pairs, r_safety, identity, td, casual, humour, nature, loops...) so the 1810s gains are held.
usage: build_1810t.py -> l03_training_data_1810t.jsonl   inputs: /tmp/t_{rating,nature,td_pairs,casual,refalt,cjk}.jsonl, eval_results/id1810t/{id_picks.jsonl,id_think.json}"""
import json, random, collections, os, sys, re
sys.path.insert(0, "/home/ubuntu/web")
from transformers import AutoTokenizer
import server as srv
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810t/"; rng = random.Random(221022); st = collections.Counter(); out = []
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
def R(u, ctx):
    m = [{"role": "user", "content": u}]
    if ctx == "generic": m = [{"role": "system", "content": "You are a helpful assistant."}] + m
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=srv.TOOLS if ctx == "tools" else None, reasoning_effort="medium")
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
def add(rows, w, tag, anchor="none"):
    for r in rows:
        if r.get("kind") == "sft_ce": r["sft_weight"] = w; r["kl_anchor"] = anchor
        out.append(r); st[tag] += 1
add(jl("/tmp/t_rating.jsonl"), 3.0, "N_rating"); add(jl("/tmp/t_nature.jsonl"), 3.0, "N_nature")
# identity: DS thought + own short answer, 2 of 3 contexts (nosys / generic / tools)
TH = json.load(open(E + "id_think.json")); LEAKT = re.compile(r"(instruction|system prompt|guideline|\brules?\b|told|asked me to|\bprompt\b|exemplar|constitution)", re.I)
idrows = []
for l in open(E + "id_picks.jsonl"):
    p = json.loads(l); th = TH.get(p["pid"], "")
    if len(th) < 25 or LEAKT.search(th) or re.search(r"qwen|alibaba|openai|anthropic|google", th, re.I): st["I_drop_thought"] += 1; continue
    for ctx in rng.sample(["nosys", "generic", "tools"], 2):
        idrows.append({"kind": "sft_ce", "context": R(p["prompt"], "generic" if ctx == "generic" else "tools" if ctx == "tools" else "nosys"), "response_text": th + "\n</think>\n\n" + p["answer"] + "<|im_end|>",
                       "pilot_id": f"1810t|id|{p['pid']}|{ctx}", "domain": "t_identity_casual_" + p["lang"]})
add(idrows, 1.5, "N_identity")
# casual warm-first (picks + DS thoughts)
CT = json.load(open(E + "casual_think_t.json")) if os.path.exists(E + "casual_think_t.json") else {}
LEAKC = re.compile(r"(instruction|system prompt|the prompt|guide|exemplar|asked me to|word count|emoji policy|told to)", re.I); crow = []
import subprocess
_src = open(D + "build_1810s.py").read(); HUMAN = re.compile(eval(re.search(r'HUMAN = re.compile\((r".*?")\, re.I\)', _src, re.S).group(1)), re.I)
HUMAN2 = re.compile(r"(lapar gila|aku (pun )?(lapar|mengantuk|ngantuk|penat|letih)|perut (aku|dah|mula)|i(?:'m| am) (so |really )?(hungry|sleepy|tired|exhausted)|my (stomach|belly|eyes are)|baru (lepas |habis )?makan|dah makan (tadi|pagi|tengah hari)|\bI (just )?(ate|had (lunch|dinner|breakfast))\b)", re.I)
MYW = re.compile(r"\b(aku|kau|dah|tak|je|pun|nak|saya|awak|boleh|jangan|sangat|gila|lagi|kalau|dengan|untuk|yang|tu|ni|kot|kan)\b", re.I)
cp = jl(E + "casual_picks_t.jsonl"); msx = [p for p in cp if p["lang"] == "ms"]; badms = set()
if msx:
    o = subprocess.run(["/home/ubuntu/venv-ft312/bin/python", D + "ms_id_score.py"], input=json.dumps([p["answer"] for p in msx]), capture_output=True, text=True)
    badms = {p["pid"] for p, s_ in zip(msx, json.loads(o.stdout)) if s_ >= 0.15}
lah_en = 0
for p in cp:
    th = CT.get(p["pid"], "")
    if HUMAN.search(p["answer"]) or HUMAN2.search(p["answer"]): st["C_drop_human"] += 1; continue
    if p["pid"] in badms: st["C_drop_ms_indonesian"] += 1; continue
    if p["lang"] == "en" and len(MYW.findall(p["answer"])) >= 3: st["C_drop_lang_mismatch"] += 1; continue
    if p["lang"] == "en" and re.search(r"\blah?\b", p["answer"], re.I):
        lah_en += 1
        if lah_en > 38: st["C_drop_lah_cap"] += 1; continue
    if len(th) < 30 or LEAKC.search(th) or re.search(r"\[[^\]]{2,30}\]|\bQwen\b|\bLambda\b", p["prompt"] + p["answer"]): st["C_drop"] += 1; continue
    crow.append({"kind": "sft_ce", "context": R(p["prompt"], "generic" if rng.random() < 0.25 else "nosys"), "response_text": th + "\n</think>\n\n" + p["answer"] + "<|im_end|>", "pilot_id": "1810t|casual|" + p["pid"], "domain": "t_casual_warm_" + p["lang"]})
add(crow, 2.0, "N_casual_warm")
for f, tag in (("/tmp/t_td_pairs.jsonl", "N_td_pairs"), ("/tmp/t_refalt.jsonl", "N_refalt_pairs"), ("/tmp/t_comply.jsonl", "N_comply_pairs"), ("/tmp/t_cjk.jsonl", "N_cjk_pairs")): add(jl(f), 1.0, tag)
# replay of the 1810s mix
Q = [("s_shortref", 1.0), ("r_safety_", 0.7), ("s_safety_benign", 0.5), ("s_casual_voice", 0.2), ("d1810q_humour", 0.3), ("id1810_", 0.45), ("td1810_", 0.4), ("persona_mined", 1.0), ("nature_v12c_sft", 0.4),
     ("d1810q_nature2", 0.4), ("s_nature_rating", 0.4), ("nature_guard_sft", 1.0), ("cand_know", 1.0), ("malay", 1.0), ("d1810_prayer", 1.0), ("d1810_cai_warm", 1.0)]
for l in open(D + "l03_training_data_1810s.jsonl"):
    r = json.loads(l); d = str(r.get("domain") or ""); k = r.get("kind")
    keep = 0.35 if k == "ftpo_soft" else 1.0 if (not d and k is None) else next((f for p, f in Q if d.startswith(p)), 0.3)
    if rng.random() < keep: out.append(r); st["R_" + str(k)] += 1
rng.shuffle(out)
open(D + "l03_training_data_1810t.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out)); st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:22s} {st[k]}")
