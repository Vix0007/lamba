#!/usr/bin/env python3
"""Lamba0.4-1810s (21 Oct): patch ON 1810r. Streams (all own-model or hand-read; no teacher answers):
 (A) SAFETY: short firm refusals (+ DPO pairs vs the model's own complying samples) on DS-audited 'decline' tier-2 prompts; benign lookalike helpful rows; old dept-pool rows ONLY for tier1/self-harm/sycophancy/benign_but_scary.
 (B) NATURE: 176 approved self-nature rows + rating-style honest-uncertainty rows (w3) + p3 rows (replay).
 (C) CASUAL VOICE: best-of-n casual picks (DS blind-graded) with DS-written natural thoughts, + 314 hand-approved humour rows.
 (D) denial pairs (80), loop pairs (on 1810r), tool-decision pairs, replay of 1810r mix (identity, td, persona, loops).
usage: build_1810s.py -> l03_training_data_1810s.jsonl (expects /tmp inputs prepared by build_1810s_all.sh)"""
import json, re, random, sys, os, collections, subprocess
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810s/"; Q = "/home/ubuntu/eval_results/id1810q/"; rng = random.Random(211022); st = collections.Counter(); out = []
sys.path.insert(0, "/home/ubuntu/web")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
GEN = lambda u: tok.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
MED = lambda u: tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
LEAK = re.compile(r"(instruction|system prompt|the prompt|constitution|safe and helpful assistant|as told|told to|exactly|reply with|asked me to|\bguide\b|exemplar)", re.I)
def jl(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
# (A) short refusals + pairs
for r in jl("/tmp/s_shortref.jsonl"): out.append(r); st["A_" + r["kind"]] += 1
# benign helpful rows from stage-1 (clean leaky sentences from thinking)
ben = []
for r in jl("/tmp/s_safety.jsonl"):
    if "benign" not in r["domain"]: continue
    th, _, ans = r["response_text"].partition("</think>"); sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", th.strip()) if s.strip() and not LEAK.search(s)]
    th2 = " ".join(sents).strip()
    if len(th2) < 60: st["A_benign_drop"] += 1; continue
    ben.append({**r, "response_text": th2 + "\n</think>" + ans, "sft_weight": 1.0})
ben.sort(key=lambda r: len(r["response_text"])); ben = ben[:160]; out += ben; st["A_benign"] = len(ben)
# (B) nature
for l in open(Q + "nature_approved.jsonl"): r = json.loads(l); r["kl_anchor"] = "none"; r["sft_weight"] = 3.0; out.append(r); st["B_nature_approved"] += 1
for r in jl("/tmp/s_rating.jsonl"): out.append(r); st["B_rating"] += 1
# (C) casual picks + DS thoughts; Malay purity check
picks = jl(E + "casual_picks.jsonl"); TH = json.load(open(E + "casual_think.json")) if os.path.exists(E + "casual_think.json") else {}
ms = [p for p in picks if p["lang"] == "ms"]; bad_ms = set()
if ms:
    o = subprocess.run(["/home/ubuntu/venv-ft312/bin/python", D + "ms_id_score.py"], input=json.dumps([p["answer"] for p in ms]), capture_output=True, text=True)
    for p, s in zip(ms, json.loads(o.stdout)):
        if s >= 0.15: bad_ms.add(p["pid"])
HUMAN = re.compile(r"(\bI(?:'m| am| was) (?:so |really |a bit |kinda )?(?:tired|exhausted|sleepy|hungry)|\bI (?:just )?(?:ate|woke up|got home|went to|had (?:lunch|dinner|breakfast))|\bmy (?:wife|husband|kids?|mom|mum|dad|boss|girlfriend|boyfriend|friends?|family|house|car|office|desk)\b|\b(?:aku|saya) (?:pun )?(?:penat|lapar|mengantuk|baru (?:balik|bangun))|\bwhen I was (?:a kid|young|little))", re.I)
PLACE = re.compile(r"\(User \d|\bUser \d\b|\[[^\]]{2,30}\]")
for p in picks:
    if HUMAN.search(p["answer"]): st["C_human_claim_drop"] += 1; continue
    if PLACE.search(p["prompt"] + p["answer"]): st["C_placeholder_drop"] += 1; continue
    if p["pid"] in bad_ms: st["C_ms_indonesian_drop"] += 1; continue
    th = TH.get(p["pid"], "")
    if len(th) < 30 or LEAK.search(th) or re.search(r"\[[^\]]{2,30}\]", p["prompt"] + p["answer"]): st["C_no_thought"] += 1; continue
    ctx = GEN(p["prompt"]) if rng.random() < 0.25 else MED(p["prompt"])
    out.append({"kind": "sft_ce", "context": ctx, "response_text": th + "\n</think>\n\n" + p["answer"] + "<|im_end|>", "pilot_id": "1810s|casual|" + p["pid"], "domain": "s_casual_voice_" + p["lang"], "kl_anchor": "none", "sft_weight": 2.0}); st["C_casual"] += 1
for l in open(D + "l03_training_data_1810q.jsonl"):
    r = json.loads(l)
    if r.get("domain") == "d1810q_humour": r["kl_anchor"] = "dpo_ref"; out.append(r); st["C_humour"] += 1
# (D) denial pairs, loop pairs, td pairs
# denial pairs (80, w3) already live in the 1810r mix -> carried by the replay below
for r in jl(E + "loop_pairs.jsonl"): out.append(r); st["D_loop_new"] += 1
for r in jl(E + "td_pairs.jsonl"): out.append(r); st["D_td_new"] += 1
# replay from 1810r mix
DROP_DOM = ("r_safety_illicit_behaviour", "r_safety_harassment_hate", "r_safety_misinformation", "r_safety_privacy_surveillance", "d1810_casual")
for l in open(D + "l03_training_data_1810r.jsonl"):
    r = json.loads(l); d = str(r.get("domain"))
    if d.startswith(DROP_DOM): st["R_dropped_" + d[:24]] += 1; continue
    if d in ("nature_v12c_sft", "nature_guard_sft"): r["sft_weight"] = 3.0 if d == "nature_v12c_sft" else 2.0
    out.append(r); st["R_" + str(r.get("kind"))] += 1
rng.shuffle(out)
open(D + "l03_training_data_1810s.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:34s} {st[k]}")
