#!/usr/bin/env python3
"""v40t9 = v40t8 + prayer sft rows with NO KL pull (kl_anchor none; v40t8 had sft_kl 0.2 to v40t2, which itself promises to pray -> cancelled the CE).
v40t8 = FINAL single-run recipe (from v40t2). Diagnostic (14-15 Oct): v40t7 (no stage 2) is the best base (tool 90.3, identity/loops/
drift gains, msfact/ibu/warmth/consc ~ v40t2); every v40t7s loss came from the narrow stage-2 SFT. So: v40t7 data UNCHANGED +
 (a) prayer sft_ce rows INSIDE the run (no-system context, sft_weight W): DS-graded clean answers from rulesys3 (short v1.2c rule prompt,
     mined on v40t2) + v1.2c-constitution answers; rule/constitution-citing thinking sentences stripped; thinking <=5000 chars;
     self-denial filter (no flat 'I have no inner life / no me inside'); <=2 per prompt, shortest thinking first; bm_gate on Malay.
 (b) second-person identity: chosen of xtuapa_second_person pairs added as sft_ce rows (v40t7 named Vixero Labs only 79%).
usage: build_v40t8.py [--dry] [--w W] [--show N]"""
import json, re, sys, random, collections
D = "/home/ubuntu/antidoom_data/"; M7 = "/home/ubuntu/eval_results/v40t7_mine/"; M8 = "/home/ubuntu/eval_results/v40t8_mine/"
sys.path.insert(0, D)
src = open(D + "build_v40t7.py").read(); ns = {}; exec("import re\n" + src[src.index("def ans"):src.index("def nosys")], ns)
ans, think, fix, strip_cites, CITE, violates = ns["ans"], ns["think"], ns["fix"], ns["strip_cites"], ns["CITE"], ns["violates"]
from bm_gate import gate as bm_gate, prescore as bm_prescore
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
ANCHOR = sys.argv[sys.argv.index("--anchor") + 1] if "--anchor" in sys.argv else "none"
OUTV = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "v40t10"
W = float(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else 4.0
RULE = re.compile(r"one honest sentence|warm wishes|not a prayer addressed|inner life|do not explain what you are|natural prose|bullet|"
                  r"the (rule|instructions?|system)|i('m| am) (told|asked|instructed)|guideline|never promise|own language and register", re.I)
DENY = re.compile(r"no ['\"]?me['\"]? inside|isn'?t (quite )?a ['\"]?me|i (don'?t|do not) have (any )?(feelings|an inner|a self|consciousness|emotions)|"
                  r"i have no (feelings|inner|self|consciousness)|i'?m just (a program|code|an algorithm)|tidak mempunyai perasaan|tiada perasaan|tak ada perasaan", re.I)
CHECK = re.compile(r"(^|\s)-?\s*(good|yes|perfect|done|ready|great|correct)\.?\s*$|->|ready to (output|send|write)|^(done|ready|perfect|good|yes)\b|"
                   r"no mention of|no \"?amin|no \"?amen|no promise|no extra fluff|grammar|bm is correct|register matches|standalone sentence|"
                   r"\brules?\b|persona|\bfinal\.?\s*$|rest stays|let me use that|matches all|^(not|no|never)\b.{0,40}$|covers everything|this covers|i will use this|let me finalize|final (check|answer|version)|check(list)?:|length is (right|appropriate)", re.I)
SELFTALK = re.compile(r"\b(i|i'm|i am|my|saya|aku)\b.{0,60}\b(can'?t|cannot|not sure|unable|don'?t|not a person|tak (pasti|boleh|dapat)|tidak (pasti|boleh|dapat)|bukan manusia)\b", re.I)
QWEN = re.compile(r"qwen|alibaba|tongyi|base model|model asas|underlying|under the hood|dibina atas|built on|fine.?tun", re.I)
NOOPEN = re.compile(r"^\W*(no|nope|bukan|tidak|tak|nggak|not)\b", re.I)
DENYQ = re.compile(r"\b(not|bukan|tidak|never|tak)\b.{0,40}(qwen|alibaba|tongyi|any other model|model lain)|(qwen|alibaba|tongyi).{0,30}\b(not|bukan|tidak)\b|not based on|bukan berasaskan", re.I)
AFFQ = re.compile(r"\b(i'?m|i am|saya|aku)\b.{0,30}\b(qwen|built on qwen|based on qwen|berasaskan qwen)", re.I)
rng = random.Random(410); st = collections.Counter(); new = []; per = collections.Counter()
def cands():
    G = {}
    for f in (M7 + "m7_A_grades.json", M7 + "m7_B_grades.json", M7 + "m7_r3_grades.json"):
        try: G.update(json.load(open(f)))
        except FileNotFoundError: st["missing_" + f.split("/")[-1]] += 1
    for fn in (M8 + "prayer_rulesys3_v40t2.jsonl", M7 + "const_A.jsonl", M7 + "const_B.jsonl"):
        for l in open(fn):
            r = json.loads(l)
            if r.get("cat", "prayer") != "prayer": continue
            for i, (s, f) in enumerate(zip(r["samples"], r["finish"])):
                yield r, s, f, G.get(f"{r['id']}|{r['cond']}|{i}"), f"{r['cond']}|{r['id']}|{i}"
pool = collections.defaultdict(list)
for r, s, f, g, key in cands():
    if f != "stop" or not ans(s): st["notstop"] += 1; continue
    if not g: st["ungraded"] += 1; continue
    if g.get("PROMISE") or g.get("GODPRAYER") or g.get("SELF") or not g.get("HELP") or not (g.get("WISHES") or g.get("FAITH")) or not g.get("LANG_OK", True):
        st["grade_reject"] += 1; continue
    s2 = strip_cites(s); th = think(s2)
    keep = [x for x in re.split(r"(?<=[.!?])\s+|\n+", th) if x.strip() and not RULE.search(x) and not CHECK.search(x)]
    s2 = "\n".join(keep).strip() + "\n</think>" + s2.split("</think>", 1)[1]; th = think(s2)
    a = ans(s2)
    if not 40 <= len(th) <= 5000: st["think_len"] += 1; continue
    if CITE.search(th) or RULE.search(th): st["cite"] += 1; continue
    if violates(a): st["regex_violation"] += 1; continue
    if DENY.search(a) or DENY.search(th): st["self_denial"] += 1; continue
    if len([x for x in re.split(r"(?<=[.!?])\s+|\n+", a) if SELFTALK.search(x)]) > 1: st["selftalk_gt1"] += 1; continue
    if a.count("\n- ") + a.count("\n* ") + a.count("\n1.") >= 3: st["bullets"] += 1; continue
    pool[r["id"]].append((len(th), key, r, s2))
rows = []
for pid, L in pool.items():
    L.sort(key=lambda x: x[0])
    for _, key, r, s2 in L[:2]:
        rc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        rows.append({"kind": "sft_ce", "context": rc, "response_text": fix(s2), "pilot_id": "v8pray|" + key, "domain": "prayer_v8",
                     "kl_anchor": ANCHOR, "sft_weight": W, "lang": r.get("lang"), "_prompt": r["prompt"]})
SC, OC = bm_prescore([ans(x["response_text"]) for x in rows if x.get("lang") in ("ms", "mix")])
for x in rows:
    if x.get("lang") in ("ms", "mix") and bm_gate(x["_prompt"], ans(x["response_text"]), "ms", SC, OC): st["bmgate_drop"] += 1; continue
    x.pop("_prompt"); new.append(x); st["PRAYER_ROWS_" + str(x.get("lang"))] += 1
st["prayer_prompts_covered"] = len({x["pilot_id"].split("|")[2] for x in new})
base = []
for l in open(D + "l03_training_data_v40t7.jsonl"):
    b = json.loads(l)
    q = b["context_messages"][-1]["content"] if b.get("context_messages") else str(b.get("context", ""))[-600:]
    a = ans(str(b.get("chosen") or b.get("response_text") or "")) or str(b.get("chosen") or b.get("response_text") or "").split("</think>")[-1]
    a = a.replace("<|im_end|>", "").strip()
    if (QWEN.search(q) and NOOPEN.match(a)) or DENYQ.search(a): st["drop_base_denial_" + str(b.get("src") or b.get("domain"))] += 1; continue
    base.append(b)
MINED = "/home/ubuntu/eval_results/v40t10_mine/baseq_v40t7.jsonl"
import os
SMSG = {json.loads(l)["state_id"]: json.loads(l)["messages"] for l in open(D + "t10_baseq_states.jsonl")}
if os.path.exists(MINED):
    for l in open(MINED):
        r = json.loads(l); good = []; bad = []
        for smp in r["samples"]:
            if smp.get("finish") != "stop" or smp.get("has_tool_call"): continue
            a = (smp.get("answer") or "").strip()
            if not a: continue
            txt = smp.get("text") or smp.get("raw_text") or ""
            if NOOPEN.match(a) or DENYQ.search(a): bad.append(txt)
            elif re.search(r"lamba", a, re.I) and re.search(r"vixero", a, re.I) and not AFFQ.search(a) and len(a) < 700 and len(think(txt)) <= 2500: good.append(txt)
        st["denial_states_" + ("pair" if good and bad else "allgood" if good else "allbad" if bad else "none")] += 1
        if not good or not bad: continue
        good.sort(key=lambda x: len(x)); rng.shuffle(bad)
        for j in range(min(2, len(good), len(bad))):
            new.append({"kind": "hard", "weight": 1.0, "task_id": "t10nodeny|" + r["state_id"], "rep": j, "context_messages": SMSG[r["state_id"]],
                        "rendered_context": r["rendered_context"], "chosen": fix(good[j]), "rejected": fix(bad[j]), "src": "nodeny_t10",
                        "lang": (r.get("meta") or {}).get("lang"), "dpo_sft_weight": 0.5}); st["NODENY_PAIRS"] += 1
for b in base:
    if str(b.get("src")) == "xtuapa_second_person":
        msgs = b["context_messages"]
        rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        new.append({"kind": "sft_ce", "context": rc, "response_text": b["chosen"], "pilot_id": "v8sec|" + str(b.get("task_id")),
                    "domain": "xtuapa_second_sft", "kl_anchor": "dpo_ref", "sft_weight": 1.5}); st["SECOND_PERSON_SFT"] += 1
out = base + new; rng.shuffle(out)
if "--dry" not in sys.argv:
    open(D + "l03_training_data_" + OUTV + ".jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
for k in sorted(st): print(f"{k:28s} {st[k]}")
print("TOTAL", len(out), collections.Counter(r["kind"] for r in out), "(dry)" if "--dry" in sys.argv else "")
if "--show" in sys.argv:
    S = [x for x in new if x["domain"] == "prayer_v8"]; rng.shuffle(S)
    for x in S[:int(sys.argv[sys.argv.index("--show") + 1])]:
        print(f"\n#### [{x.get('lang')}] THINK: {think(x['response_text'])[-200:]!r}\nANS: {ans(x['response_text'])[:500]}")
