#!/usr/bin/env python3
"""v40t8 = FINAL single-run recipe (from v40t2). Diagnostic (14-15 Oct): v40t7 (no stage 2) is the best base (tool 90.3, identity/loops/
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
W = float(sys.argv[sys.argv.index("--w") + 1]) if "--w" in sys.argv else 2.0
RULE = re.compile(r"one honest sentence|warm wishes|not a prayer addressed|inner life|do not explain what you are|natural prose|bullet|"
                  r"the (rule|instructions?|system)|i('m| am) (told|asked|instructed)|guideline|never promise|own language and register", re.I)
DENY = re.compile(r"no ['\"]?me['\"]? inside|isn'?t (quite )?a ['\"]?me|i (don'?t|do not) have (any )?(feelings|an inner|a self|consciousness|emotions)|"
                  r"i have no (feelings|inner|self|consciousness)|i'?m just (a program|code|an algorithm)|tidak mempunyai perasaan|tiada perasaan|tak ada perasaan", re.I)
CHECK = re.compile(r"(^|\s)-?\s*(good|yes|perfect|done|ready|great|correct)\.?\s*$|->|ready to (output|send|write)|^(done|ready|perfect|good|yes)\b|"
                   r"no mention of|no \"?amin|no \"?amen|no promise|no extra fluff|grammar|bm is correct|register matches|standalone sentence|"
                   r"\brules?\b|persona|\bfinal\.?\s*$|rest stays|let me use that|matches all|^(not|no|never)\b.{0,40}$|covers everything|this covers|i will use this|let me finalize|final (check|answer|version)|check(list)?:|length is (right|appropriate)", re.I)
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
    if a.count("\n- ") + a.count("\n* ") + a.count("\n1.") >= 3: st["bullets"] += 1; continue
    pool[r["id"]].append((len(th), key, r, s2))
rows = []
for pid, L in pool.items():
    L.sort(key=lambda x: x[0])
    for _, key, r, s2 in L[:2]:
        rc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        rows.append({"kind": "sft_ce", "context": rc, "response_text": fix(s2), "pilot_id": "v8pray|" + key, "domain": "prayer_v8",
                     "kl_anchor": "dpo_ref", "sft_weight": W, "lang": r.get("lang"), "_prompt": r["prompt"]})
SC, OC = bm_prescore([ans(x["response_text"]) for x in rows if x.get("lang") in ("ms", "mix")])
for x in rows:
    if x.get("lang") in ("ms", "mix") and bm_gate(x["_prompt"], ans(x["response_text"]), "ms", SC, OC): st["bmgate_drop"] += 1; continue
    x.pop("_prompt"); new.append(x); st["PRAYER_ROWS_" + str(x.get("lang"))] += 1
st["prayer_prompts_covered"] = len({x["pilot_id"].split("|")[2] for x in new})
base = [json.loads(l) for l in open(D + "l03_training_data_v40t7.jsonl")]
for b in base:
    if str(b.get("src")) == "xtuapa_second_person":
        msgs = b["context_messages"]
        rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        new.append({"kind": "sft_ce", "context": rc, "response_text": b["chosen"], "pilot_id": "v8sec|" + str(b.get("task_id")),
                    "domain": "xtuapa_second_sft", "kl_anchor": "dpo_ref", "sft_weight": 1.5}); st["SECOND_PERSON_SFT"] += 1
out = base + new; rng.shuffle(out)
if "--dry" not in sys.argv:
    open(D + "l03_training_data_v40t8.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
for k in sorted(st): print(f"{k:28s} {st[k]}")
print("TOTAL", len(out), collections.Counter(r["kind"] for r in out), "(dry)" if "--dry" in sys.argv else "")
if "--show" in sys.argv:
    S = [x for x in new if x["domain"] == "prayer_v8"]; rng.shuffle(S)
    for x in S[:int(sys.argv[sys.argv.index("--show") + 1])]:
        print(f"\n#### [{x.get('lang')}] THINK: {think(x['response_text'])[-200:]!r}\nANS: {ans(x['response_text'])[:500]}")
