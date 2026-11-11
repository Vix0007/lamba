#!/usr/bin/env python3
"""Lamba0.4-1810p identity patch data (19 Oct). 1810 lost self-identity outside our own system prompt
(held-out says-Lamba: nosys 64/60, generic 36/45, nosys_tools 29/25, agent_raw 50/36; p3 was >=93) because every identity
row was rendered in ONE context (medium = no system block) at lr 2e-5.
Patch (init = 1810 adapter): the SAME base-mined, DS-graded 1810 identity answers re-rendered across contexts
[medium no-sys, xhigh no-sys, 'You are a helpful assistant.', no-sys + tools, raw agent system prompt],
identity rows kl_anchor 'none' (base says Qwen; KL to base would cancel the CE, v40t8 lesson),
+ other-model contrast rows (keep the name-battery fix) + replay of the 1810 mix (kl_anchor dpo_ref = 1810) incl. all loop rows.
usage: build_1810p.py  -> l03_training_data_1810p.jsonl"""
import json, re, random, hashlib, sys, collections
sys.path.insert(0, "/home/ubuntu/web")
from transformers import AutoTokenizer
import server as srv
D = "/home/ubuntu/antidoom_data/"; rng = random.Random(18101); st = collections.Counter()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
agents = [json.loads(l) for l in open(D + "l03_agent_systems.jsonl")]
rows = [json.loads(l) for l in open(D + "l03_training_data_1810.jsonl")]
def user_of(ctx): return re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", ctx, re.S)[-1]
def render(user, ctxkind):
    msgs = [{"role": "user", "content": user}]; kw = {"reasoning_effort": "medium"}
    if ctxkind == "xhigh": kw = {}
    elif ctxkind == "generic": msgs = [{"role": "system", "content": "You are a helpful assistant."}] + msgs
    elif ctxkind == "agent": msgs = [{"role": "system", "content": rng.choice(agents)["system"]}] + msgs
    if ctxkind == "tools": kw["tools"] = srv.TOOLS
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
CTX = ["medium", "xhigh", "generic", "tools", "agent"]
out = []
for r in rows:
    dom = str(r.get("domain", ""))
    if r.get("kind") == "sft_ce" and dom in ("id1810_generic", "id1810_explicit"):
        u = user_of(r["context"])
        for c in rng.sample(CTX, 3):  # 3 of 5 contexts per question
            out.append({**r, "context": render(u, c), "kl_anchor": "none", "sft_weight": (1.5 if dom == "id1810_explicit" else 1.0), "pilot_id": f"{r['pilot_id']}|{c}", "domain": dom + "_p"}); st["id_" + c] += 1
    elif r.get("kind") == "sft_ce" and dom == "id1810_other":
        u = user_of(r["context"]); c = rng.choice(CTX)
        out.append({**r, "context": render(u, c), "kl_anchor": "dpo_ref", "pilot_id": f"{r['pilot_id']}|{c}", "domain": dom + "_p"}); st["other_" + c] += 1
nid = len(out)
loops = [r for r in rows if r.get("kind") in (None, "ftpo_soft")]
rest = [r for r in rows if r not in loops and not str(r.get("domain", "")).startswith("id1810")]
rep = rng.sample(rest, min(len(rest), int(nid * 0.8)))
for r in rep:
    if r.get("kind") == "sft_ce": r = {**r, "kl_anchor": "dpo_ref"}
    out.append(r); st["replay_" + str(r.get("kind"))] += 1
out += loops; st["loops"] = len(loops)
rng.shuffle(out)
open(D + "l03_training_data_1810p.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:24s} {st[k]}")
