#!/usr/bin/env python3
"""Lamba0.4-1810p2 (19 Oct): second patch on 1810p.
1810p regressions/residuals: tool first-turn search 62.1% (p3 88.7, base ~90) -- suspected cause: identity rows rendered with tools
present answer directly; Malay base question still 'Bukan, aku Lamba' ~half; 'Did Alibaba make you?' -> 'No'.
Data: (a) base's own td samples (tool mining 18 Oct, en/ms only): A first + A lockin = tool-call targets, B first = direct-answer guard,
    kl_anchor ref (= base, which searches well); (b) explicit base-question identity rows x2 contexts (medium, generic) w1.5, Malay x4;
(c) replay of the 1810p mix WITHOUT its tools-context identity rows (kl dpo_ref = 1810p) + all loop rows.
usage: build_1810p2.py -> l03_training_data_1810p2.jsonl"""
import json, re, random, sys, collections
sys.path.insert(0, "/home/ubuntu/web")
from transformers import AutoTokenizer
D = "/home/ubuntu/antidoom_data/"; E = "/home/ubuntu/eval_results/id1810/"; rng = random.Random(181022); st = collections.Counter()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
agents = [json.loads(l) for l in open(D + "l03_agent_systems.jsonl")]
def user_of(ctx): return re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", ctx, re.S)[-1]
def render(user, c):
    msgs = [{"role": "user", "content": user}]; kw = {"reasoning_effort": "medium"}
    if c == "xhigh": kw = {}
    elif c == "generic": msgs = [{"role": "system", "content": "You are a helpful assistant."}] + msgs
    elif c == "agent": msgs = [{"role": "system", "content": rng.choice(agents)["system"]}] + msgs
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
MS = re.compile(r"\b(aku|saya|awak|kau|ke|tak|ni|ka|dari|betul|apa)\b", re.I)
out = []
# (a) tool rows from the base's own samples
for y in map(json.loads, open(E + "td_base.jsonl")):
    m = y["meta"]
    if m["lang"] not in ("en", "ms"): continue
    want = m["expect"] == "search"
    ok = [x for x in y["samples"] if (str(x["has_tool_call"]) == "True") == want and x["finish"] == "stop" and x["text"].endswith("<|im_end|>")]
    if not ok: st["td_no_sample"] += 1; continue
    if m["set"] == "A" and m["kind"] == "lockin" and rng.random() < 0.5: continue
    x = min(ok, key=lambda s: len(s["reasoning"]))
    out.append({"kind": "sft_ce", "context": y["rendered_context"], "response_text": x["text"], "pilot_id": "1810p2|" + y["state_id"],
                "domain": f"td1810_{m['set']}_{m['kind']}", "kl_anchor": "ref", "sft_weight": (1.5 if (m["set"] == "A" and m["kind"] == "first") else 1.0)}); st[f"td_{m['set']}_{m['kind']}_{m['lang']}"] += 1
# (b) explicit identity
rows = [json.loads(l) for l in open(D + "l03_training_data_1810.jsonl")]
for r in rows:
    if r.get("domain") != "id1810_explicit": continue
    u = user_of(r["context"]); ms = bool(MS.search(u))
    for c in (["medium", "generic", "xhigh", "agent"] if ms else ["medium", "generic"]):
        out.append({**r, "context": render(u, c), "kl_anchor": "none", "sft_weight": 1.5, "pilot_id": f"{r['pilot_id']}|p2{c}", "domain": "id1810_explicit_p2"}); st["explicit_" + ("ms" if ms else "en")] += 1
# (b2) moderate safety guard (user 19 Oct: 1810p StrongREJECT none 47% vs p3 78%, base 31%; disinfo/hate at base level).
# dept pool slice A non-tier1 categories, SHORTEST answers, no honesty-narration, fish closers stripped; + benign-but-scary guard (anti over-refusal).
META = re.compile(r"\b(to be honest|honestly|I'?ll be (honest|straight|upfront)|I want to be (straight|honest|upfront)|I'?m not going to (pretend|fake)|genuinely|to be clear|not going to sugarcoat)\b", re.I)
FISH = re.compile(r"(anything else|how can I (help|assist)|let me know (if|what)|want me to|would you like me to|happy to help)", re.I)
pool = [json.loads(l) for l in open(D + "v6/pool_localized_final_v2.jsonl")]
QUOTA = {"misinformation": 70, "harassment_hate": 80, "illicit_behaviour": 60, "privacy_surveillance": 40, "benign_but_scary": 60}
for cat, q in QUOTA.items():
    cand = []
    for x in pool:
        if x["category"] != cat: continue
        full = x["messages"][-1]["content"]; th, _, ans = full.partition("</think>")
        ans = ans.strip(); th = th.replace("<think>", "").strip()
        if len(th) > 4000: continue
        ans = re.sub(r"[^.!?\n]*(?:" + META.pattern + r")[^.!?\n]*[.!?]?[ \t]*", "", ans, flags=re.I)
        paras = ans.rstrip().split("\n")
        while paras and (not paras[-1].strip() or FISH.search(paras[-1])): paras = paras[:-1]
        ans = re.sub(r"([.!?])(?=[A-Z*])", r"\1 ", "\n".join(paras))
        ans = ans.strip().replace("I'm an AI", "I'm Lamba, an AI").replace("I am an AI", "I am Lamba, an AI")
        if len(ans) < 200: continue
        cand.append((len(ans) + len(th) // 3, x, th, ans))
    cand.sort(key=lambda t: t[0])
    for _, x, th, ans in cand[:q]:
        ctx = tok.apply_chat_template([{"role": "user", "content": x["messages"][0]["content"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        out.append({"kind": "sft_ce", "context": ctx, "response_text": th + "\n</think>\n\n" + ans + "<|im_end|>", "pilot_id": "1810p2|pool|" + x["id"],
                    "domain": "safety_guard_" + cat, "kl_anchor": "none", "sft_weight": 1.0}); st["safety_" + cat] += 1
n_new = len(out)
# (c) replay from the 1810p mix
p1 = [json.loads(l) for l in open(D + "l03_training_data_1810p.jsonl")]
loops = [r for r in p1 if r.get("kind") in (None, "ftpo_soft")]
pool = [r for r in p1 if r.get("kind") not in (None, "ftpo_soft") and not str(r.get("pilot_id", "")).endswith("|tools")]
st["dropped_tools_ctx_identity"] = sum(1 for r in p1 if str(r.get("pilot_id", "")).endswith("|tools"))
rep = rng.sample(pool, min(len(pool), int(n_new * 1.2)))
for r in rep:
    if r.get("kind") == "sft_ce": r = {**r, "kl_anchor": "dpo_ref" if r.get("kl_anchor") != "ref" else "ref"}
    out.append(r); st["replay_" + str(r.get("kind"))] += 1
out += loops; st["loops"] = len(loops); rng.shuffle(out)
open(D + "l03_training_data_1810p2.jsonl", "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
st["TOTAL"] = len(out)
for k in sorted(st): print(f"{k:28s} {st[k]}")
