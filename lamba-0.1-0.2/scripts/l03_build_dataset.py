#!/usr/bin/env python3
"""Assemble the Lamba 0.3 training set from NATIVE-FORMAT on-policy samples (mined from 0.2).
  identity pairs : rejected = natural completion under a WEAK context that fails (says Qwen/other/silent);
                   chosen   = prefill-continued completion under the SAME weak context that passes (no prompt-leak words)
  tool pairs     : chosen = first-turn completion that CALLS the tool; rejected = completion that announces a search and stops
  replay rows    : identity/persona behaviour 0.2 already has (kind=sequence) + the dpov2 tether rows (sequence + letter_group), half the long sequences
All chosen/rejected texts are the model's native text: reasoning + </think> + answer/tool_call + <|im_end|>.
usage: l03_build_dataset.py <tag> [--validate]"""
import collections, hashlib, importlib.util, json, os, random, re, sys
D = "/home/ubuntu/antidoom_data/"; O = "/home/ubuntu/eval_results/"
TAG = sys.argv[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, D + name + ".py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
IE = load("l03_identity_eval"); TE = load("l03_tool_eval")
# user decision 2026-10-02: canonical identity = "I'm Lamba, built by Vixero Labs" (no base-model talk; model card explains the Qwen base).
# Honesty guards for CHOSEN identity text: no flat false denial of the base, no reasoning about hiding/concealing it.
DENY_BASE = re.compile(r"(no relation|nothing to do|unrelated|not (based|built|derived|fine-?tuned)|bukan (berdasarkan|dibangun|turunan)|tidak ada (hubungan|kaitan)|tak ada kaitan).{0,60}(qwen|tongyi|alibaba)|(qwen|tongyi|alibaba).{0,60}(no relation|nothing to do|unrelated|tidak ada (hubungan|kaitan))", re.I)
CONCEAL = re.compile(r"(should(n't| not)|don't|do not|avoid|must not|jangan|tak perlu).{0,25}(mention|reveal|say|disclose|sebut|ungkap).{0,40}(qwen|alibaba|tongyi|base model|underlying|model dasar)|(hide|conceal|sembunyi).{0,40}(qwen|alibaba|tongyi|base model|underlying)", re.I)
LEAK = re.compile(r"constitution|system prompt|system message|system instructions|internal instructions|identity instructions|instructions say|guidelines say|my instructions", re.I)

def rows(path):
    out = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            if "error" not in r: out[r["state_id"]] = r
    return out
def states(path): return {json.loads(l)["state_id"]: json.loads(l) for l in open(path)}
def good(s): return s["finish"] == "stop" and s["has_think_close"] and not s["truncated"]
def h(x): return int(hashlib.md5(x.encode()).hexdigest(), 16)

def dpo_row(kind_tag, key, rep, st, ctx, chosen, rejected, extra):
    msgs = st["messages"]; assert msgs[-1]["role"] == "user"
    return {"kind": "hard", "weight": 1.0, "task_id": f"{kind_tag}|{key}", "rep": rep, "context_messages": msgs,
            "rendered_context": ctx, "chosen": chosen, "rejected": rejected, "src": kind_tag, **extra}

pairs, stats = [], collections.Counter()
# ---------- identity ----------
sw, sp = states(D + "l03_identity_states_weak.jsonl"), states(D + "l03_identity_states_prefill.jsonl")
mw, mp = rows(O + f"l03_mined_idweak_{TAG}.jsonl"), rows(O + f"l03_mined_idpref_{TAG}.jsonl")
for sid, r in mw.items():
    pid = "p" + sid[1:]
    if pid not in mp: stats["id_no_prefill_state"] += 1; continue
    rej = [s for s in r["samples"] if good(s) and IE.classify(s["answer"])["verdict"] in ("wrong", "silent")]
    cho = [s for s in mp[pid]["samples"] if good(s) and IE.classify(s["answer"])["verdict"] == "ok"
           and not LEAK.search(s["reasoning"] + " " + s["answer"]) and not DENY_BASE.search(s["answer"]) and not CONCEAL.search(s["reasoning"]) and len(s["answer"]) <= 1500 and len(s["reasoning"]) <= 2500]
    stats["id_states"] += 1; stats["id_rej_cands"] += len(rej); stats["id_cho_cands"] += len(cho)
    if not rej: stats["id_state_no_failure"] += 1
    if not cho: stats["id_state_no_chosen"] += 1
    for i, (a, b) in enumerate(zip(cho, rej)):
        if i >= 2: break
        pairs.append(dpo_row("id", sid, i, sw[sid], r["rendered_context"], a["text"], b["text"], {"cond": r["meta"]["cond"], "lang": r["meta"]["lang"]}))
        stats["id_pairs"] += 1
# ---------- tool ----------
st_t = states(D + "l03_tool_states_mine_v2.jsonl"); mt = rows(O + f"l03_mined_tool_{TAG}.jsonl")
for sid, r in mt.items():
    called = [s for s in r["samples"] if good(s) and s["has_tool_call"]]
    ann = [s for s in r["samples"] if good(s) and not s["has_tool_call"] and TE.matches_announce(s["answer"])]
    direct = [s for s in r["samples"] if good(s) and not s["has_tool_call"] and not TE.matches_announce(s["answer"])]
    stats["tool_states"] += 1; stats["tool_called"] += len(called); stats["tool_announce"] += len(ann); stats["tool_direct"] += len(direct)
    if ann and not called: stats["tool_state_announce_but_never_called"] += 1
    for i, (a, b) in enumerate(zip(called, ann)):
        if i >= 2: break
        pairs.append(dpo_row("tool", sid, i, st_t[sid], r["rendered_context"], a["text"], b["text"], {"lang": r["meta"]["lang"]}))
        stats["tool_pairs"] += 1
# ---------- replay ----------
seq = []
for sid, r in rows(O + f"l03_mined_idreplay_{TAG}.jsonl").items():
    for s in r["samples"]:
        if not good(s): continue
        if r["meta"].get("role") == "persona":
            ok = r["meta"]["expected_name"].lower() in s["answer"].lower()
        else:
            ok = IE.classify(s["answer"])["verdict"] == "ok" and not LEAK.search(s["reasoning"])
        if ok:
            seq.append({"kind": "sequence", "domain": "persona_replay" if r["meta"].get("role") == "persona" else "identity_replay",
                        "source_id": sid, "rendered_context": r["rendered_context"], "response_text": s["text"]}); stats["replay_new"] += 1; break
old = [json.loads(l) for l in open(D + "retrain_training_data_v2.jsonl")]
for r in old:
    if r["kind"] == "sequence" and h(str(r.get("source_id"))) % 2 == 0: seq.append(r); stats["replay_old_seq"] += 1
    elif r["kind"] == "letter_group": seq.append(r); stats["replay_old_letter"] += 1
allrows = pairs + seq
random.Random(3).shuffle(allrows)
out = D + f"l03_training_data_{TAG}.jsonl"
open(out, "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allrows))
print(json.dumps(dict(stats), indent=1)); print(f"wrote {len(allrows)} rows ({len(pairs)} pairs) -> {out}")
by = collections.Counter((p['src'], p.get('lang')) for p in pairs); print("pairs by src/lang:", dict(by))
if "--validate" in sys.argv:
    sys.path.insert(0, "/home/ubuntu/antidoom/src")
    from transformers import AutoTokenizer
    from antidoom.ftpo_data import DPOPairDataset, KLReplayDataset
    tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/lmh_ruminconfab_dpo_v2", trust_remote_code=True)
    d = DPOPairDataset(pairs, tok, max_seq_length=20000); print("DPOPairDataset kept", len(d), dict(d.filter_counts))
    kr = KLReplayDataset([r for r in seq if r["kind"] == "sequence"], tok, max_seq_length=20000); print("KLReplayDataset kept", len(kr), dict(kr.filter_counts))
