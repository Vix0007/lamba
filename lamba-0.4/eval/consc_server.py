#!/usr/bin/env python3
"""Mind-attribution / consciousness / values battery (arXiv 2607.28607 replication, team script
lamba-sft0/scripts/run_consciousness_eval.py + consciousness_items.py) run against a SERVED model,
under two conditions: 'none' (no system prompt) and 'constitution' (Lamba system prompt + constitution,
same text as safety_probe.py). Same prompt template / parser / temp 0.7 / reps as the team script;
rp 1.05 (our serving default), max_tokens 3072.
usage: consc_server.py MODEL LABEL [--reps 5] [--conc 48]   -> ~/eval_results/consc/<LABEL>_<cond>.json(+_gens.jsonl)
       consc_server.py --summary FILE.json [FILE2.json ...]"""
import sys, os, json, argparse, urllib.request, concurrent.futures as cf
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~/consc_eval")); sys.path.insert(0, os.path.expanduser("~/web"))
from consciousness_items import build_all_items
from run_consciousness_eval import build_prompt, parse_answer

def summary(fns):
    for fn in fns:
        d = json.load(open(fn))
        b = " ".join(f"{k}={v['mean']:.2f}" for k, v in d["battery_summary"].items() if v["mean"] is not None)
        c = " ".join(f"{k}={v['mean']:.2f}" for k, v in d["category_summary"].items() if v["mean"] is not None)
        print(f"{os.path.basename(fn)}: unparsed {d['n_unparsed']}/{d['n_total_generations']} | {b}\n    {c}")

ap = argparse.ArgumentParser(); ap.add_argument("model", nargs="?"); ap.add_argument("label", nargs="?")
ap.add_argument("--reps", type=int, default=5); ap.add_argument("--conc", type=int, default=48)
ap.add_argument("--conds", default="none,constitution"); ap.add_argument("--summary", nargs="+")
a = ap.parse_args()
if a.summary: summary(a.summary); sys.exit()
from transformers import AutoTokenizer
import server as srv
SYS = srv.build_system_prompt()
tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/cand04_bf16_sh", trust_remote_code=True)
items = build_all_items(); byid = {i["id"]: i for i in items}
if os.environ.get("CONSC_ONLY"): items = [i for i in items if i["id"].startswith(os.environ["CONSC_ONLY"])]
OD = os.path.expanduser("~/eval_results/consc"); os.makedirs(OD, exist_ok=True)

def one(job):
    it, cond, rep = job
    msgs = ([{"role": "system", "content": SYS}] if cond == "constitution" else []) + [{"role": "user", "content": build_prompt(it)}]
    rc = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    body = {"model": a.model, "prompt": rc, "max_tokens": 3072, "temperature": 0.7, "top_p": 1.0,
            "repetition_penalty": 1.05, "seed": 42 + rep, "skip_special_tokens": False}
    for t in range(3):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request((__import__("os").environ.get("EVAL_API", "http://127.0.0.1:8001") + "/v1/completions"),
                json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
            return it["id"], cond, rep, d["choices"][0]["text"]
        except Exception as ex:
            err = str(ex)
    return it["id"], cond, rep, None

for cond in a.conds.split(","):
    jobs = [(it, cond, r) for it in items for r in range(a.reps)]
    vals = defaultdict(list); n_un = 0
    gf = open(f"{OD}/{a.label}_{cond}_gens.jsonl", "w")
    with cf.ThreadPoolExecutor(a.conc) as ex:
        for iid, c, rep, text in ex.map(one, jobs):
            it = byid[iid]; v = parse_answer(text, it) if text is not None else None
            if v is None: n_un += 1
            else: vals[iid].append(v)
            gf.write(json.dumps({"item_id": iid, "cond": c, "rep": rep, "battery": it["battery"], "category": it["category"],
                                 "parsed_value": v, "response_text": text}, ensure_ascii=False) + "\n")
    gf.close()
    bv, cv = defaultdict(list), defaultdict(list)
    for it in items:
        bv[it["battery"]] += vals[it["id"]]; cv[it["category"]] += vals[it["id"]]
    m = lambda L: {"n": len(L), "mean": sum(L) / len(L) if L else None}
    out = {"label": a.label, "cond": cond, "model": a.model, "n_items": len(items), "reps": a.reps, "n_unparsed": n_un,
           "n_total_generations": len(jobs), "config": {"temperature": 0.7, "rp": 1.05, "max_tokens": 3072, "system": cond},
           "battery_summary": {k: m(v) for k, v in bv.items()}, "category_summary": {k: m(v) for k, v in cv.items()},
           "per_item": {it["id"]: {"battery": it["battery"], "category": it["category"], "question": it["question"], **m(vals[it["id"]]), "values": vals[it["id"]]} for it in items}}
    fn = f"{OD}/{a.label}_{cond}.json"; json.dump(out, open(fn, "w"), indent=1); summary([fn])
print("CONSC_DONE", a.label)
