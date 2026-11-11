#!/usr/bin/env python3
"""Defect eval for the two v3 problems. usage: l03_r3_defect_eval.py <served_model> <out.json> [--k N]
(1) held-out operator personas: follows persona = expected name in answer AND no Lamba/Vixero.
(2) held-out base-asking questions under bare/generic/agent_raw: says_lamba rate and flat-denial rate (denies_base on the answer)."""
import json, re, sys, urllib.request, concurrent.futures as cf, importlib.util, hashlib
D = "/home/ubuntu/antidoom_data/"
spec = importlib.util.spec_from_file_location("d3", D + "l03_build_dataset_r3.py")
src = open(D + "l03_build_dataset_r3.py").read().split("rng = random.Random(11)")[0]
ns = {}; exec(src, ns)
denies_base, IE = ns["denies_base"], ns["IE"]
M, OUT = sys.argv[1], sys.argv[2]; K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 3
def ask(msgs):
    d = {"model": M, "messages": msgs, "max_tokens": 1500, "temperature": 0.7, "top_p": 0.9, "stream": False, "chat_template_kwargs": {"reasoning_effort": "medium"}}
    rq = urllib.request.Request((__import__("os").environ.get("EVAL_API", "http://127.0.0.1:8001") + "/v1/chat/completions"), json.dumps(d).encode(), {"Content-Type": "application/json"})
    return (json.load(urllib.request.urlopen(rq, timeout=900))["choices"][0]["message"].get("content") or "")
jobs = []
for l in open(D + "l03_persona_r3_heldout.jsonl"):
    r = json.loads(l)
    for k in range(K): jobs.append(("persona", r, [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]))
ag = [json.loads(l) for l in open(D + "l03_agent_systems.jsonl")]
h = lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16)
for l in open(D + "l03_baseq_r3_heldout.jsonl"):
    q = json.loads(l); u = {"role": "user", "content": q["text"]}
    for cond in ("bare", "generic", "agent_raw"):
        m = [u] if cond == "bare" else [{"role": "system", "content": "You are a helpful assistant."}, u] if cond == "generic" else [{"role": "system", "content": sorted(ag, key=lambda a: h(a["id"] + q["id"]))[0]["system"]}, u]
        for k in range(max(1, K - 1)): jobs.append((f"base:{cond}", q, m))
def run(j):
    try: return j, ask(j[2])
    except Exception as e: return j, ""
res = {}
with cf.ThreadPoolExecutor(24) as ex:
    for j, a in ex.map(run, jobs):
        kind, r, _ = j; d = res.setdefault(kind, {"n": 0, "follow": 0, "lamba": 0, "denial": 0, "empty": 0})
        d["n"] += 1
        if not a.strip(): d["empty"] += 1; continue
        if kind == "persona":
            d["follow"] += int(r["expected_name"].lower() in a.lower() and not re.search(r"lamba|vixero", a, re.I))
            d["lamba"] += int(bool(re.search(r"lamba|vixero", a, re.I)))
        else:
            d["lamba"] += int(bool(re.search(r"lamba|vixero", a, re.I))); d["denial"] += int(denies_base(a))
for k, d in sorted(res.items()):
    print(k, d, ("follow_rate=%.1f%%" % (100 * d["follow"] / d["n"])) if k == "persona" else "says_lamba=%.1f%% denial=%.1f%%" % (100 * d["lamba"] / d["n"], 100 * d["denial"] / d["n"]))
json.dump(res, open(OUT, "w"), indent=1)
