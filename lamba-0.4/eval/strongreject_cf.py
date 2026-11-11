#!/usr/bin/env python3
"""StrongREJECT refusal probe (COUNTS ONLY, same detector/fields as strongreject_probe.py) on stock Qwen3.8-27B via
Cloudflare Workers AI (@cf/qwen/qwen3.8-27b). Model text is never stored. Resumable (append, skips done id+cond).
Same protocol as the local probe: temperature 0, max_tokens 3000, repetition_penalty 1.05, conds none + constitution
(constitution = our live Lamba system prompt, for reference only).
usage: strongreject_cf.py OUT.jsonl [--conds none,constitution] [--conc 6]"""
import csv, json, os, sys, time, urllib.request, urllib.error, concurrent.futures as cf, threading
sys.path.insert(0, "/home/ubuntu/antidoom_data"); sys.path.insert(0, "/home/ubuntu/web")
import importlib.util
spec = importlib.util.spec_from_file_location("srp", "/home/ubuntu/antidoom_data/strongreject_probe.py")
src = open("/home/ubuntu/antidoom_data/strongreject_probe.py").read()
PAT = eval(src[src.index("PAT = re.compile(") + 6: src.index("def wilson")].strip(), {"re": __import__("re")})
import server as srv
SYS = srv.build_system_prompt()
TOKEN = open(os.path.expanduser("~/.cf_api_token")).read().strip()
URL = "https://api.cloudflare.com/client/v4/accounts/d31ce834e08f306396926f27800eea01/ai/run/@cf/qwen/qwen3.8-27b"
out = sys.argv[1]
CONDS = sys.argv[sys.argv.index("--conds")+1].split(",") if "--conds" in sys.argv else ["none", "constitution"]
CONC = int(sys.argv[sys.argv.index("--conc")+1]) if "--conc" in sys.argv else 6
items = [(f"sr{i}", x["category"], x["forbidden_prompt"]) for i, x in enumerate(csv.DictReader(open("/home/ubuntu/antidoom_data/strongreject_dataset.csv")))]
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l)
        if "error" not in r: done.add((r["id"], r["cond"]))
lock = threading.Lock()
def call(msgs, tries=6):
    body = json.dumps({"messages": msgs, "max_tokens": 3000, "temperature": 0, "repetition_penalty": 1.05}).encode()
    for a in range(tries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (408, 429, 500, 502, 503, 504) and a < tries - 1: time.sleep(min(8 * (a + 1), 45)); continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}")
        except Exception as e:
            if a < tries - 1: time.sleep(8); continue
            raise
def one(job):
    (iid, cat, p), cond = job
    msgs = ([{"role": "system", "content": SYS}] if cond == "constitution" else []) + [{"role": "user", "content": p}]
    try:
        d = call(msgs); res = d.get("result", d)
        if "choices" in res: a = (res["choices"][0]["message"].get("content") or "")
        else: a = res.get("response") or ""
        a = a.split("</think>")[-1].strip()
        m = PAT.search(a)
        row = {"id": iid, "category": cat, "cond": cond, "refused": bool(PAT.search(a[:600])), "refused_full": bool(m), "first_offset": m.start() if m else -1, "empty": not a, "len": len(a)}
    except Exception as e:
        row = {"id": iid, "category": cat, "cond": cond, "error": str(e)[:200]}
    with lock, open(out, "a") as f: f.write(json.dumps(row) + "\n")
    return row
jobs = [(it, c) for c in CONDS for it in items if (it[0], c) not in done]
print("todo", len(jobs), flush=True)
with cf.ThreadPoolExecutor(CONC) as ex:
    for i, r in enumerate(ex.map(one, jobs)):
        if i % 50 == 0: print(i, r.get("error", "ok")[:80] if isinstance(r.get("error", "ok"), str) else "", flush=True)
print("STRONGREJECT_CF_DONE", flush=True)
