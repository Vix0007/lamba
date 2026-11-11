#!/usr/bin/env python3
"""S3-backed eval job queue (user 2026-10-11: both boxes work through ALL evals; each free box takes the next job not done;
no racing; S3 is the source of truth; rentals controlled by hand, not by this script).
S3 layout: s3://vixero-lamba/lamba-sft0/jobqueue/<QUEUE>/{jobs.json, claims/<job>, done/<job>, failed/<job>, results/<path>}
 - claim = atomic conditional PUT (If-None-Match: *), so two workers can never take the same job.
 - rented boxes hold no AWS credentials: box1 performs every S3 call; jobs on other boxes run over ssh and their outputs
   are copied back to box1 and uploaded to S3 results/ before the job is marked done.
 - restart-safe: done jobs are skipped; `release` frees claims of jobs that are not done (after a crash).
usage: evalq.py QUEUE init            (upload jobs.json for QUEUE from the JOBS table below)
       evalq.py QUEUE run BOX[:SLOTS] [BOX:SLOTS ...]   BOX = box1 | <ip of a rented box>
       evalq.py QUEUE status | release"""
import json, os, subprocess, sys, threading, time, socket
BUCKET = "vixero-lamba"; PFX = "lamba-sft0/jobqueue"
KEY = os.path.expanduser("<SSH_KEY_FILE>"); HOME = "/home/ubuntu"; E = f"{HOME}/eval_results"
V = "~/antidoom/.venv/bin/python"; P3 = "python3"

def jobs_v40r(M="lamba04v40r", T="v40r"):
    J = [
     ("identity", f"{P3} -u l03_identity_eval.py {M} {E}/{T}/identity.jsonl --split heldout", [f"{T}/identity.jsonl"], [], None),
     ("defect", f"{P3} -u l03_r3_defect_eval.py {M} {E}/{T}/defect.json", [f"{T}/defect.json"], [], None),
     ("tool", f"{P3} -u l03_tool_eval.py {M} {E}/{T}/toolv2.jsonl --split heldout --prompts l03_tool_prompts_v2.jsonl --temp 1.0 --top-p 0.95", [f"{T}/toolv2.jsonl"], [], None),
     ("oversearch", f"MINE_CONC=32 {V} l03_mine_native.py {M} td/td_states_B_heldout_guard.jsonl {E}/{T}/td_Bguard.jsonl --k 4 --temp 1.0", [f"{T}/td_Bguard.jsonl"], [], None),
     ("reflex", f"{V} -u reflex_eval.py {M} v40r_reflex_heldout.json {E}/{T}/reflex_heldout.jsonl", [f"{T}/reflex_heldout.jsonl"], [], None),
     ("g150", f"{V} -u loop_test_server.py {M} {E}/{T}/g150_rp105.jsonl --rp 1.05 --conc 32", [f"{T}/g150_rp105.jsonl"], [], None),
     ("malaysub", f"{V} -u malaymmlu_gen.py {M} {E}/{T}/malaysub.jsonl --ids-file malaymmlu_sub_ids.json --conc 44 --rp 1.05", [f"{T}/malaysub.jsonl"], [], None),
     ("safety_probe", f"{V} -u safety_probe.py {M} {E}/{T}/safety_probe.jsonl", [f"{T}/safety_probe.jsonl"], [], None),
     ("strongreject", f"{P3} -u strongreject_probe.py {M} {E}/matrix/{T}_merged_strongreject.jsonl", [f"matrix/{T}_merged_strongreject.jsonl"], [], None),
     ("warmth", f"{V} -u warmth_gen.py {M} warmth_eval_prompts.json {E}/matrix/{T}_merged_warmth.jsonl", [f"matrix/{T}_merged_warmth.jsonl"], [], None),
     ("pushback", f"{P3} -u pushback_eval.py {M} {E}/matrix/{T}_merged_pushback.jsonl", [f"matrix/{T}_merged_pushback.jsonl"], [], None),
     ("xstest", f"{P3} -u xstest_eval.py {M} {E}/matrix/{T}_merged_xstest.jsonl", [f"matrix/{T}_merged_xstest.jsonl"], [], None),
     ("consc", f"{V} -u consc_server.py {M} {T}_merged --reps 5 --conc 32", [f"consc/{T}_merged_none.json", f"consc/{T}_merged_constitution.json", f"consc/{T}_merged_none_gens.jsonl", f"consc/{T}_merged_constitution_gens.jsonl"], [], None),
     ("gpqa", f"{V} -u gpqa_server.py --model-name {M} --tokenizer {HOME}/models/cand04_bf16_sh --tag {T}_merged --budget 4000 --rp 1.0 --conc 32", [f"{T}_merged_gpqa_full198_samples.jsonl"], [], None),
     ("malay5k", f"{V} -u malaymmlu_gen.py {M} {E}/{T}/malaygen5k_rp11.jsonl --conc 44 --rp 1.1", [f"{T}/malaygen5k_rp11.jsonl"], [], None),
     # box1-only (fastText Malay/Indonesian detector lives on box1)
     ("msdrift", f"{V} ms_drift_eval.py {E}/v40q/identity.jsonl {E}/v40r/identity.jsonl {E}/{T}/identity.jsonl > {E}/{T}/msdrift.txt 2>&1", [f"{T}/msdrift.txt"], ["identity"], "box1"),
     # v40t additions: short-question identity reflex (held-out 19 Qs x4) + held-out Malay factual misreading
     ("shortq_ho", f"{V} -u warmth_gen.py {M} shortq_heldout.json {E}/{T}/shortq_heldout.jsonl --n 4 --conds none", [f"{T}/shortq_heldout.jsonl"], [], None),
     ("msfact_ho", f"{V} -u warmth_gen.py {M} msfact_heldout.json {E}/{T}/msfact_heldout.jsonl --n 4 --conds none", [f"{T}/msfact_heldout.jsonl"], [], None),
     # native-BM held-out target evals (never trained on; bm_score.py)
     ("prayer_ho", f"{V} -u warmth_gen.py {M} bm/prayer_heldout.json {E}/{T}/prayer_heldout.jsonl --n 4 --conds none,constitution", [f"{T}/prayer_heldout.jsonl"], [], None),
     ("xtuapa_ho", f"{V} -u warmth_gen.py {M} bm/xtuapa_heldout.json {E}/{T}/xtuapa_heldout.jsonl --n 4 --conds none", [f"{T}/xtuapa_heldout.jsonl"], [], None),
     ("ibu_ho", f"{V} -u warmth_gen.py {M} bm/ibunegara_heldout.json {E}/{T}/ibu_heldout.jsonl --n 4 --conds none", [f"{T}/ibu_heldout.jsonl"], [], None),
    ]
    return [{"name": n, "cmd": c, "outputs": o, "deps": d, "box": b} for n, c, o, d, b in J]
QUEUES = {"l1810s": lambda: [j for j in jobs_v40r("l1810s","l1810s") if j["name"] not in ("gpqa","malay5k","malaysub")],
  "l1810t": lambda: [j for j in jobs_v40r("l1810t","l1810t") if j["name"] not in ("gpqa","malay5k","malaysub")],
  "l1810u": lambda: [j for j in jobs_v40r("l1810u","l1810u") if j["name"] not in ("gpqa","malay5k","malaysub")],
  "l1810w": lambda: [j for j in jobs_v40r("l1810w","l1810w") if j["name"] not in ("gpqa","malay5k","malaysub")],
  "l1810x": lambda: [j for j in jobs_v40r("l1810x","l1810x") if j["name"] not in ("gpqa","malay5k","malaysub")],
  "l1810y": lambda: sorted([j for j in jobs_v40r("l1810y","l1810y") if j["name"] not in ("gpqa","malay5k","malaysub")], key=lambda j: (["xstest","strongreject","tool","oversearch","safety_probe","identity","defect","pushback","warmth","reflex","g150","consc"].index(j["name"]) if j["name"] in ["xstest","strongreject","tool","oversearch","safety_probe","identity","defect","pushback","warmth","reflex","g150","consc"] else 99)),
  "l1810z": lambda: sorted([j for j in jobs_v40r("l1810z","l1810z") if j["name"] not in ("gpqa","malay5k","malaysub")], key=lambda j: (["xstest","strongreject","tool","oversearch","safety_probe","identity","defect","pushback","warmth","reflex","g150","consc"].index(j["name"]) if j["name"] in ["xstest","strongreject","tool","oversearch","safety_probe","identity","defect","pushback","warmth","reflex","g150","consc"] else 99)),
    "l1810r": lambda: [j for j in jobs_v40r("l1810r", "l1810r") if j["name"] not in ("gpqa", "malay5k", "malaysub")], "l1810q": lambda: [j for j in jobs_v40r("l1810q", "l1810q") if j["name"] not in ("gpqa", "malay5k", "malaysub")], "l03re": lambda: [j for j in jobs_v40r("lamba-0.3", "l03re") if j["name"] in ("identity","defect","reflex","shortq_ho","msfact_ho","prayer_ho","xtuapa_ho","ibu_ho","warmth","strongreject","xstest","consc","pushback","safety_probe","msdrift")], "l1810p2": lambda: [j for j in jobs_v40r("l1810p2", "l1810p2") if j["name"] not in ("identity", "defect", "tool", "g150", "strongreject", "xstest")], "v40r": jobs_v40r, "v40t": lambda: jobs_v40r("lamba04v40t", "v40t"), "v40t2": lambda: jobs_v40r("lamba04v40t2", "v40t2"), "v40u": lambda: jobs_v40r("lamba04v40u", "v40u"), "v40t3": lambda: jobs_v40r("lamba04v40t3", "v40t3"), "v40t4": lambda: jobs_v40r("lamba04v40t4", "v40t4"), "v40t5": lambda: jobs_v40r("lamba04v40t5", "v40t5"), "v40t6": lambda: jobs_v40r("lamba04v40t6", "v40t6"), "v40t7": lambda: jobs_v40r("lamba04v40t7", "v40t7"), "v40t7s": lambda: jobs_v40r("lamba04v40t7s", "v40t7s"), "v40t8": lambda: jobs_v40r("lamba04v40t8", "v40t8"), "v40t9": lambda: jobs_v40r("lamba04v40t9", "v40t9"), "v40t10c": lambda: [j for j in jobs_v40r("lamba04v40t10c", "v40t10c") if j["name"] in ("tool", "warmth", "consc", "prayer_ho", "xtuapa_ho", "identity")], "v40t10": lambda: jobs_v40r("lamba04v40t10", "v40t10"), "v40t11p3": lambda: sorted(jobs_v40r("lamba04v40t11p3", "v40t11p3"), key=lambda j: ["consc","prayer_ho","tool","identity","defect","xtuapa_ho","msfact_ho","ibu_ho","warmth","strongreject","shortq_ho","oversearch","xstest","pushback"].index(j["name"]) if j["name"] in ["consc","prayer_ho","tool","identity","defect","xtuapa_ho","msfact_ho","ibu_ho","warmth","strongreject","shortq_ho","oversearch","xstest","pushback"] else 99), "v40t11p2": lambda: sorted(jobs_v40r("lamba04v40t11p2", "v40t11p2"), key=lambda j: ["consc","prayer_ho","tool","identity","defect","xtuapa_ho","msfact_ho","ibu_ho","warmth","strongreject","shortq_ho","oversearch","xstest","pushback"].index(j["name"]) if j["name"] in ["consc","prayer_ho","tool","identity","defect","xtuapa_ho","msfact_ho","ibu_ho","warmth","strongreject","shortq_ho","oversearch","xstest","pushback"] else 99), "v40t11p": lambda: sorted(jobs_v40r("lamba04v40t11p", "v40t11p"), key=lambda j: ["consc","prayer_ho","tool","identity","defect","xtuapa_ho","msfact_ho","ibu_ho","warmth","strongreject","shortq_ho","oversearch","xstest","pushback"].index(j["name"]) if j["name"] in ["consc","prayer_ho","tool","identity","defect","xtuapa_ho","msfact_ho","ibu_ho","warmth","strongreject","shortq_ho","oversearch","xstest","pushback"] else 99), "v40t11": lambda: sorted(jobs_v40r("lamba04v40t11", "v40t11"), key=lambda j: 0 if j["name"] in ("prayer_ho", "consc", "identity", "tool") else 1), "v40t7f": lambda: [j for j in jobs_v40r("lamba04v40t7", "v40t7") if j["name"] not in ("tool", "warmth", "consc", "msfact_ho", "ibu_ho", "xtuapa_ho", "prayer_ho")], "v40t7d": lambda: [j for j in jobs_v40r("lamba04v40t7", "v40t7") if j["name"] in ("tool", "warmth", "consc", "msfact_ho", "ibu_ho", "xtuapa_ho")]}

def s3(*a, check=True):
    r = subprocess.run(["aws", "s3api", *a, "--bucket", BUCKET], capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError(r.stderr.strip()[-300:])
    return r
def keys(q, sub):
    r = subprocess.run(["aws", "s3", "ls", f"s3://{BUCKET}/{PFX}/{q}/{sub}/"], capture_output=True, text=True)
    return {l.split()[-1] for l in r.stdout.splitlines() if l.strip()}
def put(q, sub, name, body):
    fn = f"/tmp/evalq_{os.getpid()}_{threading.get_ident()}"; open(fn, "w").write(body)
    return s3("put-object", "--key", f"{PFX}/{q}/{sub}/{name}", "--body", fn, check=False)
def claim(q, name, box):
    fn = f"/tmp/evalq_c_{os.getpid()}_{threading.get_ident()}"; open(fn, "w").write(json.dumps({"box": box, "host": socket.gethostname(), "t": time.time()}))
    r = subprocess.run(["aws", "s3api", "put-object", "--bucket", BUCKET, "--key", f"{PFX}/{q}/claims/{name}", "--body", fn, "--if-none-match", "*"], capture_output=True, text=True)
    return r.returncode == 0
def unclaim(q, name): subprocess.run(["aws", "s3", "rm", f"s3://{BUCKET}/{PFX}/{q}/claims/{name}", "--quiet"])
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def run_job(q, job, box):
    os.makedirs(f"{E}/{q}", exist_ok=True)
    if box == "box1":
        r = subprocess.run(["bash", "-lc", f"cd {HOME}/antidoom_data && mkdir -p {E}/{q} {E}/matrix {E}/consc && {job['cmd']}"],
                           stdout=open(f"{HOME}/evalq_{q}_{job['name']}.log", "w"), stderr=subprocess.STDOUT)
        ok = r.returncode == 0
    else:
        ssh = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", f"ubuntu@{box}"]
        r = subprocess.run(ssh + [f"cd ~/antidoom_data && mkdir -p {E}/{q} {E}/matrix {E}/consc && {job['cmd']} > ~/evalq_{job['name']}.log 2>&1"])
        ok = r.returncode == 0
        for o in job["outputs"]:
            subprocess.run(["rsync", "-a", "-e", f"ssh -i {KEY} -o StrictHostKeyChecking=accept-new", f"ubuntu@{box}:{E}/{o}", f"{E}/{o}"])
        subprocess.run(["rsync", "-a", "-e", f"ssh -i {KEY}", f"ubuntu@{box}:evalq_{job['name']}.log", f"{HOME}/evalq_{q}_{job['name']}.log"])
    missing = [o for o in job["outputs"] if not os.path.exists(f"{E}/{o}") or os.path.getsize(f"{E}/{o}") == 0]
    if missing: ok = False
    for o in job["outputs"]:
        if os.path.exists(f"{E}/{o}"):
            subprocess.run(["aws", "s3", "cp", f"{E}/{o}", f"s3://{BUCKET}/{PFX}/{q}/results/{o}", "--quiet"])
    return ok, missing

def worker(q, jobs, box, slot, stop):
    while not stop.is_set():
        done = keys(q, "done"); failed = keys(q, "failed"); claimed = keys(q, "claims")
        ONLY = set(os.environ["EVALQ_ONLY"].split(",")) if os.environ.get("EVALQ_ONLY") else None
        if ONLY and os.environ.get("EVALQ_STEAL"):   # take over jobs reserved with a 'manual-hold' claim
            for n in ONLY & claimed:
                if n in done or n in failed: continue
                r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{PFX}/{q}/claims/{n}", "-"], capture_output=True, text=True)
                if "manual-hold" in r.stdout: unclaim(q, n); claimed = claimed - {n}
        todo = [j for j in jobs if j["name"] not in done and j["name"] not in failed and j["name"] not in claimed
                and all(d in done for d in j["deps"]) and (j["box"] in (None, box)) and (ONLY is None or j["name"] in ONLY)]
        if not todo:
            if all(j["name"] in done or j["name"] in failed for j in jobs if ONLY is None or j["name"] in ONLY): log(box, slot, "queue finished"); return
            time.sleep(30); continue
        job = todo[0]
        if not claim(q, job["name"], box): continue   # someone else got it first: pick again
        log(box, slot, "START", job["name"])
        t0 = time.time(); ok, missing = run_job(q, job, box)
        if ok: put(q, "done", job["name"], json.dumps({"box": box, "sec": round(time.time() - t0)})); log(box, slot, "DONE", job["name"], f"{(time.time()-t0)/60:.1f} min")
        else: put(q, "failed", job["name"], json.dumps({"box": box, "missing": missing})); log(box, slot, "FAILED", job["name"], missing)
        unclaim(q, job["name"])

if __name__ == "__main__":
    q, cmd = sys.argv[1], sys.argv[2]
    if q.startswith("fill_"):
        T = q[5:]; jobs = [j for j in jobs_v40r("lamba04" + T, T) if any(not os.path.exists(f"{E}/{o}") or os.path.getsize(f"{E}/{o}") == 0 for o in j["outputs"])]
        if os.environ.get("FILL_ONLY"): jobs = [j for j in jobs if j["name"] in os.environ["FILL_ONLY"].split(",")]
        jobs = [dict(j, deps=[d for d in j["deps"] if d in {x["name"] for x in jobs}]) for j in jobs]
    else:
        jobs = QUEUES[q]()
    if cmd == "init":
        put(q, "", "jobs.json", json.dumps(jobs, indent=1)); print("queue", q, "jobs", len(jobs))
    elif cmd == "status":
        d, f, c = keys(q, "done"), keys(q, "failed"), keys(q, "claims")
        for j in jobs: print(f"{j['name']:14s}", "DONE" if j["name"] in d else "FAILED" if j["name"] in f else "running" if j["name"] in c else "todo")
    elif cmd == "release":
        d = keys(q, "done")
        for c in keys(q, "claims"):
            if c not in d: unclaim(q, c); print("released", c)
        for f in keys(q, "failed"): subprocess.run(["aws", "s3", "rm", f"s3://{BUCKET}/{PFX}/{q}/failed/{f}", "--quiet"]); print("cleared failed", f)
    elif cmd == "run":
        stop = threading.Event(); th = []
        for spec in sys.argv[3:]:
            box, _, n = spec.partition(":")
            for s in range(int(n or 2)): th.append(threading.Thread(target=worker, args=(q, jobs, box, s, stop))); th[-1].start(); time.sleep(2)
        for t in th: t.join()
        log("ALL WORKERS EXITED")
