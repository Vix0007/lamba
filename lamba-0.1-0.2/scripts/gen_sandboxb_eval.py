#!/usr/bin/env python3
"""Generic sandbox-B tool-loop eval runner (any prompts file, any served model). Same runner as
gen_pcr_heldout_rescue_b.py (8-round cap, forced tool_choice=none final on cap), temp 0.8 / top_p
0.95 / seed=rep. Resumable by (id, rep).
Usage: gen_sandboxb_eval.py <served_model_name> <out.jsonl> <prompts.jsonl> <reps> <concurrency> [--ids a,b,c]
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
from gen_pcr_heldout_rescue_b import run_conversation_rescue_b  # noqa: E402


def main():
    model, out_path, prompts_fn = sys.argv[1], sys.argv[2], sys.argv[3]
    reps, conc = int(sys.argv[4]), int(sys.argv[5])
    only = set(sys.argv[7].split(",")) if len(sys.argv) > 7 and sys.argv[6] == "--ids" else None
    prompts = {}
    for l in open(prompts_fn):
        r = json.loads(l)
        if only is None or r["id"] in only:
            prompts[r["id"]] = r["prompt"]
    done = set()
    try:
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["id"], r["rep"]))
    except FileNotFoundError:
        pass
    todo = [(t, rep) for t in prompts for rep in range(reps) if (t, rep) not in done]
    print(f"{len(prompts)} tasks x {reps} reps: {len(todo)} to go", flush=True)
    lock = threading.Lock()

    def run_one(tid, rep):
        try:
            res = run_conversation_rescue_b(prompts[tid], model, temperature=0.8, top_p=0.95, seed=rep)
        except Exception as e:  # don't let one failed request kill the run; logged, retried on resume
            print(f"ERR {tid} rep={rep}: {e}", flush=True)
            return
        with lock, open(out_path, "a") as f:
            f.write(json.dumps({"id": tid, "rep": rep, "model": model, "condition": "B", **res}) + "\n")

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(run_one, t, r) for t, r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 20 == 0:
                print(f"[{i}/{len(todo)}]", flush=True)
    print("GEN_SANDBOXB_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
