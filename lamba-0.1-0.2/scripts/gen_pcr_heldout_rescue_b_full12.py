#!/usr/bin/env python3
"""Full-12-task confirmatory replication: dpov1 + sandbox-B on ALL 12 reserved held-out
tasks (not just the 7 dpov1 previously failed on). Establishes the combined fix's overall
never_correct/pcr rate on a clean, never-trained-on set -- complements the earlier
rescue-of-failures-only test. Reuses the exact same conversation runner as
gen_pcr_heldout_rescue_b.py (real tool_choice=none forced-final on cap-hit, sandbox-B).
Usage: gen_pcr_heldout_rescue_b_full12.py <served_model_name> <out.jsonl> <reps> <concurrency>
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
from gen_pcr_heldout_rescue_b import run_conversation_rescue_b  # noqa: E402


def main():
    served_model_name = sys.argv[1]
    out_path = sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    concurrency = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    prompts = {}
    for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
        r = json.loads(l)
        prompts[r["id"]] = r["prompt"]

    todo = [(tid, rep) for tid in prompts for rep in range(reps)]
    print(f"running {len(todo)} draws ({len(prompts)} held-out tasks x {reps} reps)", flush=True)

    lock = threading.Lock()

    def run_one(tid, rep):
        result = run_conversation_rescue_b(prompts[tid], served_model_name, temperature=0.8, top_p=0.95, seed=rep)
        row = {"id": tid, "rep": rep, "arm": "adapter", "condition": "B_full12", **result}
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return tid, rep

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(run_one, tid, rep) for tid, rep in todo]
        n = 0
        for fut in as_completed(futs):
            tid, rep = fut.result()
            n += 1
            print(f"[{n}/{len(todo)}] {tid} rep={rep} done", flush=True)

    print("GEN_PCR_RESCUE_B_FULL12_DONE", flush=True)


if __name__ == "__main__":
    main()
