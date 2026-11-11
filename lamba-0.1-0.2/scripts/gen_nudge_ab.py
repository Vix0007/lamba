#!/usr/bin/env python3
"""Nudge-guard v0.5 online A/B (DS #162/#163). For each of the 7 integral-reconciliation
prompts (nudge_ab_prompts.jsonl) x N_REPS, runs the SAME prompt through mode='shadow'
(control: identical behavior to 'off', but logs fire status -- so we know which
conversations WOULD have fired) and mode='nudge' (treatment: real intervention on fire).
Since decoding is temperature=0.0 (deterministic up to float non-determinism), shadow and
nudge follow the same trajectory up to the fire point, then diverge exactly there -- a
clean matched-pair natural experiment, per DS #163's sequencing note.
Resumable via (id, rep, mode) already in the output files.
Usage: gen_nudge_ab.py <served_model_name> <out_shadow.jsonl> <out_nudge.jsonl> <n_reps>
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/antidoom_data")
from gen_a_fresh_toolloop_guarded import run_conversation_guarded  # noqa: E402

WRITE_LOCK = threading.Lock()


def load_done(path):
    done = set()
    try:
        for l in open(path):
            r = json.loads(l)
            done.add((r["base_id"], r["rep"]))
    except FileNotFoundError:
        pass
    return done


def run_one(served_model_name, mode, p, rep):
    t0 = time.time()
    try:
        result = run_conversation_guarded(p["prompt"], served_model_name, mode=mode)
        result["error"] = None
    except Exception as e:
        result = {"final_text": "", "n_rounds": 0, "budget_hit": False,
                  "transcript": [], "error": str(e), "guard_mode": mode}
    row = {"base_id": p["id"], "rep": rep, "ground_truth_numeric": p["ground_truth_numeric"],
           "elapsed_s": round(time.time() - t0, 1), **result}
    return row


def main():
    if len(sys.argv) not in (5, 6, 7):
        print("usage: gen_nudge_ab.py <served_model_name> <out_shadow.jsonl> <out_nudge.jsonl> <n_reps> "
              "[concurrency] [prompts_file]")
        sys.exit(1)
    served_model_name, out_shadow, out_nudge, n_reps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    concurrency = int(sys.argv[5]) if len(sys.argv) >= 6 else 6
    prompts_file = sys.argv[6] if len(sys.argv) == 7 else "/home/ubuntu/antidoom_data/nudge_ab_prompts.jsonl"

    prompts = [json.loads(l) for l in open(prompts_file)]
    done_shadow = load_done(out_shadow)
    done_nudge = load_done(out_nudge)

    tasks = []
    for p in prompts:
        for rep in range(n_reps):
            key = (p["id"], rep)
            if key not in done_shadow:
                tasks.append(("shadow", p, rep))
            if key not in done_nudge:
                tasks.append(("nudge", p, rep))

    print(f"{len(tasks)} (mode,prompt,rep) tasks to go (concurrency={concurrency})", flush=True)
    fs = open(out_shadow, "a")
    fn = open(out_nudge, "a")
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {pool.submit(run_one, served_model_name, mode, p, rep): (mode, p, rep)
                    for mode, p, rep in tasks}
            n_done = 0
            for fut in as_completed(futs):
                mode, p, rep = futs[fut]
                row = fut.result()
                target = fs if mode == "shadow" else fn
                with WRITE_LOCK:
                    target.write(json.dumps(row) + "\n")
                    target.flush()
                    n_done += 1
                    print(f"[{n_done}/{len(tasks)}] [{mode}] {p['id']} rep={rep}: fired={row.get('guard_fired')} "
                          f"intervened={row.get('guard_intervened')} rounds={row['n_rounds']} "
                          f"elapsed={row['elapsed_s']}s", flush=True)
    finally:
        fs.close()
        fn.close()
    print("NUDGE_AB_DONE")


if __name__ == "__main__":
    main()
