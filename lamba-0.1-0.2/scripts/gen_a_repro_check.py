#!/usr/bin/env python3
"""Reproducibility check for the 5 known interesting/discordant prompts from the 47-prompt
A-check pool (per DS #151's suggestion): repeat each N times against a live backend to
estimate collapse probability, since the collapse behavior has already shown itself to be
stochastic (Afresh_A1_00 flipped from fail to pass on a bare re-run).
"""
import json
import sys
import time

sys.path.insert(0, "/home/ubuntu/antidoom_data")
from gen_a_fresh_toolloop import run_conversation  # noqa: E402

REPRO_IDS = ["Aexp_A1_06", "Aexp_A3_08", "Aexp_A3_09", "Aexp_A4_11", "Afresh_A1_00"]
N_REPS = 5


def main():
    if len(sys.argv) != 3:
        print("usage: gen_a_repro_check.py <served_model_name> <out.jsonl>")
        sys.exit(1)
    served_model_name, out_path = sys.argv[1], sys.argv[2]

    pool = {r["id"]: r for r in (json.loads(l) for l in open("/home/ubuntu/antidoom_data/a_check_pool.jsonl"))}

    done = set()
    try:
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["id"], r["rep"]))
    except FileNotFoundError:
        pass

    with open(out_path, "a") as f:
        for pid in REPRO_IDS:
            prompt = pool[pid]["prompt"]
            for rep in range(N_REPS):
                if (pid, rep) in done:
                    continue
                t0 = time.time()
                try:
                    result = run_conversation(prompt, served_model_name)
                    result["error"] = None
                except Exception as e:
                    result = {"final_text": "", "n_rounds": 0, "budget_hit": False,
                              "transcript": [], "error": str(e)}
                collapsed = len((result.get("final_text") or "").strip()) < 20
                row = {"id": pid, "rep": rep, "elapsed_s": round(time.time() - t0, 1),
                       "collapsed": collapsed, **result}
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{pid} rep={rep}: collapsed={collapsed} rounds={row['n_rounds']} "
                      f"elapsed={row['elapsed_s']}s", flush=True)
    print("REPRO_CHECK_DONE")


if __name__ == "__main__":
    main()
