#!/usr/bin/env python3
"""Same as gen_rescue_dryrun_gate.py but using the SAME ThreadPoolExecutor concurrency
pattern as the original pcr_heldout generation (concurrency=8), to test whether batch
composition is the driver of the non-reproduction (DS #199 option b)."""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402
from gen_retrain_pool_rollouts import run_conversation_sampled, numeric_str_variants  # noqa: E402
from reclassify_pcr_failure_modes import reclassify, load_needles_and_prompts  # noqa: E402


def main():
    served_model_name = sys.argv[1]
    arm_filter = sys.argv[2]

    cases = json.load(open("/home/ubuntu/antidoom_data/never_correct_cases.json"))
    cases = [c for c in cases if c["arm"] == arm_filter][:3]

    prompts = {}
    for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
        r = json.loads(l)
        prompts[r["id"]] = r["prompt"]
    gts = load_needles_and_prompts()
    lock = threading.Lock()
    results = []

    def run_one(c):
        tid, rep = c["id"], c["rep"]
        result = run_conversation_sampled(prompts[tid], served_model_name, temperature=0.8, top_p=0.95, seed=rep)
        needles = list(numeric_str_variants(gts[tid]))
        cat = reclassify(result["transcript"], result.get("final_text", ""), needles)
        has_sandbox_event = any(
            ("__import__" in (t.get("content") or "") or "ImportError" in (t.get("content") or "")
             or "is not defined" in (t.get("content") or ""))
            for t in result["transcript"] if t.get("role") == "tool"
        )
        ok = cat == "never_correct" and has_sandbox_event
        with lock:
            print(f"DRYRUN-CONCURRENT {tid} rep={rep}: category={cat} sandbox_event={has_sandbox_event} "
                  f"n_rounds={result['n_rounds']} -> {'REPRODUCED' if ok else 'DID NOT REPRODUCE'}", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(run_one, c) for c in cases]
        oks = [f.result() for f in as_completed(futs)]

    print(f"DRYRUN_CONCURRENT_GATE_RESULT: {sum(oks)}/{len(cases)} reproduced", flush=True)
    print("DRYRUN_CONCURRENT_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
