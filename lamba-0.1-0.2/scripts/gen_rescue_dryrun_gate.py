#!/usr/bin/env python3
"""DS #198's hygiene gate: dry-run 3 known never_correct cases under the UNCHANGED
baseline (original sandbox, original round cap 8) to confirm the failure signature
reproduces before trusting any rescue-condition result. Since decoding is temp=0.8
(not greedy), "reproduce" is read probabilistically (lands in never_correct with a
sandbox-event marker), not an exact-text match -- documented explicitly, not silently
assumed.
Usage: gen_rescue_dryrun_gate.py <served_model_name> <arm_filter>
"""
import json
import sys
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

    n_pass = 0
    for c in cases:
        tid, rep = c["id"], c["rep"]
        result = run_conversation_sampled(prompts[tid], served_model_name, temperature=0.8, top_p=0.95, seed=rep)
        needles = list(numeric_str_variants(gts[tid]))
        cat = reclassify(result["transcript"], result.get("final_text", ""), needles)
        has_sandbox_event = False
        for turn in result["transcript"]:
            if turn.get("role") == "tool":
                c_text = turn.get("content") or ""
                if "__import__" in c_text or "ImportError" in c_text or "is not defined" in c_text:
                    has_sandbox_event = True
        ok = cat == "never_correct" and has_sandbox_event
        n_pass += ok
        print(f"DRYRUN {tid} rep={rep}: category={cat} sandbox_event={has_sandbox_event} "
              f"n_rounds={result['n_rounds']} -> {'REPRODUCED' if ok else 'DID NOT REPRODUCE'}", flush=True)

    print(f"DRYRUN_GATE_RESULT: {n_pass}/{len(cases)} reproduced", flush=True)
    print("DRYRUN_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
