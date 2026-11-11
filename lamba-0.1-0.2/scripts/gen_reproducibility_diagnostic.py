#!/usr/bin/env python3
"""DS #199's reproducibility diagnostic: S (sequential) vs C (8-way in-flight,
padded with filler requests since only 3 target cases exist) on the same 3 known
c10 never_correct cases, same seeds, same server session. 5 reps per condition per
case (15 case-runs/condition). Reports counts for: exact signature
(never_correct & sandbox_event), sandbox_event alone, never_correct alone,
found_correct, finish_reason, n_rounds -- not just pass/fail.
"""
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
import server as srv  # noqa: E402
from gen_retrain_pool_rollouts import run_conversation_sampled, numeric_str_variants  # noqa: E402
from reclassify_pcr_failure_modes import reclassify, load_needles_and_prompts  # noqa: E402

N_REPS = 5
TARGET_CASES = None  # filled in main()


def classify_result(tid, result, gts):
    needles = list(numeric_str_variants(gts[tid]))
    cat = reclassify(result["transcript"], result.get("final_text", ""), needles)
    has_sandbox_event = any(
        ("__import__" in (t.get("content") or "") or "ImportError" in (t.get("content") or "")
         or "is not defined" in (t.get("content") or ""))
        for t in result["transcript"] if t.get("role") == "tool"
    )
    found_correct = cat in ("success", "success_but_overran", "true_rumination", "sandbox_state_failure")
    return {
        "category": cat, "sandbox_event": has_sandbox_event, "found_correct": found_correct,
        "exact_signature": cat == "never_correct" and has_sandbox_event,
        "n_rounds": result["n_rounds"], "finish_budget_hit": result["budget_hit"],
    }


def run_one_target(served_model_name, tid, rep_seed, prompts, gts, lock, out):
    result = run_conversation_sampled(prompts[tid], served_model_name, temperature=0.8, top_p=0.95, seed=rep_seed)
    cls = classify_result(tid, result, gts)
    row = {"id": tid, "rep_seed": rep_seed, **cls}
    with lock:
        out.append(row)
    return row


def run_filler(served_model_name, tid, rep_seed, prompts):
    # discard result, just occupies a concurrency slot
    run_conversation_sampled(prompts[tid], served_model_name, temperature=0.8, top_p=0.95, seed=rep_seed + 1000)


def main():
    served_model_name = sys.argv[1]
    arm_filter = sys.argv[2]

    cases = json.load(open("/home/ubuntu/antidoom_data/never_correct_cases.json"))
    target_cases = [c for c in cases if c["arm"] == arm_filter][:3]
    all_ho_ids = [json.loads(l)["id"] for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl")]
    target_ids = {c["id"] for c in target_cases}
    filler_pool_ids = list(set(all_ho_ids) - target_ids)

    prompts = {}
    for l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
        r = json.loads(l)
        prompts[r["id"]] = r["prompt"]
    gts = load_needles_and_prompts()

    lock = threading.Lock()

    # Condition S: sequential, 5 reps per case
    s_results = []
    print("=== CONDITION S (sequential) ===", flush=True)
    for rep_i in range(N_REPS):
        for c in target_cases:
            seed = c["rep"] * 100 + rep_i  # distinct seed per rep, deterministic
            row = run_one_target(served_model_name, c["id"], seed, prompts, gts, lock, s_results)
            print(f"S rep{rep_i} {c['id']} seed={seed}: {row}", flush=True)

    # Condition C: 8-way in-flight, padded with filler requests
    c_results = []
    print("=== CONDITION C (8-way in-flight, padded) ===", flush=True)
    for rep_i in range(N_REPS):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = []
            for c in target_cases:
                seed = c["rep"] * 100 + rep_i
                futs.append(pool.submit(run_one_target, served_model_name, c["id"], seed, prompts, gts, lock, c_results))
            # pad to 8 in-flight with filler requests (results discarded)
            n_filler = 8 - len(target_cases)
            for i in range(n_filler):
                fid = filler_pool_ids[(rep_i * n_filler + i) % len(filler_pool_ids)]
                futs.append(pool.submit(run_filler, served_model_name, fid, rep_i * 17 + i, prompts))
            for f in as_completed(futs):
                f.result()
        print(f"C rep{rep_i} done, batch of {len(futs)} completed", flush=True)

    def summarize(label, results):
        n = len(results)
        exact = sum(r["exact_signature"] for r in results)
        sandbox = sum(r["sandbox_event"] for r in results)
        never_correct = sum(r["category"] == "never_correct" for r in results)
        found = sum(r["found_correct"] for r in results)
        print(f"\n{label}: n={n} exact_signature={exact}/{n} sandbox_event={sandbox}/{n} "
              f"never_correct={never_correct}/{n} found_correct={found}/{n}")
        from collections import Counter
        print(f"  category breakdown: {dict(Counter(r['category'] for r in results))}")

    summarize("CONDITION S", s_results)
    summarize("CONDITION C", c_results)
    print("\nREPRODUCIBILITY_DIAGNOSTIC_DONE", flush=True)


if __name__ == "__main__":
    main()
