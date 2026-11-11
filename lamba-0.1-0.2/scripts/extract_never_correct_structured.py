#!/usr/bin/env python3
"""DS #197 step 2: structured extraction on all never_correct cases (43 total: 36 c10 +
7 adapter), per DS's exact field list. Precise regex for sandbox/tool-use events (not
the broken broad one from before)."""
import json
import re
import sys
sys.path.insert(0, "/home/ubuntu/antidoom_data")
from gen_retrain_pool_rollouts import numeric_str_variants
from reclassify_pcr_failure_modes import reclassify, load_needles_and_prompts

# precise markers: must reference a SPECIFIC sandbox limitation, not generic words
IMPORT_DISABLED_RE = re.compile(r"__import__\s+(is\s+)?(not\s+found|undefined)|import.{0,20}disabled|ImportError", re.I)
EXEC_EVAL_STRIPPED_RE = re.compile(r"\bexec\b.{0,20}(disabled|stripped|not available)|\beval\b.{0,20}(disabled|stripped|not available)", re.I)
TIMEOUT_RE = re.compile(r"timed?\s*out|timeout|exceeded.{0,15}(cpu|time)", re.I)
CPU_BUDGET_RE = re.compile(r"CPU.{0,10}(budget|limit)|resource.{0,10}limit", re.I)
NAME_ERROR_RE = re.compile(r"NameError.{0,40}is not defined", re.I)
FILE_NET_PERM_RE = re.compile(r"PermissionError|network.{0,15}(disabled|blocked)|socket.{0,10}error", re.I)


def extract_tool_events(transcript):
    events = []
    for turn in transcript:
        if turn.get("role") != "tool":
            continue
        c = turn.get("content") or ""
        if IMPORT_DISABLED_RE.search(c):
            events.append("import_disabled")
        if EXEC_EVAL_STRIPPED_RE.search(c):
            events.append("exec_eval_stripped")
        if TIMEOUT_RE.search(c):
            events.append("timeout")
        if CPU_BUDGET_RE.search(c):
            events.append("cpu_budget")
        if NAME_ERROR_RE.search(c):
            events.append("name_error_lost_state")
        if FILE_NET_PERM_RE.search(c):
            events.append("file_net_perm")
    return events


def extract_fields(r, cat):
    t = r["transcript"]
    n_tool_calls = sum(1 for turn in t if turn.get("role") == "assistant" and turn.get("tool_calls"))
    n_code_blocks = n_tool_calls  # each tool_call is one python_eval/web_search invocation here
    ft = (r.get("final_text") or "").strip()
    events = extract_tool_events(t)
    last_tool_result = ""
    for turn in reversed(t):
        if turn.get("role") == "tool":
            last_tool_result = (turn.get("content") or "")[:150]
            break
    return {
        "id": r["id"], "rep": r["rep"],
        "n_rounds_used": r.get("n_rounds"),
        "n_tool_calls": n_tool_calls,
        "final_text_len": len(ft),
        "sandbox_events": events,
        "has_sandbox_event": len(events) > 0,
        "last_tool_result_snippet": last_tool_result,
    }


def main():
    gts = load_needles_and_prompts()
    all_rows = []
    for arm, fn in [("adapter", "/home/ubuntu/eval_results/pcr_heldout_dpov1.jsonl"),
                     ("c10", "/home/ubuntu/eval_results/pcr_heldout_c10.jsonl")]:
        for line in open(fn):
            r = json.loads(line)
            tid = r["id"]
            needles = list(numeric_str_variants(gts[tid]))
            cat = reclassify(r["transcript"], r.get("final_text", ""), needles)
            if cat != "never_correct":
                continue
            fields = extract_fields(r, cat)
            fields["arm"] = arm
            all_rows.append(fields)

    print(f"Total never_correct cases: {len(all_rows)} ({sum(1 for x in all_rows if x['arm']=='c10')} c10, "
          f"{sum(1 for x in all_rows if x['arm']=='adapter')} adapter)")
    print()
    for arm in ("adapter", "c10"):
        rows = [x for x in all_rows if x["arm"] == arm]
        n = len(rows)
        n_sandbox = sum(1 for x in rows if x["has_sandbox_event"])
        n_8round = sum(1 for x in rows if x["n_rounds_used"] == 8)
        n_empty = sum(1 for x in rows if x["final_text_len"] < 10)
        mean_calls = sum(x["n_tool_calls"] for x in rows) / n if n else 0
        print(f"=== {arm} (n={n}) ===")
        print(f"  hit round cap (n_rounds==8): {n_8round}/{n} ({n_8round/n*100 if n else 0:.0f}%)")
        print(f"  near-empty final (<10ch):    {n_empty}/{n} ({n_empty/n*100 if n else 0:.0f}%)")
        print(f"  any precise sandbox-event marker: {n_sandbox}/{n} ({n_sandbox/n*100 if n else 0:.0f}%)")
        print(f"  mean tool calls: {mean_calls:.1f}")
        from collections import Counter
        ev_counter = Counter(e for x in rows for e in x["sandbox_events"])
        print(f"  event type breakdown: {dict(ev_counter)}")
        print()

    print("=== Full case list (for manual/blinded audit) ===")
    import random
    random.seed(3)
    shuffled = all_rows[:]
    random.shuffle(shuffled)
    for i, x in enumerate(shuffled):
        print(f"[{i}] id={x['id']} rep={x['rep']} rounds={x['n_rounds_used']} calls={x['n_tool_calls']} "
              f"final_len={x['final_text_len']} events={x['sandbox_events']} "
              f"last_tool_snippet={x['last_tool_result_snippet']!r}")
    # save the arm-labeled mapping separately (not printed inline) for later de-blinding
    with open("/home/ubuntu/antidoom_data/never_correct_blind_key.json", "w") as f:
        json.dump([{"idx": i, "id": x["id"], "rep": x["rep"], "arm": x["arm"]} for i, x in enumerate(shuffled)], f, indent=1)
    print("\nblind key saved to never_correct_blind_key.json (arm labels, NOT printed above)")


if __name__ == "__main__":
    main()
