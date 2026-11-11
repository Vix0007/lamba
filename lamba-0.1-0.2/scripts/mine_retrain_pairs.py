#!/usr/bin/env python3
"""Mines DPO training pairs from retrain_pool_s1337.jsonl + retrain_pool_nudge_completions.jsonl,
per DS #168-171's filtering rules:
  1. chosen (nudge answer) must be sympy-verified correct.
  2. rejected (shadow's real continuation past the fire point) must NOT be correct, or must
     be a genuine redundant continuation of an already-verified value without new info.
  3. HARD-EXCLUDE unverified_no_new_info_x2 fires from hard pairs by default (DS's stated
     default, no manual review capacity available) -- only verified_or_confirmed_no_new_info
     fires are used.
  4. Cap 3 pairs/task, cap any single integrand class at 15% of the training set.
  5. Nudge scaffold never appears in context (context = real transcript prefix only, matches
     what was actually sent to the model before the nudge -- the nudge text itself is never
     part of context, only conditions the completion).
Output: retrain_dpo_pairs_v1.jsonl (kind=hard, chosen=nudge text, rejected=shadow's real
continuation past the fire point).
"""
import json
import re
from collections import defaultdict


def strip_commas(t):
    return re.sub(r"(?<=\d),(?=\d)", "", t or "")


def numeric_str_variants(v, ndigits=6):
    s = f"{v:.{ndigits}f}"
    out = {s}
    # BUG FIX (2026-10-01): 1-decimal form was previously gated on near-integer-ness,
    # so a clean value like 2.5 never got its natural short needle. See
    # gen_retrain_pool_rollouts.py for the full note.
    for n in range(1, ndigits):
        out.add(f"{v:.{n}f}")
    if abs(v - round(v)) < 1e-9:
        iv = int(round(v))
        out.add(str(iv))
        out.add(f"{iv:,}")
    return out


def contains_value(text, needles):
    if not text:
        return False
    t2 = strip_commas(text)
    return any(n in text or n in t2 for n in needles)


def extract_shadow_rejected_continuation(transcript, fire_round):
    """Shadow's actual continuation past the fire point (what it did instead of
    stopping) -- the real redundant behavior being trained against."""
    tool_round_seen = -1
    collecting = False
    lines = []
    for turn in transcript:
        role = turn.get("role")
        if role == "tool":
            tool_round_seen += 1
            if tool_round_seen == fire_round:
                collecting = True
                continue
        if collecting:
            if role == "assistant":
                c = turn.get("content") or ""
                r = turn.get("reasoning") or ""
                lines.append(c if c else (r if r else "(redundant tool call, no text)"))
            elif role == "assistant_forced_final":
                lines.append(turn.get("content") or "(empty final)")
    return "\n".join(lines) if lines else "(no further content -- empty final)"


def build_context(prompt_text, transcript, fire_round):
    lines = [f"[USER PROMPT]\n{prompt_text}\n"]
    tool_round_seen = -1
    for turn in transcript:
        role = turn.get("role")
        if role == "assistant":
            block = "[ASSISTANT]\n"
            if turn.get("reasoning"):
                block += f"(reasoning) {turn['reasoning']}\n"
            if turn.get("content"):
                block += turn["content"] + "\n"
            if turn.get("tool_calls"):
                for tc in turn["tool_calls"]:
                    fn = tc.get("function", {})
                    block += f"[TOOL_CALL {fn.get('name')}] {fn.get('arguments')}\n"
            lines.append(block)
        elif role == "tool":
            lines.append(f"[TOOL_RESULT]\n{turn.get('content', '')}\n")
            tool_round_seen += 1
            if tool_round_seen == fire_round:
                break
    return "\n".join(lines)


def build_real_messages_context(prompt_text, transcript, fire_round):
    """The REAL model-format context (OpenAI-style messages list, real tool_call ids/
    arguments, no pretty-printing) -- what the model actually saw, truncated to the tool
    result at fire_round. This is what training/tokenization must use (via
    tokenizer.apply_chat_template), NOT build_context()'s human-readable summary above
    (that one is for DS review only -- found as a real bug: it was almost used as the
    DPO training context, which would have trained on text in the wrong format entirely,
    never seen at serving time). Reuses the exact tool_call_id reconstruction already
    validated in gen_retrain_pool_nudge_completions.py's build_prefix_to_fire_point."""
    import sys
    sys.path.insert(0, "/home/ubuntu/web")
    import server as srv  # noqa: E402 (only needed for build_system_prompt; safe to import here)
    messages = [
        {"role": "system", "content": srv.build_system_prompt()},
        {"role": "user", "content": prompt_text},
    ]
    last_ids = []
    tool_round_seen = -1
    for turn in transcript:
        role = turn.get("role")
        if role == "assistant" and turn.get("tool_calls"):
            messages.append({"role": "assistant", "content": turn.get("content") or "",
                              "tool_calls": turn["tool_calls"]})
            last_ids = [tc["id"] for tc in turn["tool_calls"]]
        elif role == "tool":
            tcid = turn.get("tool_call_id") or (last_ids.pop(0) if last_ids else "")
            messages.append({"role": "tool", "tool_call_id": tcid, "content": turn.get("content", "")})
            tool_round_seen += 1
            if tool_round_seen == fire_round:
                break
    return messages


import os

ROLLOUT_FILES = [
    "/home/ubuntu/eval_results/retrain_pool_s1337.jsonl",       # original 52-task pool
    "/home/ubuntu/eval_results/retrain_pool_s1337_v2.jsonl",    # new 41-task pool
]
NUDGE_FILES = [
    "/home/ubuntu/eval_results/retrain_pool_nudge_completions.jsonl",     # original pool's fires
    "/home/ubuntu/eval_results/retrain_pool_nudge_completions_v2.jsonl",  # new pool's fires
]
PROMPT_FILES = [
    "/home/ubuntu/antidoom_data/retrain_pool_integrals.jsonl",       # original 52
    "/home/ubuntu/antidoom_data/expanded_integral_tasks_v2.jsonl",   # new 41
]
# CRITICAL: the 12-task nudge-guard held-out generalization set (HO_*) is the reserved
# go/no-go eval split for this retrain (per DS #168-172's locked-in plan: "freeze a
# never-trained-on test split before mining -- the 12-task held-out set ... is the
# natural held-out test"). It was accidentally folded into retrain_pool_integrals.jsonl's
# task pool (52 = original 7 + held-out 12 + 33 new) and has been leaking into every
# mined pair set so far (25/206 pairs at cap=3, 33/295 at cap=6, ~12%). Hard-excluded here.
HELDOUT_EVAL_IDS = set()
if os.path.exists("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
    for _l in open("/home/ubuntu/antidoom_data/nudge_ab_heldout_prompts.jsonl"):
        HELDOUT_EVAL_IDS.add(json.loads(_l)["id"])


def main():
    rollouts = {}
    n_heldout_skipped = 0
    for fn in ROLLOUT_FILES:
        if not os.path.exists(fn):
            continue
        n_before = len(rollouts)
        for l in open(fn):
            r = json.loads(l)
            if r["id"] in HELDOUT_EVAL_IDS:
                n_heldout_skipped += 1
                continue
            rollouts[(r["id"], r["rep"])] = r
        print(f"  loaded {len(rollouts) - n_before} rollout rows from {fn}")
    nudge = {}
    for fn in NUDGE_FILES:
        if not os.path.exists(fn):
            continue
        n_before = len(nudge)
        for l in open(fn):
            r = json.loads(l)
            if r["id"] in HELDOUT_EVAL_IDS:
                continue
            nudge[(r["id"], r["rep"])] = r
        print(f"  loaded {len(nudge) - n_before} nudge-completion rows from {fn}")
    print(f"  EXCLUDED {n_heldout_skipped} rollout rows belonging to the {len(HELDOUT_EVAL_IDS)} "
          f"reserved held-out eval task ids (never eligible for training data)")
    prompts = {}
    task_class = {}
    task_gt = {}
    for fn in PROMPT_FILES:
        if not os.path.exists(fn):
            continue
        for l in open(fn):
            rec = json.loads(l)
            if rec["id"] in HELDOUT_EVAL_IDS:
                continue
            prompts[rec["id"]] = rec["prompt"]
            task_class[rec["id"]] = rec.get("integrand_class")
            task_gt[rec["id"]] = rec["ground_truth_numeric"]
    print(f"  {len(prompts)} distinct eligible-for-training tasks (held-out eval ids excluded) "
          f"across {len(PROMPT_FILES)} prompt files, "
          f"{len(rollouts)} rollouts, {len(nudge)} nudge completions")

    candidates = []
    n_excluded_unverified_soft = 0
    n_nudge_wrong = 0
    for (tid, rep), r in rollouts.items():
        if not r.get("guard_fired"):
            continue
        is_trustworthy = r["guard_fire_reason"] == "verified_or_confirmed_no_new_info"
        n = nudge.get((tid, rep))
        if n is None or n.get("error"):
            continue
        gt = task_gt.get(tid)
        if gt is None:
            continue
        needles = numeric_str_variants(gt)
        chosen_text = n.get("nudge_final_text", "")
        chosen_correct = contains_value(chosen_text, needles)
        if not chosen_correct:
            n_nudge_wrong += 1
            continue
        rejected_text = extract_shadow_rejected_continuation(r["transcript"], r["guard_fire_round"])
        rejected_correct = contains_value(rejected_text, needles)
        if not is_trustworthy:
            # DS #172: unverified_no_new_info_x2 fires -- HARD PAIRS ONLY (shadow must be
            # sympy-incorrect), reduced weight 0.4, no soft pairs from this pathway
            # (manual-review substitute unavailable, so exclude the riskier soft case).
            if rejected_correct:
                n_excluded_unverified_soft += 1
                continue
            kind, weight = "hard", 0.4
        elif rejected_correct:
            kind, weight = "soft", 1.0
        else:
            kind, weight = "hard", 1.0
        context = build_context(prompts[tid], r["transcript"], r["guard_fire_round"])
        context_messages = build_real_messages_context(prompts[tid], r["transcript"], r["guard_fire_round"])
        candidates.append({
            "kind": kind, "weight": weight, "task_id": tid, "rep": rep,
            "integrand_class": task_class.get(tid),
            "context_prettyprint_for_review": context,  # DS #178 item 4: renamed so it cannot be silently mistaken for training input
            "context_messages": context_messages,  # REAL chat-format messages, use this for training
            "chosen": chosen_text, "rejected": rejected_text,
            "fire_round": r["guard_fire_round"], "trustworthy_fire": is_trustworthy,
        })

    print(f"Raw candidates: {len(candidates)} (excluded {n_excluded_unverified_soft} unverified-soft, "
          f"{n_nudge_wrong} nudge-wrong)")

    # cap per-task and per-class, preferring highest-quality candidates first:
    # trustworthy-hard > trustworthy-soft > unverified-hard(w=0.4)
    import sys as _sys
    max_per_task = int(_sys.argv[1]) if len(_sys.argv) > 1 else 3
    by_task = defaultdict(int)
    by_class = defaultdict(int)
    final = []
    max_per_class = max(1, int(0.15 * len(candidates))) if candidates else 1

    def sort_key(c):
        return (0 if (c["trustworthy_fire"] and c["kind"] == "hard") else
                1 if (c["trustworthy_fire"] and c["kind"] == "soft") else 2)

    for c in sorted(candidates, key=sort_key):
        if by_task[c["task_id"]] >= max_per_task:
            continue
        if by_class[c["integrand_class"]] >= max_per_class:
            continue
        final.append(c)
        by_task[c["task_id"]] += 1
        by_class[c["integrand_class"]] += 1

    n_hard = sum(1 for c in final if c["kind"] == "hard")
    n_soft = sum(1 for c in final if c["kind"] == "soft")
    n_weak = sum(1 for c in final if not c["trustworthy_fire"])
    print(f"Final mined pairs: {len(final)} ({n_hard} hard, {n_soft} soft, {n_weak} from the "
          f"weaker unverified pathway at weight 0.4) [max_per_task={max_per_task}]")
    print("by integrand class:", dict(by_class))

    with open("/home/ubuntu/antidoom_data/retrain_dpo_pairs_v1.jsonl", "w") as f:
        for c in final:
            f.write(json.dumps(c) + "\n")
    print("wrote retrain_dpo_pairs_v1.jsonl")


if __name__ == "__main__":
    main()
