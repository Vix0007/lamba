#!/usr/bin/env python3
"""Build training-row SCHEMA + seed pairs from the nudge-guard A/B (DS #164's design,
NOT yet for training -- ds explicitly said "build the pipeline, do not train tonight").

Per DS #164 section 2:
- Strip the nudge scaffold from context (context = tokens up to the fire point ONLY;
  the injected "[GUARD] you have enough info..." message must NEVER appear in training
  context, since it never appears at serve time -- training on it would teach
  P(answer | state, nudge) which is useless without the nudge).
- Use PAIRED PREFERENCES, not SFT positives: shadow's continuation = rejected, nudge's
  post-nudge answer = chosen, same prefix (matched state). This is a much better data
  structure than solitary SFT targets (24 "strong" pairs where shadow was wrong) plus
  8 "weak/process" pairs (shadow was right but wasteful -- needs the #154 process-reward
  penalty on redundant tool calls, not a plain positive).
- Onset token = the first token of the REDUNDANT tool call (the one after which the
  guard fired), NOT the first token of the first verification call.

Output: all_ftpo_pairs_ruminconfab_nudge_seed.jsonl (schema-only, small seed set from
tonight's 32 fired-and-intervened pairs). NOT wired into any training config. A future
session should expand this to >=50 distinct integral tasks / >=300 pairs per DS's sizing
before ever training on it.
"""
import json
import re


def strip_commas(t):
    return re.sub(r"(?<=\d),(?=\d)", "", t or "")


def numeric_str_variants(v, ndigits=6):
    s = f"{v:.{ndigits}f}"
    out = {s}
    for n in range(1, ndigits):  # BUG FIX 2026-10-01: was range(2,..), missed clean 1-decimal values
        out.add(f"{v:.{n}f}")
    if abs(v - round(v)) < 1e-9:
        iv = int(round(v))
        out.add(str(iv))
        out.add(f"{v:.1f}")
        out.add(f"{iv:,}")
    return out


def contains_value(text, needles):
    if not text:
        return False
    t2 = strip_commas(text)
    return any(n in text or n in t2 for n in needles)


def build_pretty_context(prompt_text, transcript, up_to_round):
    """Reconstructs a human-readable context string: the prompt + the assistant/tool
    turns up through (and including) the tool round where the guard fired, EXCLUDING
    the nudge message and anything after it. up_to_round is the guard_fire_round (an
    index into the round loop, 0-based; the transcript has 2 entries per round:
    assistant+tool_calls, then tool result)."""
    lines = [f"[USER PROMPT]\n{prompt_text}\n"]
    tool_round_seen = -1
    for turn in transcript:
        role = turn.get("role")
        if role == "assistant":
            reasoning = turn.get("reasoning") or ""
            content = turn.get("content") or ""
            tool_calls = turn.get("tool_calls")
            block = "[ASSISTANT]\n"
            if reasoning:
                block += f"(reasoning) {reasoning}\n"
            if content:
                block += content + "\n"
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    block += f"[TOOL_CALL {fn.get('name')}] {fn.get('arguments')}\n"
            lines.append(block)
        elif role == "tool":
            tool_round_seen += 1
            lines.append(f"[TOOL_RESULT]\n{turn.get('content','')}\n")
            if tool_round_seen == up_to_round:
                break
    return "\n".join(lines)


def main():
    shadow_path = "/home/ubuntu/eval_results/nudge_ab_shadow_s1337.jsonl"
    nudge_path = "/home/ubuntu/eval_results/nudge_ab_nudge_s1337.jsonl"
    prompts_path = "/home/ubuntu/antidoom_data/nudge_ab_prompts.jsonl"
    out_path = "/home/ubuntu/antidoom_data/all_ftpo_pairs_ruminconfab_nudge_seed.jsonl"

    prompt_by_id = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(prompts_path)}

    shadow = {}
    for l in open(shadow_path):
        r = json.loads(l)
        shadow[(r["base_id"], r["rep"])] = r
    nudge = {}
    for l in open(nudge_path):
        r = json.loads(l)
        nudge[(r["base_id"], r["rep"])] = r

    fired = [k for k, v in shadow.items() if v.get("guard_fired")]
    strong, weak = [], []
    for k in fired:
        s, n = shadow[k], nudge.get(k)
        if n is None or n.get("error") or not n.get("guard_intervened"):
            continue
        needles = numeric_str_variants(s["ground_truth_numeric"])
        shadow_correct = contains_value(s.get("final_text", ""), needles)
        nudge_correct = contains_value(n.get("final_text", ""), needles)
        if not nudge_correct:
            continue  # only use verified-correct chosen sides, per DS's instruction

        fire_round = s["guard_fire_round"]
        context = build_pretty_context(prompt_by_id[k[0]], s["transcript"], fire_round)
        rejected_continuation = None
        # shadow's actual continuation past the fire point (the redundant behavior)
        tool_round_seen = -1
        collecting = False
        rej_lines = []
        for turn in s["transcript"]:
            if turn.get("role") == "tool":
                tool_round_seen += 1
                if tool_round_seen == fire_round:
                    collecting = True
                    continue
            if collecting:
                if turn.get("role") == "assistant":
                    rej_lines.append(turn.get("content") or turn.get("reasoning") or "(redundant tool call)")
                elif turn.get("role") == "assistant_forced_final":
                    rej_lines.append(turn.get("content") or "(empty final)")
        rejected_continuation = "\n".join(rej_lines) if rej_lines else "(no further content -- empty final)"

        chosen_continuation = n.get("final_text", "")

        row = {
            "kind": "ftpo_soft" if shadow_correct else "ftpo_hard",  # strong=hard, weak/process=soft, per DS
            "source": "ruminconfab_nudge_seed_s1337",
            "task_id": k[0], "rep": k[1],
            "context": context,
            "chosen_continuation": chosen_continuation,
            "rejected_continuation": rejected_continuation,
            "shadow_correct": shadow_correct, "nudge_correct": nudge_correct,
            "fire_round": fire_round, "fire_reason": s.get("guard_fire_reason"),
            "NOTE": "SCHEMA/SEED ONLY -- per DS #164, do not train on this without expanding "
                    "to >=50 distinct integral tasks / >=300 pairs first. Onset-token extraction "
                    "(exact token boundary for FTPO) still needs to be done against the real "
                    "tokenizer, not just this pretty-printed context.",
        }
        (weak if shadow_correct else strong).append(row)

    print(f"strong (shadow wrong, nudge rescued) pairs: {len(strong)}")
    print(f"weak/process (shadow right but wasteful) pairs: {len(weak)}")
    with open(out_path, "w") as f:
        for r in strong + weak:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(strong)+len(weak)} schema-seed rows to {out_path}")


if __name__ == "__main__":
    main()
