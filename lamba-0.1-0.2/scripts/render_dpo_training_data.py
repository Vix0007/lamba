#!/usr/bin/env python3
"""Offline chat-template rendering for the DPO retrain data, following this project's
established convention (all prior rounds render text OFFLINE at data-build time, not
inside the trainer's Dataset classes -- e.g. loop-safety/GPQA-D rollout captures already
store fully-rendered prompt text). Two real bugs avoided here, both already documented
in this project's history / just found this session:
  1. context_messages must be REAL chat-format messages, not the pretty-print summary
     (found+fixed in mine_retrain_pairs.py this session).
  2. tool_call.arguments must be a parsed dict, not a raw JSON string, or the chat
     template's `.items()` filter crashes (documented bug, hit again and fixed here).
Writes rendered_context (a plain string ending right after '<think>\n', i.e. the exact
generation-prompt boundary) into both retrain_dpo_pairs_v1_1.jsonl -> _rendered.jsonl
and kl_replay_set_v1.jsonl -> _rendered.jsonl (skips letter_group-kind rows, which use
their own established single-position format unchanged).
"""
import json
import sys

sys.path.insert(0, "/home/ubuntu/antidoom/src")
sys.path.insert(0, "/home/ubuntu/web")
from transformers import AutoTokenizer  # noqa: E402
import server as srv  # noqa: E402 (real TOOLS schema, must match generation-time parity)

TOK = AutoTokenizer.from_pretrained("/home/ubuntu/models/lmh_letterfix2_c10", trust_remote_code=True)
# DS #178's parity check (item 3) caught a REAL bug: every real generation source in this
# pipeline used chat_template_kwargs={"reasoning_effort":"medium"} (verified: both
# gen_retrain_pool_rollouts.py and loop_test_g.py pass this explicitly), NOT the template's
# own default ("xhigh") -- and DPO-pair contexts are real tool-calling conversations, so
# they also need the real tools= schema passed (verified: template's tool-instructions
# block is conditional on `tools` being non-empty; omitting it silently drops ~2000 chars
# of real system-prompt content the model actually saw during generation, first found this
# session by diffing render output with/without it). GPQA-D capability rows are exempt --
# they use the ALREADY-rendered real prompt string verbatim (prompt_is_rendered=True), no
# re-rendering, so no parity risk there.
REASONING_EFFORT = "medium"


def fix_tool_call_args(messages):
    msgs = json.loads(json.dumps(messages))  # deep copy, never mutate caller's data
    for m in msgs:
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except Exception:
                        fn["arguments"] = {}
    return msgs


def render_messages(messages):
    return TOK.apply_chat_template(fix_tool_call_args(messages), tokenize=False,
                                    add_generation_prompt=True, tools=srv.TOOLS,
                                    reasoning_effort=REASONING_EFFORT)


def render_single_turn(prompt_text):
    return TOK.apply_chat_template([{"role": "user", "content": prompt_text}],
                                    tokenize=False, add_generation_prompt=True,
                                    reasoning_effort=REASONING_EFFORT)


def main():
    # ---- DPO pairs ----
    n_dpo = 0
    with open("/home/ubuntu/antidoom_data/retrain_dpo_pairs_v1_1.jsonl") as fin, \
         open("/home/ubuntu/antidoom_data/retrain_dpo_pairs_v1_1_rendered.jsonl", "w") as fout:
        for l in fin:
            r = json.loads(l)
            r["rendered_context"] = render_messages(r["context_messages"])
            fout.write(json.dumps(r) + "\n")
            n_dpo += 1
    print(f"rendered {n_dpo} DPO pairs -> retrain_dpo_pairs_v1_1_rendered.jsonl")

    # ---- KL-replay set (sequence-kind rows only; letter_group rows pass through unchanged) ----
    n_seq = 0
    n_letter = 0
    with open("/home/ubuntu/antidoom_data/kl_replay_set_v1.jsonl") as fin, \
         open("/home/ubuntu/antidoom_data/kl_replay_set_v1_rendered.jsonl", "w") as fout:
        for l in fin:
            r = json.loads(l)
            if r.get("kind") == "sequence":
                if r.get("prompt_is_rendered"):
                    r["rendered_context"] = r["prompt_text"]
                else:
                    r["rendered_context"] = render_single_turn(r["prompt_text"])
                n_seq += 1
            else:
                n_letter += 1
            fout.write(json.dumps(r) + "\n")
    print(f"rendered {n_seq} sequence-kind KL-replay rows + passed through {n_letter} letter_group "
          f"rows unchanged -> kl_replay_set_v1_rendered.jsonl")


if __name__ == "__main__":
    main()
