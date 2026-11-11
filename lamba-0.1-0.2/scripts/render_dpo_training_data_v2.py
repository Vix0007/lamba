#!/usr/bin/env python3
"""Same rendering as render_dpo_training_data.py, pointed at the corrected (needle-bug-
fixed, cap=10) 406-pair set for the dpov2 bugfix-retrain (DS #206: redo the SAME recipe
from scratch, init=s1337/dpo_ref=s1337, not a continuation). KL-replay set is UNCHANGED
(c10's own panel behavior is the fixed safety target regardless of which policy trains)."""
import json
import sys

sys.path.insert(0, "/home/ubuntu/antidoom/src")
sys.path.insert(0, "/home/ubuntu/web")
from transformers import AutoTokenizer  # noqa: E402
import server as srv  # noqa: E402

TOK = AutoTokenizer.from_pretrained("/home/ubuntu/models/lmh_letterfix2_c10", trust_remote_code=True)
REASONING_EFFORT = "medium"


def fix_tool_call_args(messages):
    msgs = json.loads(json.dumps(messages))
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


def main():
    n_dpo = 0
    with open("/home/ubuntu/antidoom_data/retrain_dpo_pairs_v2_needlefix_cap10.jsonl") as fin, \
         open("/home/ubuntu/antidoom_data/retrain_dpo_pairs_v2_needlefix_cap10_rendered.jsonl", "w") as fout:
        for l in fin:
            r = json.loads(l)
            r["rendered_context"] = render_messages(r["context_messages"])
            fout.write(json.dumps(r) + "\n")
            n_dpo += 1
    print(f"rendered {n_dpo} DPO pairs -> retrain_dpo_pairs_v2_needlefix_cap10_rendered.jsonl")


if __name__ == "__main__":
    main()
