#!/usr/bin/env python3
"""Beat-ThinkingCap track: convert mined concise-correct traces into kind='sft_ce' training rows.
context = chat-templated prompt (same convention as every other row in this project);
response_text = the full, natural, untruncated self-rollout text (no truncation, no forced format)."""
import json
from pathlib import Path
from transformers import AutoTokenizer

IN = Path.home() / "antidoom_data" / "concise_sft" / "selected_p70.jsonl"
OUT = Path.home() / "antidoom_data" / "concise_sft" / "sft_rows.jsonl"

tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)


def main():
    rows = [json.loads(l) for l in open(IN)]
    out_rows = []
    for r in rows:
        ctx = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False,
            add_generation_prompt=True, reasoning_effort="medium"
        )
        out_rows.append({
            "kind": "sft_ce", "context": ctx, "response_text": r["self_text"],
            "pilot_id": r["id"], "domain": r["domain"],
        })
    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"WRITTEN: {len(out_rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
