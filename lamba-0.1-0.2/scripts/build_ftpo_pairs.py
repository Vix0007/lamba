"""
Build FTPO pairs for base lamba (soup-linear-l1.0) by REUSING already-identified loop
rollouts (from v3, v4, or v5) instead of generating fresh completions.

For each flagged loop rollout, we know exactly where the loop starts (character offset).
We reconstruct the context (prompt through base lamba's own chat template, plus the
rollout text up to that point) and query base lamba's own logprobs at that single
position via one prefill-only completion request (max_tokens=1, logprobs=20) -- no
generation needed. The "rejected" token is whatever actually appeared there in the
rollout (re-tokenized with base lamba's own tokenizer); "chosen" is the filtered set
of base lamba's own top alternatives at that position.

Output is written directly in antidoom's own FTPO pair schema so it can be fed to
`antidoom train` via train.dataset_jsonl, reusing their trainer unmodified.

Usage: python build_ftpo_pairs.py <source_name> [limit]
  source_name: v5 | v3 | v4
"""
import json
import os
import sys
import requests
from transformers import AutoTokenizer

MODEL_PATH = "/home/ubuntu/models/soup-linear-l1.0"
API_URL = "http://127.0.0.1:8001/v1/completions"
MODEL_NAME = "soup-linear-l1.0"
MAX_MODEL_LEN = 8192
SAFETY_MARGIN = 200  # leave room for the 1 generated token + tokenizer edge cases

CHOSEN_MAX_TOKENS = 20
BATCH_SIZE = 64

SOURCES = {
    "v5": {
        "loop_extracted": "/home/ubuntu/antidoom_data/v5/loop_extracted.jsonl",
        "semantic_results": "/home/ubuntu/antidoom_data/v5/semantic_loop_results.jsonl",
        "prompt_files": [
            "/home/ubuntu/antidoom_data/v5/real_prompts_v5_new.jsonl",
            "/home/ubuntu/antidoom_data/v5/real_prompts_v5_reuse.jsonl",
            "/home/ubuntu/antidoom_data/v5/real_prompts_v5_extra.jsonl",
            "/home/ubuntu/antidoom_data/v5/real_prompts_v5_esc.jsonl",
        ],
        "out_path": "/home/ubuntu/antidoom_data/v5/ftpo_pairs_reused.jsonl",
    },
    "v3": {
        "loop_extracted": "/home/ubuntu/antidoom_data/v3/loop_extracted.jsonl",
        "not_detected": "/home/ubuntu/antidoom_data/v3/not_detected.jsonl",
        "semantic_results": "/home/ubuntu/antidoom_data/v3/semantic_loop_results.jsonl",
        "raw_rollouts": "/home/ubuntu/antidoom_data/v3/real_rollouts.jsonl",
        "prompt_files": ["/home/ubuntu/antidoom_data/v3/real_prompts.jsonl"],
        "out_path": "/home/ubuntu/antidoom_data/v3/ftpo_pairs_reused.jsonl",
    },
    "v4": {
        "loop_extracted": "/home/ubuntu/antidoom_data/v4/loop_extracted.jsonl",
        "not_detected": "/home/ubuntu/antidoom_data/v4/not_detected.jsonl",
        "semantic_results": "/home/ubuntu/antidoom_data/v4/semantic_loop_results.jsonl",
        "raw_rollouts": "/home/ubuntu/antidoom_data/v4/real_rollouts_v4_fleet.jsonl",
        "prompt_files": ["/home/ubuntu/antidoom_data/v4/real_prompts_v4_new.jsonl"],
        "out_path": "/home/ubuntu/antidoom_data/v4/ftpo_pairs_reused.jsonl",
    },
}


def load_prompt_lookup(prompt_files):
    lookup = {}
    for f in prompt_files:
        for line in open(f):
            d = json.loads(line)
            lookup[d["pilot_id"]] = d.get("prompt", "")
    return lookup


def build_examples(cfg, limit=None):
    print("Loading rollout text + exact-match detections...")
    rollouts = {}
    for line in open(cfg["loop_extracted"]):
        d = json.loads(line)
        rollouts[d["rollout_id"]] = d

    # v3/v4 split exact-match hits (loop_extracted) from everything else
    # (not_detected) -- the semantic-only hits' rollout_text lives in the
    # latter and is otherwise invisible to this function. v5 doesn't need
    # this: its loop_extracted.jsonl already covers every flagged rollout.
    if cfg.get("not_detected"):
        print("Loading not-detected pool (semantic-only candidates' rollout text)...")
        for line in open(cfg["not_detected"]):
            d = json.loads(line)
            rollouts.setdefault(d["rollout_id"], d)

    print("Loading semantic-match detections...")
    semantic = {}
    for line in open(cfg["semantic_results"]):
        d = json.loads(line)
        semantic[d["rollout_id"]] = d

    # Quality gate for semantic-only detections on v3/v4: these pools were NOT
    # pre-filtered to already-suspected loops (unlike v5's loop_poisoned label
    # set), so a single incidental near-duplicate sentence in an otherwise
    # healthy, cleanly-concluding rollout is more likely benign self-checking
    # than a genuine doom loop. Require finish_reason == "length" (ran out of
    # budget without concluding) as corroborating evidence -- matches the
    # profile of v5's own confirmed loops (83.8% hit this exact signature).
    finish_reason_lookup = None
    if cfg.get("raw_rollouts"):
        print("Loading finish_reason lookup for semantic quality gate...")
        finish_reason_lookup = {}
        for line in open(cfg["raw_rollouts"]):
            d = json.loads(line)
            finish_reason_lookup[d["rollout_id"]] = d.get("finish_reason")

    print("Loading prompt lookup...")
    prompt_lookup = load_prompt_lookup(cfg["prompt_files"])

    examples = []
    skipped_no_prompt = 0
    skipped_no_onset = 0
    skipped_semantic_not_length = 0
    for rid, r in rollouts.items():
        pilot_id = r["pilot_id"]
        if pilot_id not in prompt_lookup:
            skipped_no_prompt += 1
            continue
        text = r["rollout_text"]
        onset = None
        if r.get("detected_span"):
            onset = r.get("detected_start_char")
        elif rid in semantic and semantic[rid].get("semantic_loop_found"):
            if finish_reason_lookup is not None and finish_reason_lookup.get(rid) != "length":
                skipped_semantic_not_length += 1
            else:
                sent_j = semantic[rid].get("semantic_sent_j")
                if sent_j:
                    idx = text.rfind(sent_j)
                    if idx >= 0:
                        onset = idx
        if onset is None:
            skipped_no_onset += 1
            continue
        ws = 0
        while onset - ws - 1 >= 0 and text[onset - ws - 1] in " \n\t":
            ws += 1
        adj_onset = onset - ws
        examples.append({
            "rollout_id": rid,
            "pilot_id": pilot_id,
            "domain": r["domain"],
            "prompt_text": prompt_lookup[pilot_id],
            "prefix_text": text[:adj_onset],
            "continuation_text": text[adj_onset:onset + 200],
        })
        if limit and len(examples) >= limit:
            break

    print(f"Built {len(examples)} examples (skipped {skipped_no_prompt} no-prompt, {skipped_no_onset} no-onset, "
          f"{skipped_semantic_not_length} semantic-hit-but-not-length-truncated)")
    return examples


def post_batch(prompts, timeout=120):
    resp = requests.post(API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompts,
        "max_tokens": 1,
        "temperature": 0.1,
        "logprobs": 20,
        "stream": False,
    }, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("choices", [])


def filter_chosen(alts, rejected):
    chosen = []
    for token_str, logprob in sorted(alts.items(), key=lambda x: -x[1]):
        if token_str == rejected or token_str.strip() == rejected.strip():
            continue
        stripped = token_str.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if not any(c.isalpha() for c in stripped):
            continue
        if not stripped.isprintable():
            continue
        if len(stripped) < 2:
            continue
        chosen.append(token_str)
        if len(chosen) >= CHOSEN_MAX_TOKENS:
            break
    return chosen


def load_done_ids(out_path):
    """rollout_ids already written to out_path in a prior (possibly crashed) run."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["source_sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a truncated last line from a hard kill
    return done


def main(source_name, limit=None):
    cfg = SOURCES[source_name]
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    examples = build_examples(cfg, limit=limit)

    done_ids = load_done_ids(cfg["out_path"])
    if done_ids:
        before = len(examples)
        examples = [ex for ex in examples if ex["rollout_id"] not in done_ids]
        print(f"Resuming: {len(done_ids)} already done, {before - len(examples)} skipped, {len(examples)} remaining")

    skipped_overlength = 0
    kept = []
    for ex in examples:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt_text"]}],
            tokenize=False, add_generation_prompt=True, reasoning_effort="medium",
        )
        full_ctx = rendered + ex["prefix_text"]
        n_tokens = len(tok(full_ctx, add_special_tokens=False).input_ids)
        if n_tokens > MAX_MODEL_LEN - SAFETY_MARGIN:
            skipped_overlength += 1
            continue
        cont_ids = tok(ex["continuation_text"], add_special_tokens=False).input_ids
        if not cont_ids:
            continue
        ex["context_with_chat_template"] = full_ctx
        ex["rejected_decoded"] = tok.decode([cont_ids[0]])
        kept.append(ex)

    examples = kept
    print(f"{len(examples)} examples fit within context length (skipped {skipped_overlength} overlength)")

    out_f = open(cfg["out_path"], "a")  # append: never clobber prior (possibly crashed) progress
    written = 0
    skipped_no_alt = 0
    skipped_batch_errors = 0
    n_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(examples), BATCH_SIZE):
        batch = examples[i:i + BATCH_SIZE]
        prompts = [ex["context_with_chat_template"] for ex in batch]
        try:
            choices = post_batch(prompts)
        except requests.exceptions.HTTPError:
            # fall back to per-item requests so one bad example doesn't cost the whole batch
            choices = []
            for p in prompts:
                try:
                    choices.extend(post_batch([p]))
                except requests.exceptions.HTTPError:
                    choices.append(None)
                    skipped_batch_errors += 1

        for ex, choice in zip(batch, choices):
            if choice is None:
                continue
            top_logprobs = (choice.get("logprobs") or {}).get("top_logprobs") or [{}]
            alts = top_logprobs[0] if top_logprobs else {}
            rejected = ex["rejected_decoded"]
            chosen = filter_chosen(alts, rejected)
            if len(chosen) < 2:
                skipped_no_alt += 1
                continue
            row = {
                "source_task": f"{source_name}_reused_rollout",
                "source_sample_id": ex["rollout_id"],
                "context_with_chat_template": ex["context_with_chat_template"],
                "rejected_decoded": rejected,
                "multi_chosen_decoded": chosen,
                "repetition": {"domain": ex["domain"], "pilot_id": ex["pilot_id"]},
            }
            out_f.write(json.dumps(row, default=str) + "\n")
            written += 1
        out_f.flush()  # crash-safety: never lose more than one in-flight batch
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  batch {i // BATCH_SIZE + 1}/{n_batches}: {written} pairs written so far", flush=True)

    out_f.close()
    print(f"\nDone. Wrote {written} FTPO pairs to {cfg['out_path']} "
          f"(skipped {skipped_no_alt} no-alt, {skipped_batch_errors} batch-errors)")


if __name__ == "__main__":
    source_name = sys.argv[1] if len(sys.argv) > 1 else "v5"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    main(source_name, limit=limit)
