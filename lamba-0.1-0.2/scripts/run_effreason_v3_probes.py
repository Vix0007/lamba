#!/usr/bin/env python3
"""Efficient-reasoning v3, step 2: run the confidence-gated probes (DS #207a fix for the
v1/v2 single-greedy-guess bug). Direct vLLM engine (same pattern as the old
run_effreason_r2_probes.py -- batched llm.generate(), NOT per-request HTTP), parameterized
by --model/--lora so this can target whichever checkpoint wins (dpov1 or dpov2), not
hardcoded to c10 like the old script was.

For free-form answers (this entire corpus -- verified no MCQ-format items in mcp/science/
malay, all numeric/symbolic/short-answer): at each probe position, force a short answer
completion K=8 times at T=0.7/top_p=0.95 (max 32 new tokens, n=8 in one SamplingParams),
record how many of the 8 match gold after normalization, AND whether gold is the UNIQUE
mode among the 8 (ties don't count as confident).

ALSO runs the t=0 pre-filter (DS: discard items the model can already answer with zero
reasoning) as the same K=8 check at cut=0/empty prefix -- adapted from DS's literal
margin>=2.0-nats spec since this corpus is free-form, not single-token MCQ.
"""
import argparse
import json
import re
import time
from collections import Counter

import os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PROBEPOS = "effreason_v3_probe_positions.jsonl"
CANDS = "effreason_v3_candidates.jsonl"
K = 8
FORCE = "\n\nI have reasoned enough. My final answer is:\n\\boxed{"


def extract_final(text):
    text = text or ""
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1]
    fences = re.findall(r"```(?:\w*\n)?(.*?)```", text, flags=re.S)
    if fences:
        return fences[-1]
    tail = text[-200:]
    bolds = re.findall(r"\*\*([^*]+)\*\*", tail)
    if bolds:
        return bolds[-1]
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def norm(s):
    s = extract_final(s or "")
    s = re.sub(r"[#*`$]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def is_confident_correct(answers, gold):
    """DS #207a rule: >=7/8 exact match to gold AND gold is the UNIQUE mode."""
    gold_n = norm(gold)
    if not gold_n:
        return False, 0
    normed = [norm(a) for a in answers]
    n_match = sum(1 for a in normed if a == gold_n)
    counts = Counter(normed)
    mode, mode_n = counts.most_common(1)[0]
    is_unique_mode = list(counts.values()).count(mode_n) == 1
    confident = n_match >= 7 and mode == gold_n and is_unique_mode
    return confident, n_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ubuntu/models/lmh_ruminconfab_dpo_v1")
    ap.add_argument("--lora", default="/home/ubuntu/adapters_ruminconfab_dpo_v1")
    ap.add_argument("out_path")
    args = ap.parse_args()

    cands = [json.loads(l) for l in open(CANDS) if l.strip()]
    prompts = {}
    for l in open(PROBEPOS):
        r = json.loads(l)
        prompts.setdefault(r["pilot_id"], r["prompt"])

    items = [json.loads(l) for l in open(PROBEPOS) if l.strip()]
    for c in cands:
        if c["pilot_id"] in prompts:
            items.append({"pilot_id": c["pilot_id"], "cut": 0, "L": len(c["self_text"]),
                           "domain": c["domain"], "prompt": prompts[c["pilot_id"]], "prefix": "",
                           "gold": c["gold"], "is_prefilter": True})

    done = set()
    try:
        for l in open(args.out_path):
            r = json.loads(l)
            done.add((r["pilot_id"], r["cut"]))
    except FileNotFoundError:
        pass
    todo = [i for i in items if (i["pilot_id"], i["cut"]) not in done]
    print(f"{len(items)} probe points, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("RUN_EFFREASON_V3_PROBES_DONE", flush=True)
        return

    kw = dict(model=args.model, quantization="fp8", dtype="bfloat16", max_model_len=8192,
              trust_remote_code=True, gpu_memory_utilization=0.90, max_num_seqs=64,
              enable_lora=True, max_lora_rank=256, max_loras=1,
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                   "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]})
    t0 = time.time()
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    lreq = LoRARequest("adapter", 1, args.lora)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=32, n=K, stop=["}"])

    CH = 64  # smaller chunk than the old script's 256 since n=8 multiplies sequences 8x
    with open(args.out_path, "a") as f:
        for i in range(0, len(todo), CH):
            batch = todo[i:i + CH]
            ctxs = []
            for b in batch:
                base = tok.apply_chat_template(
                    [{"role": "user", "content": b["prompt"]}], tokenize=False,
                    add_generation_prompt=True, reasoning_effort="medium")
                ctxs.append(base + b["prefix"] + FORCE)
            outs = llm.generate(ctxs, sp, lora_request=lreq)
            for b, o in zip(batch, outs):
                answers = ["\\boxed{" + comp.text for comp in o.outputs]
                confident, n_match = is_confident_correct(answers, b["gold"])
                f.write(json.dumps({"pilot_id": b["pilot_id"], "cut": b["cut"], "L": b["L"],
                                     "domain": b["domain"], "n_match": n_match,
                                     "confident_correct": confident,
                                     "is_prefilter": b.get("is_prefilter", False)}) + "\n")
            f.flush()
            print(f"{min(i+CH, len(todo))}/{len(todo)}", flush=True)

    print("RUN_EFFREASON_V3_PROBES_DONE", flush=True)


if __name__ == "__main__":
    main()
