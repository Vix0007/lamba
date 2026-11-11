#!/usr/bin/env python3
"""Combined capability-check runner: loads the model/adapter ONCE, then runs GPQA-D (budget-
forced, same protocol as gpqa_ab_g.py) followed by the MalayMMLU letter-bias checks (job A
full-benchmark and/or job B 5k shift-consistency, same protocol as eval_letter_adapter.py).

Built after burning ~3 separate ~5-6min model-reload cycles in one evening testing a single new
checkpoint (each of gpqa_ab_g.py / eval_letter_adapter.py builds its own fresh vLLM engine from
scratch) -- this script shares ONE engine across every check, so a full multi-eval pass on a new
adapter pays the model-load + compile tax exactly once, not once per script.

Usage:
  eval_combined.py --model <path> --lora <path> --tag <label> [--jobs gpqa,A,B] [--repetition-penalty 1.1]

Uses max_model_len=40000 throughout (GPQA's requirement; comfortably covers MalayMMLU's much
shorter single-token-scoring contexts too, so one config serves every job).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ref_eval_scripts"))
sys.path.insert(0, str(Path.home() / "antidoom_data"))

from vllm import LLM
from vllm.lora.request import LoRARequest

import gpqa_ab_g as G
import eval_letter_adapter as L
import loop_test_g as LT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", default="gpqa,A,B", help="comma list from {gpqa,A,B,gate,g150}")
    ap.add_argument("--budget", type=int, default=4000)
    # DS #118 guard: must be explicit (see loop_test_g.py).
    ap.add_argument("--repetition-penalty", type=float, required=True)
    ap.add_argument("--compare-to", default=None)
    ap.add_argument("--gate-prompts", default="/home/ubuntu/antidoom_data/loop_gate24.jsonl")
    ap.add_argument("--g150-prompts", default="/home/ubuntu/antidoom_data/loop_testset150.jsonl")
    a = ap.parse_args()
    jobs = {j.strip() for j in a.jobs.split(",") if j.strip()}

    kw = dict(model=a.model, quantization="fp8", dtype="bfloat16", max_model_len=40000,
              trust_remote_code=True, gpu_memory_utilization=0.90, max_num_seqs=64, max_logprobs=64,
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                  "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]})
    if a.lora:
        kw.update(enable_lora=True, max_lora_rank=256, max_loras=1)
    print(f"[{a.tag}] building ONE shared engine for jobs={sorted(jobs)} ...", flush=True)
    llm = LLM(**kw)
    lora_req = LoRARequest(Path(a.lora.rstrip("/")).name, 1, a.lora) if a.lora else None

    if "gpqa" in jobs:
        print(f"[{a.tag}] === GPQA-D ===", flush=True)
        G.run_with_engine(llm, lora_req, label=a.tag, model_path=a.model, budget=a.budget,
                           repetition_penalty=a.repetition_penalty, compare_to=a.compare_to)

    letter_jobs = jobs & {"A", "B"}
    if letter_jobs:
        print(f"[{a.tag}] === MalayMMLU {sorted(letter_jobs)} ===", flush=True)
        L.run_with_engine(llm, lora_req, tag=a.tag, jobs=letter_jobs)

    if "gate" in jobs:
        print(f"[{a.tag}] === loop gate24 ===", flush=True)
        LT.run_with_engine(llm, lora_req, label=a.tag, tag="gate", prompts_path=a.gate_prompts,
                            max_tokens=1200, chunk=24)

    if "g150" in jobs:
        print(f"[{a.tag}] === loop g150 ===", flush=True)
        LT.run_with_engine(llm, lora_req, label=a.tag, tag="g150", prompts_path=a.g150_prompts,
                            max_tokens=4000, chunk=75,
                            repetition_penalty=a.repetition_penalty)

    print(f"[{a.tag}] COMBINED_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
