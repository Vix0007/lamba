"""Held-out loop test generation (graph mode, resumable, 150 prompts). Greedy; identical engine flags across arms."""
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DEFAULT_PROMPTS = "/home/ubuntu/antidoom_data/loop_testset150.jsonl"
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32]


def run_jobs(label, tag, prompts_path, model=None, lora=None, mtp=False, max_tokens=4000,
             chunk=75, gpu_util=0.90, repetition_penalty=1.0, eager=False, llm=None, lreq=None,
             length_cond_rp=False, length_cond_rp_targeted=False):
    """Core generation logic (unchanged from the original main() body). If `llm` is provided
    (from eval_combined.py sharing one engine across gate/g150/other checks on the same
    adapter), reuse it and skip building a new one -- otherwise build one exactly as the
    standalone CLI always has. Built after repeatedly paying the ~8min model-reload tax once
    per loop_test_g.py invocation in the same eval chain (gate24 then g150 then ... each its
    own fresh process) despite eval_combined.py existing for exactly this reason -- don't
    repeat that mistake."""
    if mtp and lora:
        raise SystemExit("--mtp hurts with LoRA; use MTP only without --lora")

    out = Path.home() / "eval_results" / f"loop_{label}_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in open(prompts_path) if l.strip()]
    done = set()
    if out.exists():
        for l in open(out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [i for i in items if i["id"] not in done]
    print(f"[{label}/{tag}] {len(items)} prompts, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    if llm is None:
        kw = dict(
            model=model,
            quantization="fp8",
            dtype="bfloat16",
            max_model_len=8192,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_util,
            max_num_seqs=64,
        )
        if eager:
            kw["enforce_eager"] = True
        else:
            kw["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY",
                                         "cudagraph_capture_sizes": CAPTURE_SIZES}
        if lora:
            kw.update(enable_lora=True, max_lora_rank=256, max_loras=1)
        if mtp:
            kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
            kw["enable_prefix_caching"] = False
        if length_cond_rp:
            from length_conditioned_rp import LengthConditionedRepetitionPenalty
            kw["logits_processors"] = [LengthConditionedRepetitionPenalty]
        if length_cond_rp_targeted:
            from length_conditioned_rp_targeted import LengthConditionedTargetedPenalty
            kw["logits_processors"] = [LengthConditionedTargetedPenalty]

        t0 = time.time()
        llm = LLM(**kw)
        print(f"loaded in {time.time() - t0:.0f}s", flush=True)
        lreq = LoRARequest(os.path.basename(lora.rstrip("/")), 1, lora) if lora else None

    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=repetition_penalty)

    total_tok = 0.0
    total_time = 0.0
    with open(out, "a") as f:
        for i in range(0, len(todo), chunk):
            batch = todo[i:i + chunk]
            ps = [tok.apply_chat_template([{"role": "user", "content": b["prompt"]}],
                                          tokenize=False, add_generation_prompt=True,
                                          reasoning_effort="medium") for b in batch]
            t1 = time.time()
            res = llm.generate(ps, sp, lora_request=lreq)
            dt = time.time() - t1
            ntok = 0
            for b, r in zip(batch, res):
                o = r.outputs[0]
                n = len(o.token_ids)
                ntok += n
                f.write(json.dumps({"id": b["id"], "domain": b.get("domain"),
                                    "n_pairs": b.get("n_pairs"),
                                    "finish_reason": o.finish_reason,
                                    "n_tokens": n,
                                    "elapsed_s": round(dt, 3),
                                    "repetition_penalty": repetition_penalty,
                                    "text": o.text}) + "\n")
            f.flush()
            total_tok += ntok
            total_time += dt
            print(f"chunk {i // chunk + 1}: {len(batch)} prompts in {dt:.0f}s "
                  f"| {ntok / dt:.1f} tok/s | {ntok} tok", flush=True)

    print(f"ALL chunks: {total_tok:.0f} tok in {total_time:.0f}s -> {total_tok / max(total_time, 1e-9):.1f} tok/s", flush=True)
    print("DONE", flush=True)


def run_with_engine(llm, lora_req, label, tag, prompts_path, max_tokens=4000, chunk=75,
                     repetition_penalty=1.0):
    """Entry point for eval_combined.py: reuse an already-built engine/LoRARequest."""
    lora_path = lora_req.lora_path if lora_req else None
    return run_jobs(label, tag, prompts_path, lora=lora_path, max_tokens=max_tokens, chunk=chunk,
                     repetition_penalty=repetition_penalty, llm=llm, lreq=lora_req)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=75)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    # DS #118 guard: no silent default. A missing flag caused rp=1.0 candidates to be compared
    # against rp=1.1 baselines (concise-sft, effreason-r2, kl01) and produced false "regressions".
    ap.add_argument("--repetition-penalty", type=float, required=True)
    ap.add_argument("--eager", action="store_true",
                     help="enforce_eager=True, skip CUDA graph compile/capture entirely -- "
                          "much faster to start, slower per-token; use for a one-off eval "
                          "where startup time dominates a short run.")
    ap.add_argument("--length-cond-rp", action="store_true",
                     help="DS #130/#131: global-scope length-conditioned repetition-penalty "
                          "logits processor (penalizes ALL previously-seen tokens once "
                          "triggered, like standard rp but dynamic). DS #132: run this "
                          "ADDITIVE to a static --repetition-penalty (e.g. 1.1), not as a "
                          "replacement -- the earlier replacement-mode run was catastrophic.")
    ap.add_argument("--length-cond-rp-targeted", action="store_true",
                     help="DS #132 Box A: targeted variant -- same trigger logic, but "
                          "penalizes ONLY the token IDs forming the detected repeated "
                          "n-gram(s), not the whole seen-token history. Always additive to "
                          "--repetition-penalty per DS's spec.")
    a = ap.parse_args()
    if a.length_cond_rp and a.length_cond_rp_targeted:
        raise SystemExit("pick one: --length-cond-rp (global) or --length-cond-rp-targeted, not both")
    run_jobs(a.label, a.tag, a.prompts, model=a.model, lora=a.lora, mtp=a.mtp,
             max_tokens=a.max_tokens, chunk=a.chunk, gpu_util=a.gpu_util,
             repetition_penalty=a.repetition_penalty, eager=a.eager,
             length_cond_rp=a.length_cond_rp, length_cond_rp_targeted=a.length_cond_rp_targeted)


if __name__ == "__main__":
    main()