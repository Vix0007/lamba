"""DS #136 repair-SFT mining, stage 1: generate K rollouts per mining prompt from kl005-seed1337,
sampled (not greedy, needed for diversity across K), same chat template as deployment eval.
Writes ALL K completions per prompt (not just looping ones) to a resumable jsonl; a separate
pass (detect_repair_candidates.py) finds which ones actually loop and extracts onset positions.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ubuntu/models/lmh_kl005_final")
    ap.add_argument("--lora", default="/home/ubuntu/adapters_kl005_final")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.prompts) if l.strip()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [i for i in items if i["id"] not in done]
    print(f"{len(items)} prompts, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    kw = dict(
        model=a.model, quantization="fp8", dtype="bfloat16", max_model_len=8192,
        trust_remote_code=True, gpu_memory_utilization=0.90, max_num_seqs=64,
        enable_lora=True, max_lora_rank=256, max_loras=1,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                             "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
    )
    t0 = time.time()
    llm = LLM(**kw)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    lreq = LoRARequest(os.path.basename(a.lora.rstrip("/")), 1, a.lora)
    tok = llm.get_tokenizer()

    total_tok = 0.0
    total_time = 0.0
    with open(out, "a") as f:
        for i in range(0, len(todo), a.chunk):
            batch = todo[i:i + a.chunk]
            ps = [tok.apply_chat_template([{"role": "user", "content": b["prompt"]}],
                                          tokenize=False, add_generation_prompt=True,
                                          reasoning_effort="medium") for b in batch]
            sps = [SamplingParams(temperature=1.0, top_p=0.95, max_tokens=a.max_tokens,
                                   n=a.k, seed=10000 + idx * a.k)
                   for idx, b in enumerate(batch, start=i)]
            t1 = time.time()
            res = llm.generate(ps, sps, lora_request=lreq)
            dt = time.time() - t1
            ntok = 0
            for b, r in zip(batch, res):
                completions = []
                for o in r.outputs:
                    n = len(o.token_ids)
                    ntok += n
                    completions.append({"finish_reason": o.finish_reason, "n_tokens": n, "text": o.text})
                f.write(json.dumps({"id": b["id"], "prompt": b["prompt"], "domain": b.get("domain"),
                                    "completions": completions}) + "\n")
            f.flush()
            total_tok += ntok
            total_time += dt
            print(f"chunk {i // a.chunk + 1}: {len(batch)} prompts x{a.k} in {dt:.0f}s "
                  f"| {ntok / dt:.1f} tok/s | {ntok} tok", flush=True)

    print(f"ALL: {total_tok:.0f} tok in {total_time:.0f}s -> {total_tok/max(total_time,1e-9):.1f} tok/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
