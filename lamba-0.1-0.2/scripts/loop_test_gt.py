"""Held-out loop test generation (graph mode, resumable, 150 prompts). Greedy; identical engine flags across arms."""
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DEFAULT_PROMPTS = "/home/ubuntu/antidoom_data/loop_testset150.jsonl"
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32]


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
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    a = ap.parse_args()

    if a.mtp and a.lora:
        ap.error("--mtp hurts with LoRA; use MTP only without --lora")

    out = Path.home() / "eval_results" / f"loop_{a.label}_{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    items = [json.loads(l) for l in open(a.prompts) if l.strip()]
    done = set()
    if out.exists():
        for l in open(out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [i for i in items if i["id"] not in done]
    print(f"[{a.label}/{a.tag}] {len(items)} prompts, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    kw = dict(
        model=a.model,
        quantization="fp8",
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=a.gpu_util,
        max_num_seqs=64,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                            "cudagraph_capture_sizes": CAPTURE_SIZES},
    )
    if a.lora:
        kw.update(enable_lora=True, max_lora_rank=256, max_loras=1)
    if a.mtp:
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
        kw["enable_prefix_caching"] = False

    t0 = time.time()
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    lreq = LoRARequest("adapter", 1, a.lora) if a.lora else None
    sp = SamplingParams(temperature=a.temp, top_p=0.95, seed=a.seed, max_tokens=a.max_tokens)

    total_tok = 0.0
    total_time = 0.0
    with open(out, "a") as f:
        for i in range(0, len(todo), a.chunk):
            batch = todo[i:i + a.chunk]
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
                                    "text": o.text}) + "\n")
            f.flush()
            total_tok += ntok
            total_time += dt
            print(f"chunk {i // a.chunk + 1}: {len(batch)} prompts in {dt:.0f}s "
                  f"| {ntok / dt:.1f} tok/s | {ntok} tok", flush=True)

    print(f"ALL chunks: {total_tok:.0f} tok in {total_time:.0f}s -> {total_tok / max(total_time, 1e-9):.1f} tok/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()