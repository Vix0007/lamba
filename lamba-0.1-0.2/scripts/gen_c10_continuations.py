"""DS #136 repair-SFT mining, stage 3: for each repair candidate (prompt + kl005 prefix through
onset), generate c10's continuation from that exact point, greedy, matching the shared prefix.
Repaired sequence = kl005_prefix + c10_continuation. Resumable.
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
    ap.add_argument("--model", default="/home/ubuntu/models/lmh_letterfix2_c10")
    ap.add_argument("--lora", default="/home/ubuntu/adapters_letterfix2_c10")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()

    cands = [json.loads(l) for l in open(a.candidates) if l.strip()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            try:
                r = json.loads(l)
                done.add((r["id"], r["k_idx"]))
            except Exception:
                pass
    todo = [c for c in cands if (c["id"], c["k_idx"]) not in done]
    print(f"{len(cands)} candidates, {len(done)} done, {len(todo)} to go", flush=True)
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

    with open(out, "a") as f:
        for i in range(0, len(todo), a.chunk):
            batch = todo[i:i + a.chunk]
            prompts_texts = []
            for c in batch:
                chat = tok.apply_chat_template(
                    [{"role": "user", "content": c["prompt"]}], tokenize=False,
                    add_generation_prompt=True, reasoning_effort="medium")
                prefix = c["full_text"][:c["onset_char_pos"]]
                prompts_texts.append(chat + prefix)
            remaining = [c["n_tokens"] - len(tok(c["full_text"][:c["onset_char_pos"]],
                                                   add_special_tokens=False)["input_ids"])
                         for c in batch]
            sps = [SamplingParams(temperature=0.0,
                                   max_tokens=max(1024, 2 * max(1, r)))
                   for r in remaining]
            t1 = time.time()
            res = llm.generate(prompts_texts, sps, lora_request=lreq)
            dt = time.time() - t1
            ntok = 0
            for c, r in zip(batch, res):
                o = r.outputs[0]
                ntok += len(o.token_ids)
                f.write(json.dumps({
                    "id": c["id"], "k_idx": c["k_idx"], "domain": c.get("domain", "science"),
                    "prompt": c["prompt"], "prefix": c["full_text"][:c["onset_char_pos"]],
                    "c10_continuation": o.text, "finish_reason": o.finish_reason,
                    "c10_n_tokens": len(o.token_ids),
                }) + "\n")
            f.flush()
            print(f"chunk {i//a.chunk+1}: {len(batch)} in {dt:.0f}s | {ntok/dt:.1f} tok/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
