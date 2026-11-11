"""DS #139 Option A: lead-time sweep. For each of the 75 repair candidates, truncate the kl005
prefix at k tokens BEFORE the detected onset (k=0,5,10,20,50,100), generate c10's continuation
from each earlier point, and check whether the same 4-part filter passes. Usable rate vs k
tells us whether the divergence horizon is local (usable rate jumps at small k) or diffuse
(stays ~flat all the way back).
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

K_VALUES = [0, 5, 10, 20, 50, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/ubuntu/models/lmh_letterfix2_c10")
    ap.add_argument("--lora", default="/home/ubuntu/adapters_letterfix2_c10")
    ap.add_argument("--candidates", required=True)  # original repair candidates (prompt, full_text, onset_char_pos)
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
                done.add((r["id"], r["k_idx"], r["lead_k"]))
            except Exception:
                pass

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

    # build task list: for each candidate, each lead_k, compute the truncated prefix by TOKEN count
    tasks = []
    for c in cands:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": c["prompt"]}], tokenize=False,
            add_generation_prompt=True, reasoning_effort="medium")
        full_gen_ids = tok(c["full_text"], add_special_tokens=False)["input_ids"]
        onset_prefix_ids = tok(c["full_text"][:c["onset_char_pos"]], add_special_tokens=False)["input_ids"]
        onset_tok_idx = len(onset_prefix_ids)
        for k in K_VALUES:
            if (c["id"], c["k_idx"], k) in done:
                continue
            cut = max(0, onset_tok_idx - k)
            prefix_ids = full_gen_ids[:cut]
            tasks.append({"id": c["id"], "k_idx": c["k_idx"], "lead_k": k,
                          "prompt": c["prompt"], "chat": chat,
                          "prefix_ids": prefix_ids, "remaining": onset_tok_idx - cut + 200})

    print(f"{len(tasks)} generation tasks to do (of {len(cands)*len(K_VALUES)} total)", flush=True)
    if not tasks:
        print("DONE", flush=True)
        return

    with open(out, "a") as f:
        for i in range(0, len(tasks), a.chunk):
            batch = tasks[i:i + a.chunk]
            prompts_texts = [t["chat"] + tok.decode(t["prefix_ids"]) for t in batch]
            sps = [SamplingParams(temperature=0.0, max_tokens=max(1024, 2 * t["remaining"]))
                   for t in batch]
            t1 = time.time()
            res = llm.generate(prompts_texts, sps, lora_request=lreq)
            dt = time.time() - t1
            ntok = 0
            for t_item, r in zip(batch, res):
                o = r.outputs[0]
                ntok += len(o.token_ids)
                f.write(json.dumps({
                    "id": t_item["id"], "k_idx": t_item["k_idx"], "lead_k": t_item["lead_k"],
                    "domain": "science", "prompt": t_item["prompt"],
                    "prefix": tok.decode(t_item["prefix_ids"]),
                    "c10_continuation": o.text, "finish_reason": o.finish_reason,
                    "c10_n_tokens": len(o.token_ids),
                }) + "\n")
            f.flush()
            print(f"chunk {i//a.chunk+1}/{(len(tasks)+a.chunk-1)//a.chunk}: "
                  f"{len(batch)} in {dt:.0f}s | {ntok/dt:.1f} tok/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
