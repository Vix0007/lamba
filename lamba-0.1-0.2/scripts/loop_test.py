"""Held-out loop test generation (resumable). Greedy, budget max_tokens, same engine flags for both arms."""
import argparse, json, os, time
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--lora", default=None); ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--eager", action="store_true"); ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=4000); ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--chunk", type=int, default=35); ap.add_argument("--max-num-seqs", type=int, default=72)
    a = ap.parse_args()
    out = Path.home() / "eval_results" / f"loop_{a.label}_{a.tag}.jsonl"
    items = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/loop_testset.jsonl")]
    done = set()
    if out.exists():
        for l in open(out):
            try: done.add(json.loads(l)["id"])
            except Exception: pass
    todo = [i for i in items if i["id"] not in done]
    print(f"[{a.label}/{a.tag}] {len(items)} prompts, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo: return
    kw = dict(model=a.model, quantization="fp8", dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=8192, trust_remote_code=True, enforce_eager=a.eager,
              enable_lora=True, max_lora_rank=256, max_loras=1, max_num_seqs=a.max_num_seqs)
    if a.mtp: kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1}
    t0 = time.time(); llm = LLM(**kw); tok = llm.get_tokenizer()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    lreq = LoRARequest("adapter", 1, a.lora) if a.lora else None
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    f = open(out, "a")
    for i in range(0, len(todo), a.chunk):
        batch = todo[i:i + a.chunk]
        ps = [tok.apply_chat_template([{"role": "user", "content": b["prompt"]}], tokenize=False,
                                      add_generation_prompt=True, reasoning_effort="medium") for b in batch]
        t1 = time.time(); res = llm.generate(ps, sp, lora_request=lreq)
        for b, r in zip(batch, res):
            o = r.outputs[0]
            f.write(json.dumps({"id": b["id"], "domain": b["domain"], "n_pairs": b["n_pairs"],
                                "finish_reason": o.finish_reason, "n_tokens": len(o.token_ids), "text": o.text}) + "\n")
        f.flush()
        print(f"chunk {i//a.chunk+1} done in {time.time()-t1:.0f}s ({len(batch)} prompts)", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
