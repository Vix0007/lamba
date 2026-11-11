"""v7 step 2 (GPU): prefill-only top-20 next-token logprobs from the BASE model at each candidate context (max_tokens=1). Resumable jsonl append."""
import json, os, sys, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
D = "/home/ubuntu/antidoom_data"
def main():
    cands = [json.loads(l) for l in open(f"{D}/antidoom7_cands.jsonl")]
    out_p = f"{D}/antidoom7_alts.jsonl"; done = set()
    if os.path.exists(out_p):
        for l in open(out_p):
            try: done.add(json.loads(l)["source_sample_id"])
            except Exception: pass
    todo = [c for c in cands if c["source_sample_id"] not in done]
    print(f"{len(cands)} candidates, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo: return
    llm = LLM(model="/home/ubuntu/models/soup-linear-l1.0", quantization="fp8", dtype="bfloat16", max_model_len=8192, trust_remote_code=True, gpu_memory_utilization=0.90, max_num_seqs=32,
              max_logprobs=64, enable_prefix_caching=False)
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    CH = 256; t0 = time.time()
    with open(out_p, "a") as f:
        for i in range(0, len(todo), CH):
            b = todo[i:i + CH]
            res = llm.generate([c["context_with_chat_template"] for c in b], sp)
            for c, r in zip(b, res):
                lp = r.outputs[0].logprobs[0]
                top = sorted(((tid, x.decoded_token, float(x.logprob)) for tid, x in lp.items()), key=lambda z: -z[2])[:20]
                f.write(json.dumps({"source_sample_id": c["source_sample_id"], "top": top}, ensure_ascii=False) + "\n")
            f.flush(); print(f"{i+len(b)}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print("ALTS DONE", flush=True)
if __name__ == "__main__":
    main()
