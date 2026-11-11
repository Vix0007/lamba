"""Ruminconfab onset rows: query letterfix2-c10's (not base's) top-20 next-token logprobs at each
onset context. Per DS #141 tether decision: since we're fixing behavior in the DEPLOYED
letterfix2-c10 stack and want to preserve everything else about it, both the reference (tether)
and the "what would this model actually consider instead" alternatives come from letterfix2-c10
itself, not base. Same prefill-only pattern as mine_alts.py (v7), just pointed at c10+LoRA.
"""
import json, os, sys, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

D = "/home/ubuntu/antidoom_data"


def main():
    a = sys.argv[1:]
    cp = a[a.index("--cands") + 1] if "--cands" in a else f"{D}/ruminconfab_onset_cands.jsonl"
    out_p = a[a.index("--out") + 1] if "--out" in a else f"{D}/ruminconfab_onset_alts.jsonl"
    cands = [json.loads(l) for l in open(cp)]
    done = set()
    if os.path.exists(out_p):
        for l in open(out_p):
            try:
                done.add(json.loads(l)["source_sample_id"])
            except Exception:
                pass
    todo = [c for c in cands if c["source_sample_id"] not in done]
    print(f"{len(cands)} candidates, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("DONE", flush=True)
        return
    llm = LLM(model="/home/ubuntu/models/lmh_letterfix2_c10",
              quantization="fp8", dtype="bfloat16", max_model_len=8192, trust_remote_code=True,
              gpu_memory_utilization=0.90, max_num_seqs=32, max_logprobs=64,
              enable_prefix_caching=False, enable_lora=True, max_lora_rank=256, max_loras=1)
    lreq = LoRARequest("letterfix2c10", 1, "/home/ubuntu/adapters_letterfix2_c10")
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    CH = 128
    t0 = time.time()
    with open(out_p, "a") as f:
        for i in range(0, len(todo), CH):
            b = todo[i:i + CH]
            res = llm.generate([c["context_with_chat_template"] for c in b], sp, lora_request=lreq)
            for c, r in zip(b, res):
                lp = r.outputs[0].logprobs[0]
                top = sorted(((tid, x.decoded_token, float(x.logprob)) for tid, x in lp.items()),
                             key=lambda z: -z[2])[:20]
                f.write(json.dumps({"source_sample_id": c["source_sample_id"], "top": top}) + "\n")
            f.flush()
            print(f"{i+len(b)}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print("ALTS DONE", flush=True)


if __name__ == "__main__":
    main()
