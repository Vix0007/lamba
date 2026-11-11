"""Serving-speed benchmark on the BASE model (no LoRA), fixed workload: 64 loop-prone prompts x 256 greedy tokens.
usage: serve_bench.py <tag>   tags: graphs | graphs_mtp_nopc"""
import json, sys, time
from vllm import LLM, SamplingParams

TAG = sys.argv[1]
CFG = {
 "graphs":          dict(enforce_eager=False, gpu_memory_utilization=0.90,
                         compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}),
 "graphs_mtp_nopc": dict(enforce_eager=False, gpu_memory_utilization=0.92, enable_prefix_caching=False,
                         compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
                         speculative_config={"method": "mtp", "num_speculative_tokens": 1}),
}[TAG]

def main():
    items = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/loop_testset.jsonl")][:64]
    t0 = time.time()
    llm = LLM(model="/home/ubuntu/models/soup-linear-l1.0", quantization="fp8", dtype="bfloat16", max_model_len=8192,
              trust_remote_code=True, max_num_seqs=64, **CFG)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    ps = [tok.apply_chat_template([{"role": "user", "content": i["prompt"]}], tokenize=False, add_generation_prompt=True,
                                  reasoning_effort="medium") for i in items]
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    llm.generate(ps[:4], SamplingParams(temperature=0.0, max_tokens=16))          # warmup
    t1 = time.time(); outs = llm.generate(ps, sp); dt = time.time() - t1
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print("BENCH_RESULT " + json.dumps({"tag": TAG, "load_s": round(load_s), "gen_s": round(dt, 1), "tokens": ntok,
                                        "agg_tok_s": round(ntok / dt, 1)}), flush=True)

if __name__ == "__main__":
    main()
