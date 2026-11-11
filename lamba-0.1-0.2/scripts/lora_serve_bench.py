"""Does a 10-layer round-2 LoRA fit vLLM CUDA-graph mode? Base model + enable_lora (rank 256), graphs FULL_DECODE_ONLY, 64 prompts x 256 tok:
 (a) without LoRA request, (b) with the converted round-2 smoke adapter. Reports KV cache tokens (from engine log), agg tok/s, and whether outputs differ."""
import json, sys, time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
def main():
    util = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
    mtp = len(sys.argv) > 2 and sys.argv[2] == "mtp"
    items = [json.loads(l) for l in open("/home/ubuntu/antidoom_data/loop_testset.jsonl")][:64]
    t0 = time.time()
    llm = LLM(model="/home/ubuntu/models/soup-linear-l1.0", quantization="fp8", dtype="bfloat16", max_model_len=8192, trust_remote_code=True,
              gpu_memory_utilization=util, max_num_seqs=64, enable_lora=True, max_lora_rank=256, max_loras=1,
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
              **({"speculative_config": {"method": "mtp", "num_speculative_tokens": 1}, "enable_prefix_caching": False} if mtp else {}))
    print(f"LOADED mtp={mtp} util={util} in {time.time()-t0:.0f}s", flush=True)
    tok = llm.get_tokenizer()
    ps = [tok.apply_chat_template([{"role": "user", "content": i["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium") for i in items]
    sp = SamplingParams(temperature=0.0, max_tokens=256); lreq = LoRARequest("r2smoke", 1, "/home/ubuntu/adapters_r2smoke")
    llm.generate(ps[:4], SamplingParams(temperature=0.0, max_tokens=16))
    res = {}
    for name, lr in (("base_no_lora", None), ("with_lora", lreq)):
        t1 = time.time(); outs = llm.generate(ps, sp, lora_request=lr); dt = time.time() - t1
        n = sum(len(o.outputs[0].token_ids) for o in outs); res[name] = (outs, n / dt)
        print("LORA_BENCH " + json.dumps({"mode": name, "agg_tok_s": round(n / dt, 1), "tokens": n, "secs": round(dt, 1)}), flush=True)
    diff = sum(a.outputs[0].text != b.outputs[0].text for a, b in zip(res["base_no_lora"][0], res["with_lora"][0]))
    print("LORA_BENCH " + json.dumps({"outputs_differing_with_adapter": diff, "of": len(ps)}), flush=True)
if __name__ == "__main__":
    main()
