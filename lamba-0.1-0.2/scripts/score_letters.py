"""Generic base letter scorer: score_letters.py <candidates.jsonl> <out_scores.jsonl>  (vLLM eager, base fp8, next-token logprobs of ' A'..' E')."""
import json, sys, time
from vllm import LLM, SamplingParams
def main():
    cands = [json.loads(l) for l in open(sys.argv[1])]
    llm = LLM(model="/home/ubuntu/models/soup-linear-l1.0", quantization="fp8", dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=8192,
              trust_remote_code=True, enforce_eager=True, max_logprobs=64)
    tok = llm.get_tokenizer(); lid = {L: tok(" " + L, add_special_tokens=False)["input_ids"][0] for L in "ABCDE"}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=60)
    with open(sys.argv[2], "w") as f:
        for i in range(0, len(cands), 512):
            b = cands[i:i + 512]; outs = llm.generate([c["context"] for c in b], sp)
            for c, o in zip(b, outs):
                lp = o.outputs[0].logprobs[0]; f.write(json.dumps({"id": c["id"], "lp": {L: (lp[t].logprob if t in lp else -30.0) for L, t in lid.items()}}) + "\n")
            f.flush(); print(f"scored {min(i+512,len(cands))}/{len(cands)}", flush=True)
    print("DONE", flush=True)
if __name__ == "__main__":
    main()
