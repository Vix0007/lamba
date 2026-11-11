"""One vLLM engine (base model, LoRA enabled), several LoRA adapters: run the 24-prompt x 1200-token gate for each. usage: gate_multi.py <label_prefix> <adapter_dir1> [<adapter_dir2> ...]
Writes ~/eval_results/loop_<prefix><basename>_gate.jsonl per adapter and prints: mean tokens, empty outputs, newline ratio vs base, repeat-6gram, contiguous-tail loops."""
import json, os, re, statistics as st, sys, time
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
pre = sys.argv[1]; dirs = sys.argv[2:]
A = "/home/ubuntu/antidoom_data"; R = "/home/ubuntu/eval_results"
items = [json.loads(l) for l in open(f"{A}/loop_gate24.jsonl") if l.strip()]
base = {}
for l in open(f"{R}/loop_base_g150.jsonl"): r = json.loads(l); base[r["id"]] = r
def rep(t):
    w = t.split()
    if len(w) < 50: return 0.0
    sh = [" ".join(w[i:i + 6]) for i in range(len(w) - 5)]; return 1 - len(set(sh)) / len(sh)
_al = re.compile(r"[A-Za-z0-9]")
def liquid_loop(t, W=60, S=30):
    for i in range(0, len(t) - W, S):
        w = t[i:i + W]
        if len(_al.findall(w)) < 25: continue
        if t.count(w) >= 4: return True
    return False
def tail_loop(text, pmin=15):
    n = len(text)
    if n < 300: return False
    tail = text[-6000:]; m = len(tail)
    for p in range(pmin, min(1200, m // 2) + 1):
        if tail[m - p:] != tail[m - 2 * p:m - p]: continue
        i = m - p - 1
        while i >= 0 and tail[i] == tail[i + p]: i -= 1
        if m - (i + 1) >= 2 * p - 5 and len(tail[i + 1:i + 1 + p].strip()) >= 10 and not re.fullmatch(r"[\W\d_]+", tail[i + 1:i + 1 + p]): return True
    return False
def main():
    llm = LLM(model="/home/ubuntu/models/soup-linear-l1.0", quantization="fp8", dtype="bfloat16", max_model_len=8192, trust_remote_code=True, gpu_memory_utilization=0.90, max_num_seqs=64,
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}, enable_lora=True, max_lora_rank=256, max_loras=1)
    tok = llm.get_tokenizer(); sp = SamplingParams(temperature=0.0, max_tokens=1200)
    prompts = [tok.apply_chat_template([{"role": "user", "content": b["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium") for b in items]
    ids = [b["id"] for b in items]
    T = lambda t: t[:4800]
    bt = [base[i]["text"] for i in ids]
    nn_b = st.mean(T(t).count("\n\n") / max(1, len(T(t))) * 1000 for t in bt); n1_b = st.mean(T(t).count("\n") / max(1, len(T(t))) * 1000 for t in bt); rp_b = st.mean(rep(T(t)) for t in bt)
    print(f"BASE ref: '\\n\\n' {nn_b:.2f}/1000ch '\\n' {n1_b:.2f} repeat6 {rp_b:.3f} tail-loops(first4800ch) {sum(tail_loop(T(t)) for t in bt)} liquid-loops {sum(liquid_loop(t) for t in bt)}/24", flush=True)
    for k, d in enumerate(dirs):
        name = pre + os.path.basename(d.rstrip("/")); t0 = time.time()
        res = llm.generate(prompts, sp, lora_request=LoRARequest(name, k + 1, d))
        rows = [{"id": i, "domain": b.get("domain"), "finish_reason": r.outputs[0].finish_reason, "n_tokens": len(r.outputs[0].token_ids), "text": r.outputs[0].text} for i, b, r in zip(ids, items, res)]
        with open(f"{R}/loop_{name}_gate.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tx = [r["text"] for r in rows]
        nn = st.mean(T(t).count("\n\n") / max(1, len(T(t))) * 1000 for t in tx); n1 = st.mean(T(t).count("\n") / max(1, len(T(t))) * 1000 for t in tx)
        print(f"GATE {name}: mean_tokens {st.mean(r['n_tokens'] for r in rows):.0f} empty(<20tok) {sum(r['n_tokens'] < 20 for r in rows)}/24 stop {sum(r['finish_reason']=='stop' for r in rows)} | newline ratio nn {100*nn/max(nn_b,1e-9):.0f}% n1 {100*n1/max(n1_b,1e-9):.0f}% | repeat6 {st.mean(rep(T(t)) for t in tx):.3f} (base {rp_b:.3f}) | tail-loops {sum(tail_loop(t) for t in tx)} | liquid-loops {sum(liquid_loop(t) for t in tx)}/24 | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
