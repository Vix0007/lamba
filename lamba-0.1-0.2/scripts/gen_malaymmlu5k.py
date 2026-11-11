"""Generative (free-text, brief-reasoning-then-answer) MalayMMLU-5k scoring -- NOT the project's
usual strict single-letter-token protocol. Built to get a genuinely comparable number against
ilmu-v3.1's direct-answer 83.23% (both YTL's press-release 86.98% and ilmu's measured number use
some form of free generation, not single-token logprob scoring). Pure eval-tooling addition, no
training/methodology decision -- follows the exact model-loading pattern already used by
loop_test_g.py (graph mode, resumable, same engine flags) to stay consistent with the rest of the
project's tooling and avoid re-deriving vLLM config from scratch.
Usage: gen_malaymmlu5k.py --model <dir> --lora <dir|None> --label <label> [--max-tokens 800]
"""
import argparse, json, os, re, time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DATA = "/home/ubuntu/datasets/MalayMMLU_0shot.json"
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32]
ANSWER_RE = re.compile(r"Jawapan\s*:\s*\(?\s*([A-Ea-e])\s*\)?", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{\s*\(?([A-Ea-e])\)?\s*\}")


def extract_pred(text):
    # search the LAST 400 chars first (final answer is near the end), fall back to whole text
    tail = text[-400:]
    for rx in (ANSWER_RE, BOXED_RE):
        matches = rx.findall(tail)
        if matches:
            return matches[-1].upper()
    for rx in (ANSWER_RE, BOXED_RE):
        matches = rx.findall(text)
        if matches:
            return matches[-1].upper()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--chunk", type=int, default=96)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    a = ap.parse_args()

    items = json.load(open(DATA))
    out = Path.home() / "eval_results" / f"malaymmlu5k_generative_{a.label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [it for it in items if it["id"] not in done]
    print(f"[{a.label}] {len(items)} items, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    kw = dict(model=a.model, quantization="fp8", dtype="bfloat16", max_model_len=8192,
              trust_remote_code=True, gpu_memory_utilization=a.gpu_util, max_num_seqs=64,
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                   "cudagraph_capture_sizes": CAPTURE_SIZES})
    if a.lora:
        kw.update(enable_lora=True, max_lora_rank=256, max_loras=1)
    t0 = time.time()
    llm = LLM(**kw)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    lreq = LoRARequest(os.path.basename(a.lora.rstrip("/")), 1, a.lora) if a.lora else None
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    PROMPT_TMPL = (
        "{prompt}\n\n"
        "Fikirkan secara ringkas (1-3 ayat), kemudian nyatakan jawapan akhir anda dalam "
        "format tepat: Jawapan: <huruf>"
    )

    total_tok, total_time, n_correct, n_scored = 0.0, 0.0, 0, 0
    with open(out, "a") as f:
        for i in range(0, len(todo), a.chunk):
            batch = todo[i:i + a.chunk]
            ps = [tok.apply_chat_template(
                [{"role": "user", "content": PROMPT_TMPL.format(prompt=b["prompt"])}],
                tokenize=False, add_generation_prompt=True, reasoning_effort="low") for b in batch]
            t1 = time.time()
            res = llm.generate(ps, sp, lora_request=lreq)
            dt = time.time() - t1
            ntok = 0
            for b, r in zip(batch, res):
                o = r.outputs[0]
                n = len(o.token_ids)
                ntok += n
                pred = extract_pred(o.text)
                correct = int(pred == b["key"]) if pred else 0
                n_scored += 1
                n_correct += correct
                f.write(json.dumps({"id": b["id"], "subject": b.get("subject"),
                                    "key": b["key"], "pred": pred, "correct": correct,
                                    "finish_reason": o.finish_reason, "n_tokens": n,
                                    "content": o.text}) + "\n")
            f.flush()
            total_tok += ntok
            total_time += dt
            print(f"chunk {i//a.chunk+1}: {len(batch)} items in {dt:.0f}s | "
                  f"{ntok/dt:.1f} tok/s | running acc {n_correct}/{n_scored} "
                  f"({100*n_correct/n_scored:.1f}%)", flush=True)

    print(f"ALL: {total_tok:.0f} tok in {total_time:.0f}s -> {total_tok/max(total_time,1e-9):.1f} tok/s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
