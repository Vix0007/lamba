#!/usr/bin/env python3
"""Round-2 efficient-reasoning, step 2: run the prefix probes (DS's answer-ready-prefix onset
fix). For each candidate x cut-fraction, truncate self's own reasoning at that point, force a
short answer completion, and record whether it's still correct. Step 3 (assemble) picks the
EARLIEST correct cut per candidate as the true onset, instead of the pilot's raw teacher-length
proxy.
"""
import json, os, re, time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

PROBES = Path.home() / "antidoom_data" / "effreason_r2_probes.jsonl"
OUT = Path.home() / "antidoom_data" / "effreason_r2_probe_results.jsonl"
MODEL = "/home/ubuntu/models/lmh_letterfix2_c10"
LORA = "/home/ubuntu/adapters_letterfix2_c10"
FORCE = "\n\nI have reasoned enough. My final answer is:\n\\boxed{"


def extract_final(text):
    text = text or ""
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1]
    fences = re.findall(r"```(?:\w*\n)?(.*?)```", text, flags=re.S)
    if fences:
        return fences[-1]
    tail = text[-200:]
    bolds = re.findall(r"\*\*([^*]+)\*\*", tail)
    if bolds:
        return bolds[-1]
    m2 = re.search(r"answer\s*:?\s*(.+)", tail, flags=re.I | re.S)
    if m2:
        return m2.group(1)
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def norm(s):
    s = extract_final(s or "")
    s = re.sub(r"^\s*answer\s*:?\s*", "", s.strip(), flags=re.I)
    s = re.sub(r"[#*`$]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def main():
    items = [json.loads(l) for l in open(PROBES) if l.strip()]
    done = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                r = json.loads(l)
                done.add((r["pilot_id"], r["frac"]))
            except Exception:
                pass
    todo = [i for i in items if (i["pilot_id"], i["frac"]) not in done]
    print(f"{len(items)} probes, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("DONE", flush=True)
        return

    kw = dict(model=MODEL, quantization="fp8", dtype="bfloat16", max_model_len=8192,
              trust_remote_code=True, gpu_memory_utilization=0.90, max_num_seqs=64,
              enable_lora=True, max_lora_rank=256, max_loras=1,
              compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                   "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]})
    t0 = time.time()
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    lreq = LoRARequest("adapter", 1, LORA)
    sp = SamplingParams(temperature=0.0, max_tokens=80, stop=["}"])

    CH = 256
    with open(OUT, "a") as f:
        for i in range(0, len(todo), CH):
            batch = todo[i:i + CH]
            ctxs = []
            for b in batch:
                base = tok.apply_chat_template(
                    [{"role": "user", "content": b["prompt"]}], tokenize=False,
                    add_generation_prompt=True, reasoning_effort="medium")
                ctxs.append(base + b["prefix"] + FORCE)
            t1 = time.time()
            outs = llm.generate(ctxs, sp, lora_request=lreq)
            for b, o in zip(batch, outs):
                ans = "\\boxed{" + o.outputs[0].text
                correct = bool(b["gold"]) and norm(ans) == norm(b["gold"])
                f.write(json.dumps({"pilot_id": b["pilot_id"], "frac": b["frac"], "cut": b["cut"],
                                    "correct": correct, "probe_answer": ans[:120]}) + "\n")
            f.flush()
            print(f"{min(i+CH, len(todo))}/{len(todo)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
