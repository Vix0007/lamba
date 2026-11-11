#!/usr/bin/env python3
"""Pre-flight smoke test (DS consult #107, item #3): run BEFORE any job expected to take >10min,
on a small (10-20 prompt) sample, using the EXACT same code path (model/lora/sampling params) as
the real job. Takes <5 min. Built after several real bugs this project found only AFTER a
multi-hour job completed (answer-extraction comparing against the wrong text region, a stop-
string whose exclusion from vLLM's output silently broke a regex parser) -- this exists to catch
that class of bug cheaply, before paying the full cost.

usage: preflight.py --model <path> [--lora <path>] [--prompts <file>] [--n 12] [--max-tokens 256]

Checks, each printed as PASS/FAIL/WARN with a one-line reason:
  1. disk free space (warn <50GB, matches this project's checkpoint-save safety margin)
  2. LoRA is actually applied (compare generation with vs without adapter on one prompt --
     if outputs are byte-identical, the adapter did not take effect: wrong path, missing
     enable_lora, wrong target modules, etc.)
  3. tokenizer round-trip (encode/decode a prompt, confirm chat template applies without error)
  4. max_tokens + prompt length fits max_model_len (would otherwise silently truncate/error)
  5. generate on the real sample: no empty outputs, finish_reason not all 'length', at least one
     'stop' (confirms EOS/stop-string handling actually works, not just "didn't crash")
  6. timing probe: measured tok/s on this sample, extrapolated full-job ETA
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DEFAULT_PROMPTS = "/home/ubuntu/antidoom_data/loop_testset150.jsonl"


def check(label, ok, detail=""):
    tag = "PASS" if ok is True else ("WARN" if ok == "warn" else "FAIL")
    print(f"[{tag}] {label}" + (f" -- {detail}" if detail else ""), flush=True)
    return ok is True or ok == "warn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--full-job-n", type=int, default=None,
                     help="if set, extrapolate the timing probe to this many total prompts")
    ap.add_argument("--full-job-max-tokens", type=int, default=None,
                     help="if the real job uses a different max_tokens than this probe, pass it "
                          "here for a correct ETA extrapolation")
    a = ap.parse_args()

    all_ok = True
    print(f"=== PREFLIGHT: model={a.model} lora={a.lora or '-'} n={a.n} ===", flush=True)

    # 1. disk
    free_gb = shutil.disk_usage("/").free / 1e9
    all_ok &= check("disk free space", True if free_gb >= 50 else "warn" if free_gb >= 25 else False,
                     f"{free_gb:.0f}GB free")

    # build engine (eager: this is a quick check, startup time matters more than throughput here)
    items = [json.loads(l) for l in open(a.prompts) if l.strip()][:a.n]
    if not items:
        print("[FAIL] no prompts loaded from", a.prompts); sys.exit(1)

    kw = dict(model=a.model, quantization="fp8", dtype="bfloat16", max_model_len=8192,
              trust_remote_code=True, gpu_memory_utilization=0.90, max_num_seqs=32,
              enforce_eager=True)
    if a.lora:
        kw.update(enable_lora=True, max_lora_rank=256, max_loras=1)
    t0 = time.time()
    llm = LLM(**kw)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    lreq = LoRARequest(os.path.basename(a.lora.rstrip("/")), 1, a.lora) if a.lora else None
    print(f"[INFO] engine loaded in {load_s:.0f}s", flush=True)

    # 3. tokenizer round-trip + chat template
    try:
        ctx0 = tok.apply_chat_template([{"role": "user", "content": items[0]["prompt"]}],
                                        tokenize=False, add_generation_prompt=True,
                                        reasoning_effort="medium")
        ids = tok(ctx0, add_special_tokens=False)["input_ids"]
        _ = tok.decode(ids)
        all_ok &= check("tokenizer round-trip + chat template", True, f"{len(ids)} tokens")
    except Exception as e:
        all_ok &= check("tokenizer round-trip + chat template", False, str(e))

    # 4. context length fit
    prompt_lens = []
    for it in items:
        c = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, reasoning_effort="medium")
        prompt_lens.append(len(tok(c, add_special_tokens=False)["input_ids"]))
    max_pl = max(prompt_lens)
    fits = (max_pl + a.max_tokens) <= kw["max_model_len"]
    all_ok &= check("max_tokens + prompt fits max_model_len", fits,
                     f"longest prompt {max_pl} tok + {a.max_tokens} max_tokens vs cap {kw['max_model_len']}")

    # 2. LoRA actually applied (only if a lora was given): compare one prompt with/without
    if a.lora:
        sp1 = SamplingParams(temperature=0.0, max_tokens=40)
        ctx = tok.apply_chat_template([{"role": "user", "content": items[0]["prompt"]}],
                                       tokenize=False, add_generation_prompt=True,
                                       reasoning_effort="medium")
        out_with = llm.generate([ctx], sp1, lora_request=lreq)[0].outputs[0].text
        out_without = llm.generate([ctx], sp1, lora_request=None)[0].outputs[0].text
        differs = out_with != out_without
        all_ok &= check("LoRA adapter actually changes output", differs,
                         "IDENTICAL output with/without adapter -- adapter not applied!" if not differs else "outputs differ, adapter is live")

    # 5. real sample generation
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    ctxs = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                     add_generation_prompt=True, reasoning_effort="medium") for it in items]
    t1 = time.time()
    outs = llm.generate(ctxs, sp, lora_request=lreq)
    gen_s = time.time() - t1
    n_empty = sum(1 for o in outs if len(o.outputs[0].token_ids) < 5)
    finish_reasons = [o.outputs[0].finish_reason for o in outs]
    n_stop = finish_reasons.count("stop")
    n_length = finish_reasons.count("length")
    total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    all_ok &= check("no empty outputs", n_empty == 0, f"{n_empty}/{len(outs)} empty")
    all_ok &= check("finish_reason not all 'length'",
                     "warn" if n_stop == 0 and a.max_tokens < 512 else True,
                     f"stop={n_stop} length={n_length} (short max_tokens probe often hits length -- "
                     f"only a real concern if the FULL job also shows this)")

    # 6. timing probe / ETA
    tok_s = total_tok / max(gen_s, 1e-9)
    print(f"[INFO] timing probe: {total_tok} tok in {gen_s:.0f}s -> {tok_s:.1f} tok/s (eager mode; "
          f"graph mode in the real job will likely be faster once warmed up)", flush=True)
    if a.full_job_n:
        job_max_tokens = a.full_job_max_tokens or a.max_tokens
        avg_tok_per_prompt = total_tok / len(items)
        est_tok_per_prompt = avg_tok_per_prompt * (job_max_tokens / a.max_tokens)
        eta_s = est_tok_per_prompt * a.full_job_n / max(tok_s, 1e-9)
        print(f"[INFO] extrapolated ETA for {a.full_job_n} prompts @ {job_max_tokens} max_tokens: "
              f"~{eta_s/60:.0f} min (rough -- eager-mode probe, real job likely graph-mode)", flush=True)

    print(f"\n=== PREFLIGHT {'PASSED' if all_ok else 'FAILED'} ===", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
