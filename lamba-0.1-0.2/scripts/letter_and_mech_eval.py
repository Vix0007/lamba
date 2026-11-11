"""ONE model load, two jobs (base lamba, fp8, eager, no LoRA):
 (1) score letter logprobs for every letter candidate -> ~/antidoom_data/letter_scored.jsonl
 (2) MalayMMLU-5k position-vs-content experiment on the BASE: as-is, inline-format, and cyclic option shifts
     -> ~/eval_results/mech_base_malay5k.jsonl   (analyse with mech_analyze.py)
Resumable per job: skips a job whose output already has all rows."""
import json, os, re, sys, time
from pathlib import Path
from vllm import LLM, SamplingParams

MODEL = "/home/ubuntu/models/soup-linear-l1.0"
CAND = "/home/ubuntu/antidoom_data/letter_candidates.jsonl"
OUT1 = "/home/ubuntu/antidoom_data/letter_scored.jsonl"
OUT2 = str(Path.home() / "eval_results" / "mech_base_malay5k.jsonl")
MM = str(Path.home() / "datasets" / "MalayMMLU_0shot.json")

def nlines(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0

def main():
    cands = [json.loads(l) for l in open(CAND)]
    mm = json.load(open(MM))
    todo1 = nlines(OUT1) < len(cands)
    # MalayMMLU variants
    items = []
    for it in mm:
        n = len(it["options"]); texts = [re.sub(r"^[A-J]\.\s*", "", o) for o in it["options"]]
        stem = it["prompt"].split("\nA.")[0] if "\nA." in it["prompt"] else None
        if stem is None or n < 2: continue
        g = ord(it["key"]) - 65
        if g >= n: continue
        variants = [("asis", it["prompt"] + "\nAnswer:", list(range(n)))]
        variants.append(("inline", stem + " " + " ".join(f"{chr(65+i)}. {t}" for i, t in enumerate(texts)) + "\nAnswer:", list(range(n))))
        for s in range(1, n):
            order = [(i + s) % n for i in range(n)]
            body = stem + "\n" + "\n".join(f"{chr(65+i)}. {texts[order[i]]}" for i in range(n))
            variants.append((f"shift{s}", body + "\nAnswer:", order))
        for name, ctx, order in variants:
            items.append({"id": it["id"], "variant": name, "context": ctx, "order": order, "gold_orig": g, "n": n,
                          "subject": it["subject_eng"]})
    todo2 = nlines(OUT2) < len(items)
    print(f"job1 letter candidates: {len(cands)} (todo={todo1}) | job2 mech contexts: {len(items)} (todo={todo2})", flush=True)
    if not (todo1 or todo2): return
    t0 = time.time()
    llm = LLM(model=MODEL, quantization="fp8", dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=8192,
              trust_remote_code=True, enforce_eager=True, max_logprobs=64)
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)
    tok = llm.get_tokenizer()
    lid = {L: tok(" " + L, add_special_tokens=False)["input_ids"][0] for L in "ABCDE"}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=60)

    def score(contexts):
        outs = llm.generate(contexts, sp)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            res.append({L: (lp[t].logprob if t in lp else -30.0) for L, t in lid.items()})
        return res

    if todo1:
        with open(OUT1, "w") as f:
            for i in range(0, len(cands), 512):
                b = cands[i:i + 512]; r = score([c["context"] for c in b])
                for c, s in zip(b, r): f.write(json.dumps({"id": c["id"], "lp": s}) + "\n")
                f.flush(); print(f"job1 {min(i+512,len(cands))}/{len(cands)}", flush=True)
    if todo2:
        with open(OUT2, "w") as f:
            for i in range(0, len(items), 1024):
                b = items[i:i + 1024]; r = score([x["context"] for x in b])
                for x, s in zip(b, r):
                    f.write(json.dumps({"id": x["id"], "variant": x["variant"], "order": x["order"], "gold_orig": x["gold_orig"],
                                        "n": x["n"], "subject": x["subject"], "lp": s}) + "\n")
                f.flush(); print(f"job2 {min(i+1024,len(items))}/{len(items)}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
