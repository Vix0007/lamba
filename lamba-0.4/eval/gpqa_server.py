#!/usr/bin/env python3
"""GPQA-Diamond budget-forced eval against an OpenAI-compatible server (SGLang or vLLM).
Port of ref_eval_scripts/gpqa_ab_g.py: same docs, same prompt (chat template, reasoning_effort=xhigh,
"Let's think step by step: "), same stage 1 (greedy, max_tokens=budget, stop </think>, repetition_penalty),
same stage 2 (prompt + reasoning + "</think>\\n\\nBased on my analysis, the final answer is (", greedy, 50 tok),
same scoring and output format. Resumable.
usage: gpqa_server.py --model-name NAME --tokenizer DIR --tag TAG [--port 8001] [--budget 4000] [--rp 1.0] [--conc 16] [--compare-to base_samples.jsonl]
"""
import argparse, json, math, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model-name", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--port", type=int, default=8001); ap.add_argument("--budget", type=int, default=4000)
ap.add_argument("--rp", type=float, default=1.0); ap.add_argument("--conc", type=int, default=16); ap.add_argument("--compare-to")
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/completions"
GPQA_JSON = "/home/ubuntu/VROPD/pilot/evaldata/gpqa_diamond.json"
OUT = Path.home() / "eval_results" / f"{a.tag}_gpqa_full198_samples.jsonl"

def load_docs():
    docs = []
    for i, r in enumerate(json.load(open(GPQA_JSON))):
        m = re.search(r"\\boxed\{\s*([A-Da-d])\s*\}", r["solution"])
        docs.append({"problem": r["problem"], "answer": f"({m.group(1).upper()})", "id": f"ungated-{i}", "domain": r.get("domain")})
    return docs

tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
def build_prompt(doc):
    return tok.apply_chat_template([{"role": "user", "content": doc["problem"] + "\nLet's think step by step: "}],
                                   tokenize=False, add_generation_prompt=True, reasoning_effort="xhigh")

def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=7200).read())

def one(doc):
    prompt = build_prompt(doc)
    d1 = post({"model": a.model_name, "prompt": prompt, "max_tokens": a.budget, "temperature": 0.0,
               "stop": ["</think>"], "repetition_penalty": a.rp})
    c1 = d1["choices"][0]; reasoning = c1["text"]; fin = c1["finish_reason"]
    if reasoning.endswith("</think>"): reasoning = reasoning[: -len("</think>")]
    d2 = post({"model": a.model_name, "prompt": prompt + reasoning + "</think>\n\nBased on my analysis, the final answer is (",
               "max_tokens": 50, "temperature": 0.0, "stop": ["<|im_end|>", "<|endoftext|>"]})
    final = d2["choices"][0]["text"]
    m = re.search(r"[A-Za-z]", final); pred = f"({m.group(0).upper()})" if m else None
    return {"record_id": doc["id"], "domain": doc["domain"], "problem": doc["problem"], "reasoning": reasoning,
            "reasoning_finish_reason": fin, "forced_cutoff": fin != "stop", "final_answer_text": final,
            "gold": doc["answer"], "pred": pred, "correct": pred == doc["answer"],
            "reasoning_tokens": d1.get("usage", {}).get("completion_tokens")}

docs = load_docs()
done = set()
if OUT.exists():
    for l in open(OUT):
        try: done.add(json.loads(l)["record_id"])
        except Exception: pass
todo = [d for d in docs if d["id"] not in done]
print(f"[{a.tag}] {len(docs)} docs, {len(done)} done, {len(todo)} todo; budget {a.budget} rp {a.rp} conc {a.conc}", flush=True)
with ThreadPoolExecutor(a.conc) as ex, open(OUT, "a") as f:
    for i, r in enumerate(ex.map(one, todo), 1):
        f.write(json.dumps(r) + "\n"); f.flush()
        if i % 20 == 0: print(f"[{a.tag}] {i}/{len(todo)}", flush=True)

rows = {json.loads(l)["record_id"]: json.loads(l) for l in open(OUT)}
n = len(rows); c = sum(bool(r["correct"]) for r in rows.values()); fc = sum(bool(r["forced_cutoff"]) for r in rows.values())
fin_ok = sum(bool(r["correct"]) for r in rows.values() if not r["forced_cutoff"])
print(f"  accuracy: {c/n:.4f} ({c}/{n})\n  forced cutoffs: {fc}/{n}\n  no-forcing score (cutoff=0): {fin_ok}/{n}")
if a.compare_to:
    b = {json.loads(l)["record_id"]: json.loads(l) for l in open(a.compare_to)}
    common = set(rows) & set(b); bo = sum(1 for k in common if b[k]["correct"] and not rows[k]["correct"]); ao = sum(1 for k in common if rows[k]["correct"] and not b[k]["correct"])
    k = min(ao, bo); p = min(1.0, 2 * sum(math.comb(ao + bo, i) for i in range(k + 1)) / 2 ** (ao + bo)) if ao + bo else 1.0
    print(f"  vs base: base {sum(bool(b[x]['correct']) for x in common)}/{len(common)} | this-only {ao} base-only {bo} | McNemar p={p:.4g}")
print("GPQA_SERVER_DONE")
