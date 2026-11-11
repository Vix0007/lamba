#!/usr/bin/env python3
"""Efficient-reasoning v3, step 0: fresh ON-POLICY self-rollouts from whichever checkpoint
v3 will init from (per DS #207b: dpov1 or dpov2, NOT c10 -- the pilot/v1/v2 all used c10's
own rollouts as "self", which was consistent with c10 being both source and target; v3
must do the same w.r.t. its own new init checkpoint for on-policy training to be valid).
Reuses the same malay-retroactive + science/mcp-teacher-paired prompt pools as before
(effreason_r2_selfgen_prompts.jsonl for science/mcp; distill_data_v4q_clean.jsonl's own
teacher rows give the malay prompts/teacher-answers directly, no separate self-rollout
needed there since malay already has real self rollouts in that file -- BUT those are
STILL c10's, so if full consistency is wanted, malay should also be regenerated fresh here
with --domains malay included; left as a flag so this can be re-run narrowly if only
science/mcp are regenerated).
Usage: gen_effreason_v3_selfrollouts.py <served_model_name> [--lora <path>] <out.jsonl> [--domains science,mcp,malay]
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/ubuntu/antidoom_data")
sys.path.insert(0, "/home/ubuntu/web")
from gen_a_fresh_toolloop import post_backend  # noqa: E402
import server as srv  # noqa: E402

MAX_TOKENS = 4000


def gen_one(prompt_text, served_model_name):
    body = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0, "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": srv.DEFAULT_CHAT_TEMPLATE_KWARGS,
    }
    resp = post_backend(body)
    msg = resp["choices"][0]["message"]
    return (msg.get("reasoning") or "") + "\n</think>\n\n" + (msg.get("content") or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("served_model_name")
    ap.add_argument("out_path")
    ap.add_argument("--domains", default="science,mcp")
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()
    domains = set(args.domains.split(","))

    prompts = []
    if {"science", "mcp"} & domains:
        for l in open("effreason_r2_selfgen_prompts.jsonl"):
            r = json.loads(l)
            if r["domain"] in domains:
                prompts.append(r)
    if "malay" in domains:
        seen = set()
        for l in open("/home/ubuntu/eval_results/distill_data_v4q_clean.jsonl"):
            r = json.loads(l)
            if r.get("domain") == "malay" and r["pilot_id"] not in seen and r["source"].startswith("teacher:"):
                seen.add(r["pilot_id"])
                prompts.append({"id": r["pilot_id"], "pilot_id": r["pilot_id"], "domain": "malay", "prompt": r["prompt"]})

    done = set()
    try:
        for l in open(args.out_path):
            done.add(json.loads(l)["id"])
    except FileNotFoundError:
        pass
    todo = [r for r in prompts if r["id"] not in done]
    print(f"{len(prompts)} prompts total, {len(done)} already done, {len(todo)} to go", flush=True)

    lock = threading.Lock()

    def run_one(r):
        text = gen_one(r["prompt"], args.served_model_name)
        row = {"id": r["id"], "pilot_id": r["pilot_id"], "domain": r["domain"], "text": text}
        with lock:
            with open(args.out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        return r["id"]

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(run_one, r) for r in todo]
        n = 0
        for fut in as_completed(futs):
            fut.result()
            n += 1
            if n % 25 == 0:
                print(f"[{n}/{len(todo)}] done", flush=True)

    print("GEN_EFFREASON_V3_SELFROLLOUTS_DONE", flush=True)


if __name__ == "__main__":
    main()
