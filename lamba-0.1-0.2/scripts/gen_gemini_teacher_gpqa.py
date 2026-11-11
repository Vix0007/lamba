#!/usr/bin/env python3
"""Distillation data generation: query Gemini (stronger teacher) for CONCISE, correct solutions
to the same 198 GPQA-D prompts our own model has been evaluated on. These become 'chosen' targets
for FTPO/soft-KL distillation pairs (chosen = Gemini's concise-correct answer style, rejected =
our own model's long/budget-cut completions on the same prompt, already pulled from S3).
Concurrent (per project's "always concurrent" rule), resumable.
"""
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/s3_reads")
import teachers

MODEL_FAMILY = "gemini"  # tries gemini-3.8-flash then gemini-3.7-flash (FAMILY dict order)
PROMPTS_SRC = Path.home() / "eval_results" / "gpqa_s3_pull" / "letterfix2c10_gpqa_full198_samples.jsonl"
OUT = Path.home() / "eval_results" / "gemini_teacher_gpqa198.jsonl"
CONCURRENCY = 4  # keep modest -- this SA hit 429 RESOURCE_EXHAUSTED earlier today under heavier load
WRITE_LOCK = threading.Lock()

SYSTEM = ("You are answering a hard multiple-choice science question (GPQA-Diamond style). "
          "Think only as much as needed to be CONFIDENT and CORRECT, then stop -- do not "
          "re-verify a conclusion you already trust, do not restate the problem, do not hedge. "
          "End your answer with exactly one line: 'FINAL ANSWER: (X)' where X is A, B, C, or D.")


def call(prompt, tries=5):
    for attempt in range(tries):
        try:
            txt, prov = teachers.call_text(MODEL_FAMILY, SYSTEM, prompt, temp=0.2, max_tokens=4000, tries=1)
            return txt, prov
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = min(20 * (attempt + 1), 120)
                time.sleep(wait)
                continue
            if attempt < tries - 1:
                time.sleep(5)
                continue
            raise
    raise RuntimeError("exhausted retries")


def extract_letter(text):
    m = re.search(r"FINAL ANSWER:\s*\(?([A-D])\)?", text, re.I)
    if m:
        return f"({m.group(1).upper()})"
    m = re.findall(r"\(([A-D])\)", text)
    return f"({m[-1].upper()})" if m else None


def run_one(item):
    t0 = time.time()
    try:
        txt, prov = call(item["problem"])
        pred = extract_letter(txt)
        rec = {"record_id": item["record_id"], "domain": item.get("domain"), "gold": item["gold"],
               "pred": pred, "correct": pred == item["gold"], "text": txt,
               "n_chars": len(txt), "provider": prov, "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        rec = {"record_id": item["record_id"], "domain": item.get("domain"), "gold": item.get("gold"),
               "error": str(e)[:300], "text": "", "correct": False, "n_chars": 0,
               "elapsed_s": round(time.time() - t0, 1)}
    return rec


def main():
    items = [json.loads(l) for l in open(PROMPTS_SRC) if l.strip()]
    done = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["record_id"])
            except Exception:
                pass
    todo = [i for i in items if i["record_id"] not in done]
    print(f"{len(items)} prompts, {len(done)} done, {len(todo)} to go (concurrency={CONCURRENCY})", flush=True)
    n_done = 0
    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = {pool.submit(run_one, item): item for item in todo}
        for fut in as_completed(futs):
            item = futs[fut]
            rec = fut.result()
            with WRITE_LOCK:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_done += 1
                print(f"[{n_done}/{len(todo)}] {item['record_id']} correct={rec.get('correct')} "
                      f"n_chars={rec.get('n_chars')} ({rec.get('elapsed_s','?')}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
