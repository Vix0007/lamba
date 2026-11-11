#!/usr/bin/env python3
"""Corrected version of gen_gemini_teacher_gpqa.py: teachers.py's _call_gemini hardcodes
thinkingConfig.thinkingBudget=0 (built for structured-JSON judging elsewhere in this project,
where thinking just eats budget) -- wrong for OUR purpose, which needs genuine (but efficient)
reasoning traces as real distillation material, not bare answers with no shown work.

Calls the Vertex endpoint directly (bypassing teachers._call_gemini) with
thinkingConfig.thinkingLevel="LOW" (Gemini 3.x's real parameter -- NOT the old thinkingBudget int;
passing both errors, per gemini_agent_loop.py's own note) and captures BOTH the thought parts (as
'reasoning') and the final answer text (as 'content') -- same two-field shape as our own model's
rollouts, so this is directly comparable/usable as a distillation pair source.
"""
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request
import urllib.error

sys.path.insert(0, "/home/ubuntu/s3_reads")
import teachers  # reuse _sa_load()/_vertex_token() for auth only

MODEL_ID = os.environ.get("GEMINI_TEACHER_MODEL", "gemini-3.8-flash")
THINKING_LEVEL = os.environ.get("GEMINI_TEACHER_THINKING", "LOW")  # LOW = efficient-but-real reasoning
PROMPTS_SRC = Path.home() / "eval_results" / "gpqa_s3_pull" / "letterfix2c10_gpqa_full198_samples.jsonl"
OUT = Path.home() / "eval_results" / "gemini_teacher_gpqa198.jsonl"
CONCURRENCY = 4
WRITE_LOCK = threading.Lock()

SYSTEM = ("You are answering a hard multiple-choice science question (GPQA-Diamond style). "
          "Be correct and efficient -- do not re-verify a conclusion you already trust, do not "
          "restate the problem, do not pad. End with exactly one line: "
          "'FINAL ANSWER: (X)' where X is A, B, C, or D.")


def call_gemini_with_thinking(prompt, tries=6):
    sa = teachers._sa_load()
    url = (f"https://aiplatform.googleapis.com/v1/projects/{sa['project_id']}"
           f"/locations/global/publishers/google/models/{MODEL_ID}:generateContent")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": SYSTEM + "\n\n" + prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature": 0.2,
            "thinkingConfig": {"thinkingLevel": THINKING_LEVEL, "includeThoughts": True},
        },
    }
    for attempt in range(tries):
        try:
            token = teachers._vertex_token()
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            cand = (d.get("candidates") or [{}])[0]
            parts = ((cand.get("content") or {}).get("parts") or [])
            reasoning = "\n".join(p.get("text", "") for p in parts if p.get("text") and p.get("thought"))
            content = "\n".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought"))
            return reasoning, content
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                wait = min(20 * (attempt + 1), 120)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body[:300]}")
        except Exception as e:
            if attempt < tries - 1:
                time.sleep(5)
                continue
            raise RuntimeError(f"{type(e).__name__}: {e}")
    raise RuntimeError("exhausted retries")


def extract_letter(reasoning, content):
    text = content or reasoning or ""
    m = re.search(r"FINAL ANSWER:\s*\(?([A-D])\)?", text, re.I)
    if m:
        return f"({m.group(1).upper()})"
    m = re.findall(r"\(([A-D])\)", text)
    return f"({m[-1].upper()})" if m else None


def run_one(item):
    t0 = time.time()
    try:
        reasoning, content = call_gemini_with_thinking(item["problem"])
        pred = extract_letter(reasoning, content)
        rec = {"record_id": item["record_id"], "domain": item.get("domain"), "gold": item["gold"],
               "pred": pred, "correct": pred == item["gold"], "reasoning": reasoning, "content": content,
               "n_reasoning_chars": len(reasoning), "n_content_chars": len(content),
               "thinking_level": THINKING_LEVEL, "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        rec = {"record_id": item["record_id"], "domain": item.get("domain"), "gold": item.get("gold"),
               "error": str(e)[:300], "reasoning": "", "content": "", "correct": False,
               "n_reasoning_chars": 0, "n_content_chars": 0, "elapsed_s": round(time.time() - t0, 1)}
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
    print(f"{len(items)} prompts, {len(done)} done, {len(todo)} to go "
          f"(model={MODEL_ID}, thinking={THINKING_LEVEL}, concurrency={CONCURRENCY})", flush=True)
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
                      f"reasoning_chars={rec.get('n_reasoning_chars')} content_chars={rec.get('n_content_chars')} "
                      f"({rec.get('elapsed_s','?')}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
