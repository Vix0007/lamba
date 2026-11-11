#!/usr/bin/env python3
"""GPQA-D capability comparison: gemini-3.8-flash vs our own lamba runs, matched to the SAME
b4000 protocol used everywhere else in this project (gpqa_ab_g.py --budget 4000): same 198
GPQA-D problems, greedy (temp=0), a hard 4000-token output cap, neutral prompt (NOT the
"be efficient" framing used by the earlier teacher-distillation scripts -- this is a capability
comparison, not distillation-data mining). Gemini 3.x has no literal thinking-token-budget knob
(only qualitative thinkingLevel), so maxOutputTokens=4000 is the matched cap and thinkingLevel=HIGH
lets it use as much of that budget as it wants, same spirit as our own model's uncapped-until-4000
greedy reasoning. Captures thought parts (reasoning) and final text (content) separately.
Concurrent + resumable, per project convention.
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

MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
PROMPTS_SRC = Path.home() / "eval_results" / "gpqa_s3_pull" / "letterfix2c10_gpqa_full198_samples.jsonl"
OUT = Path.home() / "eval_results" / "gemini38_gpqa_b4000.jsonl"
CONCURRENCY = 4
WRITE_LOCK = threading.Lock()
BUDGET = 4000  # matches --budget 4000 used for every lamba GPQA-D run in this project

SYSTEM = ("You are answering a hard multiple-choice science question (GPQA-Diamond style). "
          "Reason through the problem, then end your answer with exactly one line: "
          "'FINAL ANSWER: (X)' where X is A, B, C, or D.")


def _post(payload, tries=6):
    sa = teachers._sa_load()
    url = (f"https://aiplatform.googleapis.com/v1/projects/{sa['project_id']}"
           f"/locations/global/publishers/google/models/{MODEL_ID}:generateContent")
    for attempt in range(tries):
        try:
            token = teachers._vertex_token()
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.loads(r.read())
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


def call_gemini(prompt):
    """Two-stage protocol, matching gpqa_ab_g.py exactly: stage 1 reasons up to BUDGET
    tokens; if it hits MAX_TOKENS without a clean 'FINAL ANSWER:' line, stage 2 makes a
    SEPARATE short forced-answer follow-up call (continuing the same reasoning context),
    not a crude regex scrape of truncated text."""
    payload1 = {
        "contents": [{"role": "user", "parts": [{"text": SYSTEM + "\n\n" + prompt}]}],
        "generationConfig": {
            "maxOutputTokens": BUDGET,
            "temperature": 0.0,
            "thinkingConfig": {"thinkingLevel": "HIGH", "includeThoughts": True},
        },
    }
    d1 = _post(payload1)
    cand = (d1.get("candidates") or [{}])[0]
    finish_reason = cand.get("finishReason", "")
    parts = ((cand.get("content") or {}).get("parts") or [])
    reasoning = "\n".join(p.get("text", "") for p in parts if p.get("text") and p.get("thought"))
    content = "\n".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought"))
    usage1 = d1.get("usageMetadata", {})

    stage2_used = False
    if re.search(r"FINAL ANSWER:\s*\(?[A-D]\)?", content, re.I):
        return reasoning, content, finish_reason, usage1, stage2_used

    # Stage 2: forced short answer, continuing the SAME context (thinking budget already spent,
    # so no thinkingConfig here -- mirrors our own stage2 which is plain greedy, no reasoning).
    stage2_used = True
    followup_text = reasoning + ("\n" + content if content else "")
    payload2 = {
        "contents": [
            {"role": "user", "parts": [{"text": SYSTEM + "\n\n" + prompt}]},
            {"role": "model", "parts": [{"text": followup_text[-6000:] or "(no reasoning captured)"}]},
            {"role": "user", "parts": [{"text": "Based on your reasoning above, give your final answer now. "
                                                 "Respond with EXACTLY one line: 'FINAL ANSWER: (X)' where X "
                                                 "is A, B, C, or D. No other text."}]},
        ],
        "generationConfig": {"maxOutputTokens": 100, "temperature": 0.0},
    }
    d2 = _post(payload2)
    cand2 = (d2.get("candidates") or [{}])[0]
    parts2 = ((cand2.get("content") or {}).get("parts") or [])
    content2 = "\n".join(p.get("text", "") for p in parts2 if p.get("text"))
    usage2 = d2.get("usageMetadata", {})
    usage_combined = {"stage1": usage1, "stage2": usage2}
    return reasoning, content + "\n" + content2, finish_reason, usage_combined, stage2_used


def extract_letter(content):
    m = re.search(r"FINAL ANSWER:\s*\(?([A-D])\)?", content, re.I)
    if m:
        return f"({m.group(1).upper()})"
    return None  # stage2 failed to produce a clean answer -- mark unparseable, do NOT guess


def run_one(item):
    t0 = time.time()
    try:
        reasoning, content, finish_reason, usage, stage2_used = call_gemini(item["problem"])
        pred = extract_letter(content)
        rec = {"record_id": item["record_id"], "domain": item.get("domain"), "gold": item["gold"],
               "pred": pred, "correct": pred == item["gold"], "forced": stage2_used,
               "finish_reason": finish_reason, "reasoning": reasoning, "content": content,
               "n_reasoning_chars": len(reasoning), "n_content_chars": len(content),
               "usage": usage, "elapsed_s": round(time.time() - t0, 1)}
    except Exception as e:
        rec = {"record_id": item["record_id"], "domain": item.get("domain"), "gold": item.get("gold"),
               "error": str(e)[:300], "reasoning": "", "content": "", "correct": False,
               "forced": True, "elapsed_s": round(time.time() - t0, 1)}
    return rec


def main():
    items = [json.loads(l) for l in open(PROMPTS_SRC) if l.strip()]
    # PROMPTS_SRC rows carry the FULL chat-template-rendered lamba prompt; we just need the
    # underlying GPQA question text + options, which is embedded in it. Re-derive a clean
    # question string once, cache alongside record_id so re-runs don't re-parse.
    for it in items:
        if "problem" not in it:
            # extract everything between the last <|im_start|>user and the assistant marker
            p = it["prompt"]
            m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", p, re.S)
            it["problem"] = m.group(1).strip() if m else p

    done = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        for l in open(OUT):
            try:
                done.add(json.loads(l)["record_id"])
            except Exception:
                pass
    todo = [it for it in items if it["record_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    with open(OUT, "a") as f, ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(run_one, it): it for it in todo}
        n = 0
        for fut in as_completed(futs):
            rec = fut.result()
            with WRITE_LOCK:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            n += 1
            ok = "OK" if rec.get("correct") else ("ERR" if rec.get("error") else "WRONG")
            print(f"[{n}/{len(todo)}] {rec['record_id']} {ok} forced={rec.get('forced')} "
                  f"elapsed={rec['elapsed_s']}s", flush=True)

    # summary
    rows = [json.loads(l) for l in open(OUT)]
    correct = sum(1 for r in rows if r.get("correct"))
    forced = sum(1 for r in rows if r.get("forced"))
    errors = sum(1 for r in rows if r.get("error"))
    print(f"GEMINI38_GPQA_B4000_DONE: {correct}/{len(rows)} correct, forced={forced}, errors={errors}",
          flush=True)


if __name__ == "__main__":
    main()
