#!/usr/bin/env python3
"""Efficient-reasoning v3, step 1: correctness-filter candidates (same gate as v1/v2: self's
own answer must match teacher's stated answer after normalization, self reasoned >=1.3x
longer than teacher) from FRESH on-policy self-rollouts (gen_effreason_v3_selfrollouts.py
output), then emit candidate PROBE POSITIONS per DS #207a's fixed design:
  - reasoning-step boundaries (paragraph breaks, "\n\n") in self's own text
  - PLUS up to 20 uniformly spaced points covering the FULL self-rollout length [0, L]
    (NOT bounded by the old teacher-length-proxy multiple -- that was the v1/v2 bug)
No tool-call boundaries here: this corpus (mcp/science/malay) is single-turn free-form
reasoning, no tool use, verified against real sample prompts this session.
"""
import json
import re
from pathlib import Path

SELFGEN = Path(__file__).parent / "effreason_v3_selfrollouts.jsonl"  # set by caller / symlink
TEACHER_SRC = Path.home() / "eval_results" / "distill_data_v4q_clean.jsonl"
OUT_CANDS = Path(__file__).parent / "effreason_v3_candidates.jsonl"
OUT_PROBEPOS = Path(__file__).parent / "effreason_v3_probe_positions.jsonl"
MIN_EXCESS_RATIO = 1.3
MAX_UNIFORM_POINTS = 20


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


def reasoning_step_boundaries(text):
    """Paragraph-break positions in the raw text -- a crude but real proxy for
    'reasoning step boundary' with no extra model calls needed."""
    return sorted({m.start() for m in re.finditer(r"\n\n+", text)})


def main():
    teacher_by_pid = {}
    for l in open(TEACHER_SRC):
        r = json.loads(l)
        if r["source"].startswith("teacher:"):
            teacher_by_pid.setdefault(r["pilot_id"], []).append(r)

    cands = []
    n_no_teacher = n_wrong = n_short = 0
    for l in open(SELFGEN):
        r = json.loads(l)
        pid = r["pilot_id"]
        tlist = teacher_by_pid.get(pid)
        if not tlist:
            n_no_teacher += 1
            continue
        teacher = min(tlist, key=lambda x: len(x["think"]))
        ta, sa = norm(teacher.get("answer", "")), norm(r["text"])
        if not ta or not sa or ta != sa:
            n_wrong += 1
            continue
        teacher_len = len(teacher["think"])
        self_text = r["text"]
        if len(self_text) < teacher_len * MIN_EXCESS_RATIO or len(self_text) < 200:
            n_short += 1
            continue
        cands.append({"pilot_id": pid, "domain": r["domain"], "self_text": self_text,
                       "gold": ta, "teacher_len": teacher_len})

    print(f"no-teacher-match {n_no_teacher}, wrong-answer {n_wrong}, too-short {n_short}, "
          f"kept {len(cands)}")
    from collections import Counter
    print("domain dist:", Counter(c["domain"] for c in cands))

    with open(OUT_CANDS, "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")

    # need the original prompt text too -- pull from the selfgen prompt pools
    prompts = {}
    for l in open("effreason_r2_selfgen_prompts.jsonl"):
        pr = json.loads(l)
        prompts[pr["pilot_id"]] = pr["prompt"]
    for l in open(TEACHER_SRC):
        r = json.loads(l)
        if r["source"].startswith("teacher:") and r["pilot_id"] not in prompts:
            prompts[r["pilot_id"]] = r["prompt"]

    n_probes = n_skipped_no_prompt = 0
    with open(OUT_PROBEPOS, "w") as f:
        for c in cands:
            pid = c["pilot_id"]
            prompt = prompts.get(pid)
            if not prompt:
                n_skipped_no_prompt += 1
                continue
            text = c["self_text"]
            L = len(text)
            step_pts = [p for p in reasoning_step_boundaries(text) if 30 <= p <= L - 20]
            uniform_pts = sorted({int(L * i / MAX_UNIFORM_POINTS) for i in range(1, MAX_UNIFORM_POINTS + 1)
                                   if 30 <= int(L * i / MAX_UNIFORM_POINTS) <= L - 20})
            all_pts = sorted(set(step_pts) | set(uniform_pts))
            for cut in all_pts:
                f.write(json.dumps({"pilot_id": pid, "cut": cut, "L": L, "domain": c["domain"],
                                     "prompt": prompt, "prefix": text[:cut], "gold": c["gold"]}) + "\n")
                n_probes += 1
    print(f"WRITTEN: {len(cands)} candidates -> {OUT_CANDS}")
    print(f"WRITTEN: {n_probes} probe positions ({n_skipped_no_prompt} candidates skipped, no prompt found) -> {OUT_PROBEPOS}")


if __name__ == "__main__":
    main()
