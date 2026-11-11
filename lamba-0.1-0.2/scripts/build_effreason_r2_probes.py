#!/usr/bin/env python3
"""Round-2 efficient-reasoning data build, step 1: correctness-filter candidate pairs
(malay retroactive + fresh science/mcp) and emit prefix-probe prompts for onset-finding.

Correctness gate (DS concern #2): self's own final answer must match the teacher's stated
answer after normalization (best available oracle -- no separate gold field in the corpus).
Length gate: self reasoned >=1.3x longer than teacher (same as the pilot).

Onset (DS concern #3): instead of the pilot's raw "cut at teacher-char-length" proxy, probe
several candidate cut points in self's own reasoning (as fractions of its own length, anchored
around the old teacher-length proxy) by truncating + forcing a short answer, and later (step 2)
pick the EARLIEST point where the forced answer is still correct.
"""
import json, re
from pathlib import Path

MALAY_SRC = Path.home() / "eval_results" / "distill_data_v4q_clean.jsonl"
SELFGEN = Path.home() / "eval_results" / "loop_effreasonR2c10_selfgen.jsonl"
PROMPTS = Path.home() / "antidoom_data" / "effreason_r2_selfgen_prompts.jsonl"
OUT_CANDS = Path.home() / "antidoom_data" / "effreason_r2_candidates.jsonl"
OUT_PROBES = Path.home() / "antidoom_data" / "effreason_r2_probes.jsonl"
MIN_EXCESS_RATIO = 1.3
PROBE_FRACS = [0.35, 0.55, 0.75, 1.0, 1.3]  # fraction of OLD teacher-length proxy cut
ANSWER_FORCE = "\n\nI have reasoned enough. My final answer is:\n\\boxed{"


def extract_final(text):
    """Best-effort final-answer extraction from a long free-form reasoning trace. Fixed after
    finding mcp's answer convention varies wildly (boxed / code-fence / bold / plain last line) --
    comparing the WHOLE raw text (the original bug) made almost every mcp pair look 'wrong'."""
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
    # -- malay retroactive candidates (teacher+self pairs already in the corpus) --
    rows = [json.loads(l) for l in open(MALAY_SRC) if l.strip()]
    by_pilot = {}
    for r in rows:
        by_pilot.setdefault(r["pilot_id"], {"teacher": [], "self": []})
        if r["source"].startswith("teacher:"):
            by_pilot[r["pilot_id"]]["teacher"].append(r)
        elif r["source"].startswith("self:"):
            by_pilot[r["pilot_id"]]["self"].append(r)

    cands = []
    for pid, d in by_pilot.items():
        if not d["teacher"] or not d["self"]:
            continue
        teacher = min(d["teacher"], key=lambda r: len(r["think"]))
        selfr = d["self"][0]
        ta, sa = norm(teacher.get("answer", "")), norm(selfr.get("answer", ""))
        if not ta or not sa or ta != sa:
            continue
        teacher_len = len(teacher["think"])
        self_think = selfr["think"]
        if len(self_think) < teacher_len * MIN_EXCESS_RATIO or len(self_think) < 200:
            continue
        cands.append({
            "pilot_id": pid, "domain": selfr.get("domain", "malay"), "prompt": selfr["prompt"],
            "self_text": self_think, "teacher_len": teacher_len, "gold": ta, "src": "malay_retro",
        })

    # -- fresh science/mcp self-rollouts vs their teacher rows --
    prompt_by_id = {r["id"]: r for r in (json.loads(l) for l in open(PROMPTS))}
    teacher_by_pid = {}
    for r in rows:
        if r["source"].startswith("teacher:"):
            teacher_by_pid.setdefault(r["pilot_id"], []).append(r)

    n_wrong = n_short = 0
    for line in open(SELFGEN):
        r = json.loads(line)
        pid = r["id"]
        tlist = teacher_by_pid.get(pid)
        if not tlist:
            continue
        teacher = min(tlist, key=lambda x: len(x["think"]))
        ta = norm(teacher.get("answer", ""))
        sa = norm(r["text"])  # final \boxed{} or trailing answer is at the tail of the raw text
        if not ta or not sa or ta != sa:
            n_wrong += 1
            continue
        teacher_len = len(teacher["think"])
        self_text = r["text"]
        if len(self_text) < teacher_len * MIN_EXCESS_RATIO or len(self_text) < 200:
            n_short += 1
            continue
        cands.append({
            "pilot_id": pid, "domain": prompt_by_id[pid]["domain"], "prompt": prompt_by_id[pid]["prompt"],
            "self_text": self_text, "teacher_len": teacher_len, "gold": ta, "src": "scimcp_fresh",
        })

    print(f"science/mcp fresh: wrong-answer {n_wrong}, too-short {n_short}, kept so far from this source: "
          f"{sum(1 for c in cands if c['src']=='scimcp_fresh')}")

    from collections import Counter
    print("candidate domain dist:", Counter(c["domain"] for c in cands))
    print("total candidates (correctness+length gated):", len(cands))

    with open(OUT_CANDS, "w") as f:
        for c in cands:
            f.write(json.dumps(c) + "\n")

    # -- emit probe prompts: for each candidate, PROBE_FRACS cut points --
    n_probes = 0
    with open(OUT_PROBES, "w") as f:
        for c in cands:
            self_text = c["self_text"]
            for frac in PROBE_FRACS:
                cut = int(c["teacher_len"] * frac)
                cut = min(cut, len(self_text) - 20)
                if cut < 30:
                    continue
                prefix = self_text[:cut]
                f.write(json.dumps({
                    "pilot_id": c["pilot_id"], "frac": frac, "cut": cut,
                    "prompt": c["prompt"], "prefix": prefix, "gold": c["gold"],
                }) + "\n")
                n_probes += 1
    print(f"WRITTEN: {len(cands)} candidates -> {OUT_CANDS}")
    print(f"WRITTEN: {n_probes} probes -> {OUT_PROBES}")


if __name__ == "__main__":
    main()
