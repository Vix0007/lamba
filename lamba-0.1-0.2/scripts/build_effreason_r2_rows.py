#!/usr/bin/env python3
"""Round-2 efficient-reasoning, step 3: assemble final loop_span rows using the
answer-ready-prefix onset (DS concern #3 fix) instead of the pilot's raw teacher-length proxy.

For each candidate, take the EARLIEST probe cut-fraction where a forced short answer from that
truncated prefix is still correct (matches the gold/teacher answer) -- that's the true point
where the model has "reasoned enough". Target = the model's own real continuation from there
(same mechanism as the pilot / loop-fix recipe: kind=loop_span, klmask loss, K<=32 tok tether).
"""
import json, re
from pathlib import Path
from transformers import AutoTokenizer

CANDS = Path.home() / "antidoom_data" / "effreason_r2_candidates.jsonl"
PROBES = Path.home() / "antidoom_data" / "effreason_r2_probe_results.jsonl"
OUT = Path.home() / "antidoom_data" / "effreason_r2_pairs.jsonl"

tok = AutoTokenizer.from_pretrained("/home/ubuntu/models/soup-linear-l1.0", trust_remote_code=True)


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
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"[#*`$\\{}]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def main():
    cands = {c["pilot_id"]: c for c in (json.loads(l) for l in open(CANDS))}
    probes = {}
    for l in open(PROBES):
        p = json.loads(l)
        probes.setdefault(p["pilot_id"], []).append(p)

    # FIX (after the first round-2 attempt regressed badly on loop metrics, especially science):
    # requiring only the SINGLE earliest correct probe accepted lucky guesses on MCQ-format items
    # (69% of the original 428 rows had onset_cut < 150 chars -- suppressing after a sentence or
    # two, before the model had actually engaged). Now require: (a) the probe point AND every
    # later probe point also correct ("stable to the end", not a one-off fluke reversed later),
    # (b) take the SECOND of the first stable-correct pair as onset (a little more margin), and
    # (c) an absolute floor (>=150 chars) so a lucky first-guess can't set the onset on its own.
    FLOOR_CHARS = 150

    rows = []
    n_no_correct_probe = n_cut_too_early = 0
    for pid, c in cands.items():
        plist = sorted(probes.get(pid, []), key=lambda p: p["frac"])
        gold = norm(c["gold"])
        correct_flags = [bool(norm(p["probe_answer"] + "}") == gold and gold) for p in plist]
        onset_cut = None
        for i in range(len(plist) - 1):
            if correct_flags[i] and correct_flags[i + 1] and all(correct_flags[i:]):
                cand_cut = plist[i + 1]["cut"]
                if cand_cut >= FLOOR_CHARS:
                    onset_cut = cand_cut
                    break
        if onset_cut is None:
            n_no_correct_probe += 1
            continue

        self_text = c["self_text"]
        prefix = self_text[:onset_cut]
        target = self_text[onset_cut:onset_cut + 400]
        if len(target) < 30:
            continue
        ctx = tok.apply_chat_template(
            [{"role": "user", "content": c["prompt"]}], tokenize=False,
            add_generation_prompt=True, reasoning_effort="medium"
        ) + prefix
        rows.append({
            "kind": "loop_span", "context": ctx, "target_text": target,
            "pilot_id": pid, "domain": c["domain"], "src": c["src"],
            "onset_cut": onset_cut, "teacher_len_proxy": c["teacher_len"],
            "self_len": len(self_text),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print(f"candidates: {len(cands)} | no correct probe at any fraction: {n_no_correct_probe} | "
          f"cut too early: {n_cut_too_early}")
    print(f"WRITTEN: {len(rows)} rows -> {OUT}")
    print("domain dist:", Counter(r["domain"] for r in rows))
    print("onset vs old-proxy ratio (mean):",
          sum(r["onset_cut"] / max(r["teacher_len_proxy"], 1) for r in rows) / len(rows))


if __name__ == "__main__":
    main()
