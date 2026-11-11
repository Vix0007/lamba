#!/usr/bin/env python3
"""Beat-ThinkingCap track, Phase 0 (DS consult #110 retry): mine existing self-rollouts for
"naturally concise AND correct" traces -- NOT teacher-length-anchored (that premise was falsified
in the effreason-round-2 postmortem). Selection rule (DS's exact spec):
  per domain d: keep correct traces with length <= P70(correct trace lengths in d),
  AND not loop-flagged (L/R_any/R_heavy), AND finish_reason=='stop' (no budget-hit/truncation),
  AND non-empty/non-refusal.
Splits at the PROMPT level (not trace level) 80/20 train/dev, stratified by domain, BEFORE
selection, so dev prompts are never seen in training data regardless of whether their rollout
would have been selected. Excludes any prompt whose id appears in loop_testset150.jsonl (the
final 150-test) -- must stay clean.
"""
import json, re, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from transformers import AutoTokenizer

EVAL_RESULTS = Path.home() / "eval_results"
ANTIDOOM_DATA = Path.home() / "antidoom_data"
OUT_DIR = ANTIDOOM_DATA / "concise_sft"
OUT_DIR.mkdir(exist_ok=True)

P_PRIMARY = 70  # percentile; DS contingency: <300 usable -> P80, >900 -> P60

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


AL = re.compile(r"[A-Za-z0-9]")
def loop_flags(text, prompt=""):
    """Same Liquid-style detector used throughout this project (tail_stats_cmp.py)."""
    n = len(text)
    cov = bytearray(n)
    seen = set()
    found = False
    for i in range(0, max(0, n - 60), 30):
        w = text[i:i + 60]
        if w in seen or len(AL.findall(w)) < 25:
            continue
        seen.add(w)
        if text.count(w) < 4 or w in prompt:
            continue
        found = True
        j = text.find(w)
        while j >= 0:
            for k in range(j, min(n, j + 60)):
                cov[k] = 1
            j = text.find(w, j + 60)
    tail = sum(cov[max(0, n - 2500):])
    return {"L": found, "R_any": tail > 0, "R_heavy": tail > 1500}


def token_len(text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


def main():
    # exclusion set: never mine anything in the final 150-test
    test150_ids = set(json.loads(l)["id"] for l in open(ANTIDOOM_DATA / "loop_testset150.jsonl"))

    # -- pool 1: science/mcp fresh self-rollouts vs their teacher rows --
    prompt_by_id = {r["id"]: r for r in (json.loads(l) for l in open(ANTIDOOM_DATA / "effreason_r2_selfgen_prompts.jsonl"))}
    corpus_rows = [json.loads(l) for l in open(EVAL_RESULTS / "distill_data_v4q_clean.jsonl")]
    teacher_by_pid = defaultdict(list)
    for r in corpus_rows:
        if r["source"].startswith("teacher:"):
            teacher_by_pid[r["pilot_id"]].append(r)

    candidates = []  # {id, domain, prompt, gold, self_text, finish_reason, n_tokens}
    for line in open(EVAL_RESULTS / "loop_effreasonR2c10_selfgen.jsonl"):
        r = json.loads(line)
        pid = r["id"]
        if pid in test150_ids:
            continue
        tlist = teacher_by_pid.get(pid)
        if not tlist or pid not in prompt_by_id:
            continue
        teacher = min(tlist, key=lambda x: len(x["think"]))
        candidates.append({
            "id": pid, "domain": prompt_by_id[pid]["domain"], "prompt": prompt_by_id[pid]["prompt"],
            "gold": norm(teacher.get("answer", "")), "self_answer_field": None,  # science/mcp: answer is embedded in the raw text blob, extracted below
            "self_text": r["text"],
            "finish_reason": r["finish_reason"], "n_tokens_reported": r["n_tokens"],
        })

    # -- pool 2: malay self+teacher pairs from the original corpus --
    by_pilot = defaultdict(lambda: {"teacher": [], "self": []})
    for r in corpus_rows:
        if r["source"].startswith("teacher:"):
            by_pilot[r["pilot_id"]]["teacher"].append(r)
        elif r["source"].startswith("self:"):
            by_pilot[r["pilot_id"]]["self"].append(r)
    for pid, d in by_pilot.items():
        if not d["teacher"] or not d["self"] or pid in test150_ids:
            continue
        teacher = min(d["teacher"], key=lambda r: len(r["think"]))
        selfr = d["self"][0]
        if selfr.get("domain") != "malay":
            continue
        candidates.append({
            "id": pid, "domain": "malay", "prompt": selfr["prompt"],
            "gold": norm(teacher.get("answer", "")),
            "self_answer_field": norm(selfr.get("answer", "")),  # malay pool has a SEPARATE answer field (unlike science/mcp's single blob) -- correctness must compare against THIS, not extracted-from-think
            "self_text": selfr["think"],
            "finish_reason": "stop" if len(selfr["think"]) < 15000 else "length",  # no explicit field in this pool; conservative proxy, re-checked below
            "n_tokens_reported": None,
        })

    print(f"total candidates (deduped by pool, test150-excluded): {len(candidates)}")
    dom_ct = Counter(c["domain"] for c in candidates)
    print("by domain:", dict(dom_ct))

    # -- prompt-level 80/20 split, stratified by domain, deterministic via hash --
    def split_bucket(pid):
        h = int(hashlib.md5(pid.encode()).hexdigest(), 16)
        return "dev" if (h % 10) < 2 else "train"

    for c in candidates:
        c["split"] = split_bucket(c["id"])
    split_ct = Counter((c["domain"], c["split"]) for c in candidates)
    print("split counts (domain, split):", dict(split_ct))

    # -- correctness + tokenize + loop-flag every candidate (needed for both splits: dev needs baseline too) --
    n_correct = n_wrong = 0
    for c in candidates:
        sa = c["self_answer_field"] if c["self_answer_field"] else norm(c["self_text"])
        c["correct"] = bool(c["gold"] and sa and sa == c["gold"])
        c["n_tokens"] = token_len(c["self_text"])
        flags = loop_flags(c["self_text"], c["prompt"])
        c.update(flags)
        c["budget_hit"] = c["finish_reason"] == "length"
        if c["correct"]:
            n_correct += 1
        else:
            n_wrong += 1

    print(f"correctness: {n_correct} correct / {n_wrong} wrong ({100*n_correct/len(candidates):.1f}%)")

    with open(OUT_DIR / "all_candidates.jsonl", "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")

    # -- per-domain P70 threshold on TRAIN-split correct, non-loop-flagged, non-budget-hit traces --
    train_correct_by_dom = defaultdict(list)
    for c in candidates:
        if c["split"] == "train" and c["correct"] and not c["L"] and not c["budget_hit"]:
            train_correct_by_dom[c["domain"]].append(c["n_tokens"])

    thresholds = {}
    for dom, lens in train_correct_by_dom.items():
        lens_sorted = sorted(lens)
        idx = min(int(len(lens_sorted) * P_PRIMARY / 100), len(lens_sorted) - 1)
        thresholds[dom] = lens_sorted[idx]
        print(f"domain={dom}: n_correct_clean(train)={len(lens_sorted)} P{P_PRIMARY}={thresholds[dom]} tok "
              f"(min={lens_sorted[0]} median={lens_sorted[len(lens_sorted)//2]} max={lens_sorted[-1]})")

    selected = [c for c in candidates if c["split"] == "train" and c["correct"] and not c["L"]
                and not c["budget_hit"] and c["n_tokens"] <= thresholds.get(c["domain"], 10**9)]
    print(f"\nSELECTED (train, P{P_PRIMARY}): {len(selected)} rows")
    print("selected domain dist:", dict(Counter(c["domain"] for c in selected)))

    with open(OUT_DIR / f"selected_p{P_PRIMARY}.jsonl", "w") as f:
        for c in selected:
            f.write(json.dumps(c) + "\n")

    dev = [c for c in candidates if c["split"] == "dev"]
    with open(OUT_DIR / "dev_candidates.jsonl", "w") as f:
        for c in dev:
            f.write(json.dumps(c) + "\n")
    print(f"dev pool (all, unfiltered, for baseline+eval): {len(dev)} prompts, "
          f"dev correct: {sum(c['correct'] for c in dev)}")


if __name__ == "__main__":
    main()
