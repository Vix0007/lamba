"""DS #136 repair-SFT mining, stage 2: scan the K-rollout jsonl(s), find completions that
loop, extract the onset position (first token of the 2nd+ occurrence of the repeated span,
same detector as tail_stats_*.py / #135's diagnostic), and write repair candidates: (prompt,
prefix-through-onset, onset_char_pos, full looping text for reference).

Usage: detect_repair_candidates.py <rollouts.jsonl...> --out <candidates.jsonl>
"""
import argparse
import json
import re

al = re.compile(r"[A-Za-z0-9]")


def find_onset(t: str, prompt: str):
    """Returns the char position of the first token of the 2nd occurrence of the earliest
    qualifying repeated span, or None if no loop detected. Same detector as elsewhere in
    this project (60-char window, >=25 alnum chars, >=4 occurrences, not verbatim-in-prompt)."""
    n = len(t)
    seen = set()
    best = None
    for i in range(0, n - 60, 30):
        w = t[i:i + 60]
        if w in seen or len(al.findall(w)) < 25:
            continue
        seen.add(w)
        if t.count(w) < 4 or w in prompt:
            continue
        first = t.find(w)
        second = t.find(w, first + 60)
        if second > 0 and (best is None or second < best):
            best = second
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout_files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n_prompts = 0
    n_completions = 0
    n_candidates = 0
    with open(a.out, "w") as fout:
        for path in a.rollout_files:
            for line in open(path):
                if not line.strip():
                    continue
                row = json.loads(line)
                n_prompts += 1
                prompt = row["prompt"]
                for k_idx, comp in enumerate(row["completions"]):
                    n_completions += 1
                    text = comp["text"]
                    onset = find_onset(text, prompt)
                    if onset is None:
                        continue
                    n_candidates += 1
                    fout.write(json.dumps({
                        "id": row["id"], "k_idx": k_idx, "domain": row.get("domain", "science"),
                        "prompt": prompt, "full_text": text, "onset_char_pos": onset,
                        "n_tokens": comp["n_tokens"], "finish_reason": comp["finish_reason"],
                    }) + "\n")
    print(f"prompts={n_prompts} completions={n_completions} candidates={n_candidates} "
          f"yield={n_candidates/max(1,n_completions)*100:.2f}%")


if __name__ == "__main__":
    main()
