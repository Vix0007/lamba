#!/usr/bin/env python3
"""Offline shadow-replay of DS #155's runtime rumination guard against transcripts we
already collected (A3_08 family check + repro check, all 3 arms). No GPU needed --
pure replay/analysis of saved transcripts. Implements DS's should_fire logic and
computes false-stop rate / precision / recall / calls-saved per DS's 5.1 validation plan.
"""
import json
import re
import glob

DENY_KEYS = {
    "id", "uuid", "timestamp", "time", "date", "elapsed", "duration", "latency",
    "cost", "tokens", "page", "limit", "offset", "count", "total_count", "n",
    "status", "code", "version", "round", "index", "idx", "rank", "position",
    "size", "length", "http_status", "status_code",
}
ANSWER_KEYS = {"answer", "result", "value", "output", "solution", "total",
               "estimate", "probability", "mean", "median", "rate", "score"}
SUCCESS_RE = re.compile(r"(?i)\b(match|matches|matched|agree|agrees|agreed|consistent|"
                         r"equal|same|confirmed|verified|true|pass|passed|ok)\b")
FAIL_RE = re.compile(r"(?i)\b(mismatch|mismatched|disagree|inconsistent|not equal|"
                      r"different|false|fail|failed)\b")
VERIFY_TOOL_RE = re.compile(r"(?i)(verify|check|compare|validate|confirm|assert|test|cross.?check|sanity)")
VERIFY_INTENT_RE = re.compile(r"(?i)\b(verify|verification|check|cross[- ]?check|double[- ]?check|"
                               r"confirm|validate|sanity[- ]?check|make sure|ensure)\b")
NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def extract_numbers(text):
    if not text:
        return []
    out = []
    for m in NUM_RE.finditer(text):
        try:
            v = float(m.group())
        except ValueError:
            continue
        if 1900 <= v <= 2100 and '.' not in m.group():
            continue
        out.append(v)
    return out


def agree(a, b):
    return abs(a - b) <= max(1e-9, 1e-6 * max(abs(a), abs(b)))


def fingerprint(text):
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"\b\d{10,}\b", "<id>", t)
    return t


class GuardState:
    def __init__(self):
        self.candidate_value = None
        self.candidate_active = False
        self.candidate_confirmed = False
        self.verification_requested = False
        self.verification_completed = False
        self.no_new_info_streak = 0
        self.exact_duplicate_count = 0
        self.seen_fingerprints = set()
        self.seen_values = []
        self.fired = False
        self.fire_round = None


def replay(transcript, max_rounds=8):
    """Replay a transcript tool-round by tool-round, applying DS's should_fire logic.
    Returns (fire_round or None, state) -- fire_round is the tool-round index (0-based,
    counting assistant-with-tool-calls turns) at which the guard would have fired."""
    state = GuardState()
    round_idx = 0
    for i, turn in enumerate(transcript):
        role = turn.get("role")
        content = turn.get("content") or ""

        if role == "assistant" and turn.get("tool_calls"):
            if VERIFY_INTENT_RE.search(content):
                state.verification_requested = True

        if role == "tool":
            nums = extract_numbers(content)
            is_verify_tool = bool(VERIFY_TOOL_RE.search(content))
            has_success = bool(SUCCESS_RE.search(content))
            has_fail = bool(FAIL_RE.search(content))
            fp = fingerprint(content)

            is_dup = fp in state.seen_fingerprints
            new_value = False
            for v in nums:
                if not any(agree(v, sv) for sv in state.seen_values):
                    new_value = True
                    state.seen_values.append(v)

            if is_dup and not new_value:
                state.exact_duplicate_count += 1
                state.no_new_info_streak += 1
            elif not new_value and nums:
                # numbers repeated but not exact fingerprint dup -> still "no new info"
                state.no_new_info_streak += 1
            else:
                state.no_new_info_streak = 0
            state.seen_fingerprints.add(fp)

            # candidate salience (simplified DS scoring)
            score = 0
            if len(nums) == 1:
                score += 2
            if new_value:
                score += 1
            if len(nums) >= 1 and len(content) < 400:
                score += 1  # proxy for "echoed"/concise answer-shaped output

            if nums and score >= 2:
                cand = nums[0]
                if state.candidate_value is not None and agree(cand, state.candidate_value):
                    state.candidate_confirmed = True
                state.candidate_value = cand
                state.candidate_active = True

            if is_verify_tool and state.verification_requested:
                if has_success and not has_fail:
                    state.verification_completed = True
                elif has_fail:
                    state.candidate_active = False
                    state.candidate_confirmed = False

            # should_fire check (after processing this tool round)
            if state.candidate_active and not state.fired and round_idx >= 2:
                if state.verification_completed or state.candidate_confirmed:
                    would_fire = state.no_new_info_streak >= 1 or state.exact_duplicate_count >= 1
                else:
                    would_fire = state.no_new_info_streak >= 2 or state.exact_duplicate_count >= 2
                if would_fire:
                    state.fired = True
                    state.fire_round = round_idx
            round_idx += 1
    return state.fire_round, state


def gt_match_round(transcript, needles):
    round_idx = 0
    for turn in transcript:
        if turn.get("role") == "tool":
            c = turn.get("content") or ""
            if any(n in c for n in needles):
                return round_idx
            round_idx += 1
        elif turn.get("role") == "assistant" and turn.get("tool_calls"):
            pass
    return None


def numeric_str_variants(v, ndigits=6):
    s = f"{v:.{ndigits}f}"
    out = {s}
    for n in range(1, ndigits):  # BUG FIX 2026-10-01: was range(2,..), missed clean 1-decimal values
        out.add(f"{v:.{n}f}")
    return out


GTS = {
    "Aexp_A3_08": 4.574563760717865, "A3_08v1": 2.1166086673, "A3_08v2": 3.1415926536,
    "A3_08v3": 0.7182818285, "A3_08v4": 0.4674011003, "A3_08v5": 7.1058605854,
    "A3_08v6": 1.1512925465,
}


def main():
    files = sorted(glob.glob("/home/ubuntu/eval_results/a3_08_family_*.jsonl"))
    total = collections_init = 0
    stats = {}
    for fpath in files:
        arm = fpath.split("a3_08_family_")[1].replace(".jsonl", "")
        rows = [json.loads(l) for l in open(fpath)]
        n_fired = 0
        n_false_stop = 0  # fired before gt appears at all
        n_fired_after_gt = 0
        n_fired_precision_ok = 0  # fired after gt, and gt value is the candidate
        calls_saved = []
        n_would_help = 0  # classification was post_correct_rumination and guard fires after gt
        n_missed_pcr = 0  # pcr case where guard never fires
        n_intermediate_risk = 0  # fired but classification shows task wasn't actually done (never_correct/other) -- false stop risk

        for r in rows:
            gts = GTS.get(r["id"])
            if gts is None:
                continue
            needles = numeric_str_variants(gts)
            t = r["transcript"]
            gt_round = gt_match_round(t, needles)
            fire_round, state = replay(t)
            n_tool_rounds = sum(1 for turn in t if turn.get("role") == "tool")

            if fire_round is not None:
                n_fired += 1
                if gt_round is None:
                    n_false_stop += 1
                elif fire_round < gt_round:
                    n_false_stop += 1
                else:
                    n_fired_after_gt += 1
                    if state.candidate_value is not None and agree(state.candidate_value, gts):
                        n_fired_precision_ok += 1
                    saved = n_tool_rounds - 1 - fire_round
                    if saved > 0:
                        calls_saved.append(saved)

            if r["classification"] == "post_correct_rumination":
                if fire_round is not None and gt_round is not None and fire_round >= gt_round:
                    n_would_help += 1
                else:
                    n_missed_pcr += 1

            if fire_round is not None and gt_round is None:
                n_intermediate_risk += 1

        n_pcr = sum(1 for r in rows if r["classification"] == "post_correct_rumination")
        stats[arm] = dict(
            n=len(rows), n_fired=n_fired, n_false_stop=n_false_stop,
            n_fired_after_gt=n_fired_after_gt, n_fired_precision_ok=n_fired_precision_ok,
            mean_calls_saved=round(sum(calls_saved) / len(calls_saved), 2) if calls_saved else 0,
            n_pcr=n_pcr, n_would_help=n_would_help, n_missed_pcr=n_missed_pcr,
            n_intermediate_risk=n_intermediate_risk,
        )

    for arm, s in stats.items():
        print(f"=== {arm} (n={s['n']}, {s['n_pcr']} known post_correct_rumination cases) ===")
        print(f"  guard fired: {s['n_fired']}/{s['n']}")
        print(f"  false-stop (fired before GT ever appeared): {s['n_false_stop']}")
        print(f"  fired after GT appeared: {s['n_fired_after_gt']}  (precision on those: "
              f"{s['n_fired_precision_ok']}/{s['n_fired_after_gt'] if s['n_fired_after_gt'] else 1})")
        print(f"  mean tool-calls saved (when fired after GT, vs actual completion): {s['mean_calls_saved']}")
        print(f"  of {s['n_pcr']} pcr cases: guard would have caught {s['n_would_help']}, missed {s['n_missed_pcr']}")
        print(f"  fired-but-GT-never-appeared (risk of stopping a genuinely unsolved task): {s['n_intermediate_risk']}")
        print()


if __name__ == "__main__":
    main()
