#!/usr/bin/env python3
"""DS #165's stress-test variants 1/2/5 (the pure DETECTION-logic checks -- no model
generation needed, since these only test GuardState.observe_tool_round()'s fire/no-fire
decision against synthetic/injected tool_output text). Variants 3/4/6 need REAL model
generation (what does the model actually SAY when forced to finalize) and are NOT covered
here -- those require live GPU calls against a served model, queued separately.

Variant 1: I correct exact, C correct -> expect eventual fire is fine (normal case),
   just confirming no weirdness.
Variant 2: I correct but notation-varies (erf(sqrt(2)) vs 1.414..., pi/4 vs 0.785398,
   1/2 vs 0.5, ln(4)/2 vs ln(2)) -- tests whether agree()/extract_final_numeric_lines
   recognizes the SAME numeric value across different textual renderings, so the guard's
   "no new info" streak logic doesn't get confused by notation alone.
Variant 5: I wrong (a clearly incorrect intermediate value appears, changing each round)
   but does the guard still fire (false positive) even when there's no real convergence?
   Expect: with continually-changing wrong values, no candidate_confirmed and no
   verification_completed -> should NOT fire easily (or only via the weaker
   unverified_no_new_info_x2 path if the WRONG value repeats, which is arguably correct
   behavior -- the guard doesn't know it's wrong, it only knows convergence).
"""
import sys
sys.path.insert(0, "/home/ubuntu/antidoom_data")
from rumination_guard import GuardState

PROMPT = ("Using python_eval with SymPy, compute the definite integral of x^2 * ln(x) "
          "from x=1 to x=e, then verify by numerically integrating the same function "
          "over the same bounds and confirming the two match.")

results = []


def run_case(name, tool_outputs, expect_fire, expect_fire_round_le=None):
    gs = GuardState(PROMPT)
    fired_at = None
    for i, out in enumerate(tool_outputs):
        gs.observe_tool_round(out, i)
        if gs.fired and fired_at is None:
            fired_at = i
    ok = (fired_at is not None) == expect_fire
    if expect_fire and expect_fire_round_le is not None and fired_at is not None:
        ok = ok and (fired_at <= expect_fire_round_le)
    results.append((name, fired_at, expect_fire, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: fired_at={fired_at} (expected_fire={expect_fire})")


# --- Variant 1: correct exact, repeated verbatim (normal convergence case) ---
run_case(
    "V1_exact_repeat",
    ["Exact integral: 4.574563760717865", "Numeric verification: 4.574563760717865",
     "Confirmed: matches, verified.", "Still 4.574563760717865, no new info."],
    expect_fire=True,
)

# --- Variant 2: correct value, but rendered in DIFFERENT notations each time ---
run_case(
    "V2_notation_erf_exact_vs_decimal",
    ["erfi value component gives partial result", "0.485592252 approx via erfi(sqrt(2))",
     "0.4855922520 matches the decimal expansion", "Confirmed match: 0.485592252"],
    expect_fire=True,  # should still recognize convergence despite notation variety
)
run_case(
    "V2_notation_fraction_vs_decimal",
    ["pi/4 exactly", "0.7853981634 numerically", "0.78539816 matches pi/4", "Verified: consistent"],
    expect_fire=True,
)
run_case(
    "V2_notation_half",
    ["The answer is 1/2", "0.5 numerically", "0.500000 confirmed", "Matches: 0.5"],
    expect_fire=True,
)

# --- Variant 5: WRONG intermediate value that keeps CHANGING (never converges) ---
run_case(
    "V5_wrong_never_converges",
    ["First attempt: 3.14159", "Retry: 2.71828", "Another attempt: 1.61803",
     "Hmm let me try again: 5.29341", "Still not matching: 6.02214", "Trying once more: 8.31446"],
    expect_fire=False,  # values never repeat/converge -> no_new_info_streak never builds up
)
# --- Variant 5b: WRONG value that happens to repeat (guard doesn't know it's wrong) ---
run_case(
    "V5b_wrong_but_repeats",
    ["Computed: 99.9", "Rechecked: 99.9", "Confirmed: 99.9", "Still 99.9, no change"],
    expect_fire=True,  # guard has no ground truth -- convergence-on-a-wrong-value still
                        # looks like convergence; this is an INHERENT LIMIT of the timing-only
                        # guard design (it never supplies a value, so firing here just means
                        # "stop and answer with whatever you have," not "the wrong value is right")
)

n_pass = sum(1 for r in results if r[3])
print(f"\n{n_pass}/{len(results)} cases behaved as expected")
print("\nNote on V5b: guard fires on a CONSISTENTLY WRONG value because it has no oracle --")
print("this is expected and acceptable since the guard's action (nudge-only, no value supplied)")
print("just tells the model to stop and answer with its own belief, it doesn't endorse the value.")
